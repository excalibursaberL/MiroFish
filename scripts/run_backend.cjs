const { existsSync } = require('node:fs')
const { resolve } = require('node:path')
const { spawn, spawnSync } = require('node:child_process')

const projectRoot = resolve(__dirname, '..')
const backendRoot = resolve(projectRoot, 'backend')
const pythonCandidates = process.platform === 'win32'
  ? [resolve(backendRoot, '.venv', 'Scripts', 'python.exe')]
  : [resolve(backendRoot, '.venv', 'bin', 'python')]

const commandExists = (command) => {
  const probe = process.platform === 'win32' ? 'where.exe' : 'which'
  return spawnSync(probe, [command], { stdio: 'ignore' }).status === 0
}

let command
let args
const projectPython = pythonCandidates.find((candidate) => existsSync(candidate))

if (projectPython) {
  command = projectPython
  args = ['run.py']
  console.log(`[backend] using project virtual environment: ${projectPython}`)
} else if (commandExists('uv')) {
  command = 'uv'
  args = ['run', 'python', 'run.py']
  console.log('[backend] project .venv not found; using uv')
} else {
  command = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3')
  args = ['run.py']
  console.warn('[backend] project .venv and uv were not found; falling back to system Python')
}

const child = spawn(command, args, {
  cwd: backendRoot,
  stdio: 'inherit',
  env: process.env
})

let stopping = false

const stopBackendTree = (signal = 'SIGTERM') => {
  if (stopping || child.exitCode !== null) return
  stopping = true
  if (process.platform === 'win32') {
    spawnSync('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], {
      stdio: 'ignore'
    })
  } else {
    child.kill(signal)
  }
}

for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(signal, () => {
    stopBackendTree(signal)
    process.exitCode = 0
  })
}

process.on('exit', () => stopBackendTree())

const launcherParentPid = process.ppid
const parentWatchdog = setInterval(() => {
  try {
    process.kill(launcherParentPid, 0)
  } catch {
    stopBackendTree()
    process.exit(0)
  }
}, 1000)

child.on('error', (error) => {
  console.error(`[backend] failed to start: ${error.message}`)
  process.exitCode = 1
})

child.on('close', (code, signal) => {
  clearInterval(parentWatchdog)
  if (signal) {
    console.log(`[backend] stopped by ${signal}`)
  }
  process.exitCode = code ?? 1
})
