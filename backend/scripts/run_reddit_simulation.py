"""
OASIS Reddit模拟预设脚本
此脚本读取配置文件中的参数来执行模拟，实现全程自动化

功能特性:
- 完成模拟后不立即关闭环境，进入等待命令模式
- 支持通过IPC接收Interview命令
- 支持单个Agent采访和批量采访
- 支持远程关闭环境命令

使用方式:
    python run_reddit_simulation.py --config /path/to/simulation_config.json
    python run_reddit_simulation.py --config /path/to/simulation_config.json --no-wait  # 完成后立即关闭
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import signal
import sys
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

# 全局变量：用于信号处理
_shutdown_event = None
_cleanup_done = False

# 添加项目路径
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# 加载项目根目录的 .env 文件（包含 LLM_API_KEY 等配置）
from dotenv import load_dotenv
from app.finance.token_usage import normalize_token_usage
from app.utils.openai_chat_compat import deepseek_v4_request_options

_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)


import re


class UnicodeFormatter(logging.Formatter):
    """自定义格式化器，将 Unicode 转义序列转换为可读字符"""
    
    UNICODE_ESCAPE_PATTERN = re.compile(r'\\u([0-9a-fA-F]{4})')
    
    def format(self, record):
        result = super().format(record)
        
        def replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return match.group(0)
        
        return self.UNICODE_ESCAPE_PATTERN.sub(replace_unicode, result)


class MaxTokensWarningFilter(logging.Filter):
    """过滤掉 camel-ai 关于 max_tokens 的警告（我们故意不设置 max_tokens，让模型自行决定）"""
    
    def filter(self, record):
        # 过滤掉包含 max_tokens 警告的日志
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


# 在模块加载时立即添加过滤器，确保在 camel 代码执行前生效
logging.getLogger().addFilter(MaxTokensWarningFilter())


def setup_oasis_logging(log_dir: str):
    """配置 OASIS 的日志，使用固定名称的日志文件"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 清理旧的日志文件
    for f in os.listdir(log_dir):
        old_log = os.path.join(log_dir, f)
        if os.path.isfile(old_log) and f.endswith('.log'):
            try:
                os.remove(old_log)
            except OSError:
                pass
    
    formatter = UnicodeFormatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
    
    loggers_config = {
        "social.agent": os.path.join(log_dir, "social.agent.log"),
        "social.twitter": os.path.join(log_dir, "social.twitter.log"),
        "social.rec": os.path.join(log_dir, "social.rec.log"),
        "oasis.env": os.path.join(log_dir, "oasis.env.log"),
        "table": os.path.join(log_dir, "table.log"),
    }
    
    for logger_name, log_file in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False


try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_reddit_agent_graph
    )
except ImportError as e:
    print(f"错误: 缺少依赖 {e}")
    print("请先安装: pip install oasis-ai camel-ai")
    sys.exit(1)


# IPC相关常量
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

class CommandType:
    """命令类型常量"""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class AgentTokenUsageRecorder:
    """Record provider-reported token usage for each OASIS Agent call."""

    def __init__(self, simulation_dir: str, agent_configs: List[Dict[str, Any]]):
        self.path = os.path.join(simulation_dir, "llm_token_usage.jsonl")
        self.phase = "other"
        self.round_number = None
        self.agent_metadata = {
            int(config.get("agent_id", index)): config
            for index, config in enumerate(agent_configs)
        }
        with open(self.path, "w", encoding="utf-8"):
            pass

    def set_context(self, phase: str, round_number: Optional[int] = None) -> None:
        self.phase = phase
        self.round_number = round_number

    @staticmethod
    def _response_usage(response: Any) -> Any:
        usage = getattr(response, "usage_dict", None)
        if usage is not None:
            return usage
        provider_response = getattr(response, "response", None)
        if isinstance(provider_response, dict):
            return provider_response.get("usage")
        return getattr(provider_response, "usage", None)

    @staticmethod
    def _response_model(response: Any, agent: Any) -> str:
        provider_response = getattr(response, "response", None)
        if isinstance(provider_response, dict):
            model = provider_response.get("model")
        else:
            model = getattr(provider_response, "model", None)
        if model:
            return str(model)
        backend = getattr(agent, "model_backend", None)
        return str(getattr(backend, "model_type", "") or "")

    def _append(
        self,
        *,
        agent_id: int,
        agent: Any,
        response: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        usage = normalize_token_usage(self._response_usage(response))
        metadata = self.agent_metadata.get(agent_id, {})
        record = {
            "recorded_at": datetime.now().isoformat(),
            "agent_id": agent_id,
            "full_population_agent_id": metadata.get(
                "full_population_agent_id"
            ),
            "agent_class": metadata.get("agent_class", ""),
            "phase": self.phase,
            "round": self.round_number,
            "model": self._response_model(response, agent),
            "usage_available": usage["usage_available"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "provider_usage": usage["provider_usage"],
            "status": "error" if error is not None else "ok",
            "error": str(error) if error is not None else None,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def instrument(self, agent_graph: Any) -> None:
        """Wrap every Agent model call without changing OASIS/CAMEL code."""
        for agent_id, agent in agent_graph.get_agents():
            if getattr(agent, "_finance_token_usage_instrumented", False):
                continue
            original = agent._aget_model_response

            async def tracked(
                *args,
                __agent_id=agent_id,
                __agent=agent,
                __original=original,
                **kwargs,
            ):
                try:
                    response = await __original(*args, **kwargs)
                except Exception as error:
                    self._append(
                        agent_id=__agent_id,
                        agent=__agent,
                        error=error,
                    )
                    raise
                self._append(
                    agent_id=__agent_id,
                    agent=__agent,
                    response=response,
                )
                return response

            agent._aget_model_response = tracked
            agent._finance_token_usage_instrumented = True


class IPCHandler:
    """IPC命令处理器"""
    
    def __init__(
        self,
        simulation_dir: str,
        env,
        agent_graph,
        token_recorder: Optional[AgentTokenUsageRecorder] = None,
        finance_mode: bool = False,
    ):
        self.simulation_dir = simulation_dir
        self.env = env
        self.agent_graph = agent_graph
        self.token_recorder = token_recorder
        self.finance_mode = finance_mode
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        self._running = True
        
        # 确保目录存在
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def update_status(self, status: str):
        """更新环境状态"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_command(self) -> Optional[Dict[str, Any]]:
        """轮询获取待处理命令"""
        if not os.path.exists(self.commands_dir):
            return None
        
        # 获取命令文件（按时间排序）
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        
        return None
    
    def send_response(self, command_id: str, status: str, result: Dict = None, error: str = None):
        """发送响应"""
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        # 删除命令文件
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    async def handle_interview(self, command_id: str, agent_id: int, prompt: str) -> bool:
        """
        处理单个Agent采访命令
        
        Returns:
            True 表示成功，False 表示失败
        """
        try:
            if self.token_recorder:
                self.token_recorder.set_context("manual_interview")
            # 获取Agent
            agent = self.agent_graph.get_agent(agent_id)
            
            # 创建Interview动作
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompt}
            )
            
            # 执行Interview
            actions = {agent: interview_action}
            await self.env.step(actions)
            
            # 从数据库获取结果
            result = self._get_interview_result(agent_id)
            
            self.send_response(command_id, "completed", result=result)
            print(f"  Interview完成: agent_id={agent_id}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"  Interview失败: agent_id={agent_id}, error={error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False
    
    async def handle_batch_interview(self, command_id: str, interviews: List[Dict]) -> bool:
        """
        处理批量采访命令
        
        Args:
            interviews: [{"agent_id": int, "prompt": str}, ...]
        """
        try:
            if self.token_recorder:
                self.token_recorder.set_context(
                    "post_social_prediction" if self.finance_mode else "manual_interview"
                )
            result = await self.execute_batch_interviews(interviews)
            self.send_response(command_id, "completed", result=result)
            results = result["results"]
            print(f"  批量Interview完成: {len(results)} 个Agent")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"  批量Interview失败: {error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False

    async def execute_batch_interviews(self, interviews: List[Dict]) -> Dict[str, Any]:
        """Execute a batch inside the live environment without using IPC files."""
        actions = {}
        agent_ids = []
        for interview in interviews:
            agent_id = int(interview.get("agent_id"))
            prompt = str(interview.get("prompt", ""))
            try:
                agent = self.agent_graph.get_agent(agent_id)
                actions[agent] = ManualAction(
                    action_type=ActionType.INTERVIEW,
                    action_args={"prompt": prompt},
                )
                agent_ids.append(agent_id)
            except Exception as error:
                print(f"  警告: 无法获取Agent {agent_id}: {error}")
        if not actions:
            raise ValueError("没有有效的Agent")
        await self.env.step(actions)
        results = {
            agent_id: self._get_interview_result(agent_id)
            for agent_id in agent_ids
        }
        return {"interviews_count": len(results), "results": results}
    
    def _get_interview_result(self, agent_id: int) -> Dict[str, Any]:
        """从数据库获取最新的Interview结果"""
        db_path = os.path.join(self.simulation_dir, "reddit_simulation.db")
        
        result = {
            "agent_id": agent_id,
            "response": None,
            "timestamp": None
        }
        
        if not os.path.exists(db_path):
            return result
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 查询最新的Interview记录
            cursor.execute("""
                SELECT user_id, info, created_at
                FROM trace
                WHERE action = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (ActionType.INTERVIEW.value, agent_id))
            
            row = cursor.fetchone()
            if row:
                user_id, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    result["response"] = info.get("response", info)
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json
            
            conn.close()
            
        except Exception as e:
            print(f"  读取Interview结果失败: {e}")
        
        return result
    
    async def process_commands(self) -> bool:
        """
        处理所有待处理命令
        
        Returns:
            True 表示继续运行，False 表示应该退出
        """
        command = self.poll_command()
        if not command:
            return True
        
        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})
        
        print(f"\n收到IPC命令: {command_type}, id={command_id}")
        
        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id,
                args.get("agent_id", 0),
                args.get("prompt", "")
            )
            return True
            
        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", [])
            )
            return True
            
        elif command_type == CommandType.CLOSE_ENV:
            print("收到关闭环境命令")
            self.send_response(command_id, "completed", result={"message": "环境即将关闭"})
            return False
        
        else:
            self.send_response(command_id, "failed", error=f"未知命令类型: {command_type}")
            return True


class RedditSimulationRunner:
    """Reddit模拟运行器"""
    
    # Reddit可用动作（不包含INTERVIEW，INTERVIEW只能通过ManualAction手动触发）
    AVAILABLE_ACTIONS = [
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_COMMENT,
        ActionType.DISLIKE_COMMENT,
        ActionType.SEARCH_POSTS,
        ActionType.SEARCH_USER,
        ActionType.TREND,
        ActionType.REFRESH,
        ActionType.DO_NOTHING,
        ActionType.FOLLOW,
        ActionType.MUTE,
    ]
    
    def __init__(self, config_path: str, wait_for_commands: bool = True):
        """
        初始化模拟运行器
        
        Args:
            config_path: 配置文件路径 (simulation_config.json)
            wait_for_commands: 模拟完成后是否等待命令（默认True）
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.simulation_dir = os.path.dirname(config_path)
        self.random_seed_state = self._apply_random_seed(
            self.config.get("random_seed", 4004)
        )
        self.wait_for_commands = wait_for_commands
        self.env = None
        self.agent_graph = None
        self.ipc_handler = None
        self.token_recorder = AgentTokenUsageRecorder(
            self.simulation_dir,
            self.config.get("agent_configs", []),
        )
        self.finance_s1 = self.config.get("finance_s1") or {}
        self._finance_action_count = 0
        self._finance_action_log = os.path.join(
            self.simulation_dir, "reddit", "actions.jsonl"
        )
        if self.finance_s1:
            os.makedirs(os.path.dirname(self._finance_action_log), exist_ok=True)
            with open(self._finance_action_log, "w", encoding="utf-8"):
                pass

    def _apply_random_seed(self, value: Any) -> Dict[str, Any]:
        """Seed every local random source used by the S1/OASIS process.

        Provider-side LLM sampling is deliberately reported separately: the
        configured DeepSeek endpoint does not expose a reproducible seed
        contract, so local scheduling can be fixed while model text may still
        vary between otherwise identical runs.
        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("random_seed must be an integer")
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("random_seed must be between 0 and 4294967295")

        os.environ["PYTHONHASHSEED"] = str(value)
        random.seed(value)
        report = {
            "random_seed": value,
            "python_random_seeded": True,
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
            "numpy_seeded": False,
            "torch_seeded": False,
            "llm_provider_seeded": False,
            "llm_provider_seed_note": (
                "DeepSeek API does not expose a deterministic seed contract; "
                "provider output can still vary."
            ),
        }
        try:
            import numpy as np

            np.random.seed(value)
            report["numpy_seeded"] = True
        except ImportError:
            report["numpy_seed_note"] = "NumPy is not installed"
        try:
            import torch

            torch.manual_seed(value)
            report["torch_seeded"] = True
        except ImportError:
            report["torch_seed_note"] = "PyTorch is not installed"

        with open(
            os.path.join(self.simulation_dir, "random_seed_state.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_profile_path(self) -> str:
        """获取Profile文件路径"""
        return os.path.join(self.simulation_dir, "reddit_profiles.json")
    
    def _get_db_path(self) -> str:
        """获取数据库路径"""
        return os.path.join(self.simulation_dir, "reddit_simulation.db")

    def _trace_max_rowid(self) -> int:
        """Return the committed OASIS trace boundary for round attribution."""
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return 0
        with sqlite3.connect(db_path) as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(rowid), 0) FROM trace"
            ).fetchone()
        return int(row[0] or 0) if row else 0

    def _log_finance_record(self, record: Dict[str, Any]) -> None:
        """Write the action/event contract consumed by SimulationRunner."""
        if not self.finance_s1:
            return
        with open(self._finance_action_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _is_valid_finance_forecast_response(result: Dict[str, Any]) -> bool:
        """Check enough of the JSON contract to retry before social turns."""
        raw = result.get("response") if isinstance(result, dict) else None
        text = raw.strip() if isinstance(raw, str) else ""
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                return False
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return False
        if not isinstance(payload, dict):
            return False
        probabilities = payload.get("probabilities", payload)
        if not isinstance(probabilities, dict):
            return False
        try:
            values = [
                float(probabilities.get(long_name, probabilities.get(short_name)))
                for long_name, short_name in (
                    ("up_probability", "up"),
                    ("neutral_probability", "neutral"),
                    ("down_probability", "down"),
                )
            ]
        except (TypeError, ValueError):
            return False
        return (
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
            and sum(values) > 0.0
        )

    async def _run_finance_belief_snapshot(self, round_number: int) -> Dict[str, Any]:
        """Measure every investor's current belief after one social round.

        These are private ``INTERVIEW`` actions.  They are deliberately
        excluded from ``social_actions.jsonl`` and are stored separately so a
        belief measurement cannot be mistaken for a social influence event.
        A failed measurement is persisted and does not stop the social run;
        the adapter later emits explicit ``status=missing`` rows per Agent.
        """
        finance = self.finance_s1 or {}
        if not finance.get("belief_snapshot_enabled", True):
            return {
                "round": round_number,
                "success": False,
                "error": "belief snapshots disabled",
                "results": {},
                "attempts": [],
                "attempt_count": 0,
            }
        templates = finance.get("round_belief_snapshot_interviews") or []
        interviews = []
        for item in templates:
            rendered = dict(item)
            rendered["prompt"] = str(item.get("prompt", "")).replace(
                "__ROUND_NUMBER__", str(round_number)
            )
            rendered["retry_prompt"] = str(item.get("retry_prompt", rendered["prompt"])).replace(
                "__ROUND_NUMBER__", str(round_number)
            )
            interviews.append(rendered)
        if not interviews:
            payload = {
                "round": round_number,
                "success": False,
                "error": "round_belief_snapshot_interviews is empty",
                "results": {},
                "attempts": [],
                "attempt_count": 0,
            }
        else:
            attempts: List[Dict[str, Any]] = []
            try:
                self.token_recorder.set_context(
                    "belief_snapshot", round_number
                )
                first = await self.ipc_handler.execute_batch_interviews(interviews)
                for result in first.get("results", {}).values():
                    if isinstance(result, dict):
                        result["attempt_count"] = 1
                attempts.append({"attempt": 1, "result": first})
                invalid_ids = {
                    int(agent_id)
                    for agent_id, result in first.get("results", {}).items()
                    if not self._is_valid_finance_forecast_response(result)
                }
                if invalid_ids:
                    retry_interviews = [
                        {
                            "agent_id": int(item["agent_id"]),
                            "prompt": item.get("retry_prompt") or item.get("prompt", ""),
                        }
                        for item in interviews
                        if int(item["agent_id"]) in invalid_ids
                    ]
                    retry = await self.ipc_handler.execute_batch_interviews(retry_interviews)
                    for result in retry.get("results", {}).values():
                        if isinstance(result, dict):
                            result["attempt_count"] = 2
                    attempts.append({"attempt": 2, "result": retry})
                    first["results"].update(retry.get("results", {}))
                payload = {
                    "round": round_number,
                    "success": True,
                    "results": first.get("results", {}),
                    "attempts": attempts,
                    "attempt_count": len(attempts),
                }
            except Exception as error:
                payload = {
                    "round": round_number,
                    "success": False,
                    "error": str(error),
                    "results": {},
                    "attempts": attempts,
                    "attempt_count": max(1, len(attempts)),
                }
        payload["timestamp"] = datetime.now().isoformat()
        snapshot_path = os.path.join(
            self.simulation_dir, "round_belief_interviews.jsonl"
        )
        with open(snapshot_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._log_finance_record(
            {
                "event_type": "belief_snapshot_completed",
                "round": round_number,
                "timestamp": payload["timestamp"],
                "success": payload["success"],
                "attempt_count": payload["attempt_count"],
            }
        )
        return payload

    def _create_model(self):
        """
        创建LLM模型
        
        统一使用项目根目录 .env 文件中的配置（优先级最高）：
        - LLM_API_KEY: API密钥
        - LLM_BASE_URL: API基础URL
        - LLM_MODEL_NAME: 模型名称
        """
        # 优先从 .env 读取配置
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        
        # 如果 .env 中没有，则使用 config 作为备用
        if not llm_model:
            llm_model = self.config.get("llm_model", "gpt-4o-mini")
        
        # 设置 camel-ai 所需的环境变量
        if llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("缺少 API Key 配置，请在项目根目录 .env 文件中设置 LLM_API_KEY")
        
        if llm_base_url:
            os.environ["OPENAI_API_BASE_URL"] = llm_base_url
        
        print(f"LLM配置: model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else '默认'}...")
        
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=llm_model,
            model_config_dict=deepseek_v4_request_options(
                llm_model,
                thinking_mode="disabled",
            ) or None,
        )
    
    def _get_active_agents_for_round(
        self, 
        env, 
        current_hour: int,
        round_num: int
    ) -> List:
        """
        根据时间和配置决定本轮激活哪些Agent
        """
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])
        
        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)
        
        peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
        off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
        
        if current_hour in peak_hours:
            multiplier = time_config.get("peak_activity_multiplier", 1.5)
        elif current_hour in off_peak_hours:
            multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
        else:
            multiplier = 1.0
        
        target_count = int(random.uniform(base_min, base_max) * multiplier)
        
        candidates = []
        for cfg in agent_configs:
            if self.finance_s1 and cfg.get("agent_class") != "investor":
                continue
            agent_id = cfg.get("agent_id", 0)
            active_hours = cfg.get("active_hours", list(range(8, 23)))
            activity_level = cfg.get("activity_level", 0.5)
            
            if current_hour not in active_hours:
                continue
            
            if random.random() < activity_level:
                candidates.append(agent_id)
        
        selected_ids = random.sample(
            candidates, 
            min(target_count, len(candidates))
        ) if candidates else []
        
        active_agents = []
        for agent_id in selected_ids:
            try:
                agent = env.agent_graph.get_agent(agent_id)
                active_agents.append((agent_id, agent))
            except Exception:
                pass
        
        return active_agents
    
    async def run(self, max_rounds: int = None):
        """运行Reddit模拟
        
        Args:
            max_rounds: 最大模拟轮数（可选，用于截断过长的模拟）
        """
        print("=" * 60)
        print("OASIS Reddit模拟")
        print(f"配置文件: {self.config_path}")
        print(f"模拟ID: {self.config.get('simulation_id', 'unknown')}")
        print(f"等待命令模式: {'启用' if self.wait_for_commands else '禁用'}")
        print("=" * 60)
        
        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = (total_hours * 60) // minutes_per_round
        
        # 如果指定了最大轮数，则截断
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                print(f"\n轮数已截断: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
        
        print(f"\n模拟参数:")
        print(f"  - 总模拟时长: {total_hours}小时")
        print(f"  - 每轮时间: {minutes_per_round}分钟")
        print(f"  - 总轮数: {total_rounds}")
        if max_rounds:
            print(f"  - 最大轮数限制: {max_rounds}")
        print(f"  - Agent数量: {len(self.config.get('agent_configs', []))}")
        
        print("\n初始化LLM模型...")
        model = self._create_model()
        
        print("加载Agent Profile...")
        profile_path = self._get_profile_path()
        if not os.path.exists(profile_path):
            print(f"错误: Profile文件不存在: {profile_path}")
            return
        
        self.agent_graph = await generate_reddit_agent_graph(
            profile_path=profile_path,
            model=model,
            available_actions=self.AVAILABLE_ACTIONS,
        )
        self.token_recorder.instrument(self.agent_graph)
        
        db_path = self._get_db_path()
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"已删除旧数据库: {db_path}")
        
        print("创建OASIS环境...")
        self.env = oasis.make(
            agent_graph=self.agent_graph,
            platform=oasis.DefaultPlatformType.REDDIT,
            database_path=db_path,
            semaphore=30,  # 限制最大并发 LLM 请求数，防止 API 过载
        )
        
        await self.env.reset()
        print("环境初始化完成\n")
        
        # 初始化IPC处理器
        self.ipc_handler = IPCHandler(
            self.simulation_dir,
            self.env,
            self.agent_graph,
            token_recorder=self.token_recorder,
            finance_mode=bool(self.finance_s1),
        )
        self.ipc_handler.update_status("running")

        if self.finance_s1:
            self._log_finance_record(
                {
                    "event_type": "simulation_start",
                    "round": 0,
                    "timestamp": datetime.now().isoformat(),
                    "scenario_id": self.finance_s1.get("scenario_id"),
                }
            )
        
        # 执行初始事件
        event_config = self.config.get("event_config", {})
        initial_posts = event_config.get("initial_posts", [])
        
        if initial_posts:
            print(f"执行初始事件 ({len(initial_posts)}条初始帖子)...")
            initial_actions = {}
            for post in initial_posts:
                agent_id = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    agent = self.env.agent_graph.get_agent(agent_id)
                    if agent in initial_actions:
                        if not isinstance(initial_actions[agent], list):
                            initial_actions[agent] = [initial_actions[agent]]
                        initial_actions[agent].append(ManualAction(
                            action_type=ActionType.CREATE_POST,
                            action_args={"content": content}
                        ))
                    else:
                        initial_actions[agent] = ManualAction(
                            action_type=ActionType.CREATE_POST,
                            action_args={"content": content}
                        )
                except Exception as e:
                    print(f"  警告: 无法为Agent {agent_id}创建初始帖子: {e}")
            
            if initial_actions:
                await self.env.step(initial_actions)
                print(f"  已发布 {len(initial_actions)} 条初始帖子")
                if self.finance_s1:
                    for post in initial_posts:
                        finance_event = post.get("finance_event") or {}
                        self._log_finance_record(
                            {
                                "event_type": "current_event_published",
                                "round": 0,
                                "timestamp": datetime.now().isoformat(),
                                "agent_id": post.get("poster_agent_id", 0),
                                "event_id": finance_event.get("event_id"),
                                "publisher_name": finance_event.get("publisher_name"),
                            }
                        )

        # Finance S1 needs a true baseline from the same Agent instances. The
        # private interview happens after the current event is public and
        # before any investor gets an LLM social-action turn.
        if self.finance_s1:
            pre_interviews = self.finance_s1.get("pre_social_interviews") or []
            if not pre_interviews:
                raise ValueError("finance_s1.pre_social_interviews is required")
            self._log_finance_record(
                {
                    "event_type": "pre_social_prediction_start",
                    "round": 0,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            pre_path = os.path.join(
                self.simulation_dir, "pre_social_interviews.json"
            )
            try:
                self.token_recorder.set_context("pre_social_prediction", 0)
                pre_result = await self.ipc_handler.execute_batch_interviews(
                    pre_interviews
                )
                for result in pre_result.get("results", {}).values():
                    if isinstance(result, dict):
                        result["attempt_count"] = 1
                attempts = [{"attempt": 1, "result": pre_result}]
                invalid_ids = {
                    int(agent_id)
                    for agent_id, result in pre_result.get("results", {}).items()
                    if not self._is_valid_finance_forecast_response(result)
                }
                if invalid_ids:
                    retry_interviews = [
                        {
                            "agent_id": int(item["agent_id"]),
                            "prompt": item.get("retry_prompt") or item["prompt"],
                        }
                        for item in pre_interviews
                        if int(item["agent_id"]) in invalid_ids
                    ]
                    retry_result = await self.ipc_handler.execute_batch_interviews(
                        retry_interviews
                    )
                    for result in retry_result.get("results", {}).values():
                        if isinstance(result, dict):
                            result["attempt_count"] = 2
                    attempts.append({"attempt": 2, "result": retry_result})
                    pre_result["results"].update(retry_result.get("results", {}))
                pre_payload = {
                    "success": True,
                    "stage": "pre_social",
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    **pre_result,
                }
            except Exception as error:
                pre_payload = {
                    "success": False,
                    "stage": "pre_social",
                    "attempt_count": 1,
                    "error": str(error),
                    "results": {},
                }
            with open(pre_path, "w", encoding="utf-8") as handle:
                json.dump(pre_payload, handle, ensure_ascii=False, indent=2)
            self._log_finance_record(
                {
                    "event_type": "pre_social_prediction_end",
                    "round": 0,
                    "timestamp": datetime.now().isoformat(),
                    "success": pre_payload["success"],
                    "prediction_count": len(pre_payload.get("results", {})),
                }
            )
            if not pre_payload["success"]:
                raise RuntimeError(pre_payload["error"])
        
        # 主模拟循环
        print("\n开始模拟循环...")
        start_time = datetime.now()
        
        for round_num in range(total_rounds):
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24
            simulated_day = simulated_minutes // (60 * 24) + 1
            trace_start_rowid = self._trace_max_rowid() if self.finance_s1 else 0

            if self.finance_s1:
                self._log_finance_record(
                    {
                        "event_type": "round_start",
                        "round": round_num + 1,
                        "simulated_hours": simulated_minutes / 60,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            
            active_agents = self._get_active_agents_for_round(
                self.env, simulated_hour, round_num
            )
            
            if not active_agents:
                if self.finance_s1:
                    self._log_finance_record(
                        {
                            "event_type": "round_end",
                            "round": round_num + 1,
                            "simulated_hours": simulated_minutes / 60,
                            "trace_start_rowid": trace_start_rowid,
                            "trace_end_rowid": trace_start_rowid,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    await self._run_finance_belief_snapshot(round_num + 1)
                continue
            
            actions = {
                agent: LLMAction()
                for _, agent in active_agents
            }

            self.token_recorder.set_context(
                "social_interaction", round_num + 1
            )
            await self.env.step(actions)
            trace_end_rowid = self._trace_max_rowid() if self.finance_s1 else 0

            if self.finance_s1:
                for agent_id, _agent in active_agents:
                    self._finance_action_count += 1
                    self._log_finance_record(
                        {
                            "round": round_num + 1,
                            "timestamp": datetime.now().isoformat(),
                            "platform": "reddit",
                            "agent_id": agent_id,
                            "agent_name": f"investor_{agent_id + 1:03d}",
                            "action_type": "LLM_ACTION",
                            "action_args": {},
                            "success": True,
                        }
                    )
                self._log_finance_record(
                    {
                        "event_type": "round_end",
                        "round": round_num + 1,
                        "simulated_hours": simulated_minutes / 60,
                        "trace_start_rowid": trace_start_rowid,
                        "trace_end_rowid": trace_end_rowid,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                await self._run_finance_belief_snapshot(round_num + 1)
            
            if (round_num + 1) % 10 == 0 or round_num == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (round_num + 1) / total_rounds * 100
                print(f"  [Day {simulated_day}, {simulated_hour:02d}:00] "
                      f"Round {round_num + 1}/{total_rounds} ({progress:.1f}%) "
                      f"- {len(active_agents)} agents active "
                      f"- elapsed: {elapsed:.1f}s")
        
        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n模拟循环完成!")
        print(f"  - 总耗时: {total_elapsed:.1f}秒")
        print(f"  - 数据库: {db_path}")

        if self.finance_s1:
            self._log_finance_record(
                {
                    "event_type": "simulation_end",
                    "round": total_rounds,
                    "total_rounds": total_rounds,
                    "total_actions": self._finance_action_count,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        
        # 是否进入等待命令模式
        if self.wait_for_commands:
            print("\n" + "=" * 60)
            print("进入等待命令模式 - 环境保持运行")
            print("支持的命令: interview, batch_interview, close_env")
            print("=" * 60)
            
            self.ipc_handler.update_status("alive")
            
            # 等待命令循环（使用全局 _shutdown_event）
            try:
                while not _shutdown_event.is_set():
                    should_continue = await self.ipc_handler.process_commands()
                    if not should_continue:
                        break
                    try:
                        await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                        break  # 收到退出信号
                    except asyncio.TimeoutError:
                        pass
            except KeyboardInterrupt:
                print("\n收到中断信号")
            except asyncio.CancelledError:
                print("\n任务被取消")
            except Exception as e:
                print(f"\n命令处理出错: {e}")
            
            print("\n关闭环境...")
        
        # 关闭环境
        self.ipc_handler.update_status("stopped")
        await self.env.close()
        
        print("环境已关闭")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='OASIS Reddit模拟')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='配置文件路径 (simulation_config.json)'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='最大模拟轮数（可选，用于截断过长的模拟）'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        default=False,
        help='模拟完成后立即关闭环境，不进入等待命令模式'
    )
    
    args = parser.parse_args()
    
    # 在 main 函数开始时创建 shutdown 事件
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 初始化日志配置（使用固定文件名，清理旧日志）
    simulation_dir = os.path.dirname(args.config) or "."
    setup_oasis_logging(os.path.join(simulation_dir, "log"))
    
    runner = RedditSimulationRunner(
        config_path=args.config,
        wait_for_commands=not args.no_wait
    )
    await runner.run(max_rounds=args.max_rounds)


def setup_signal_handlers():
    """
    设置信号处理器，确保收到 SIGTERM/SIGINT 时能够正确退出
    让程序有机会正常清理资源（关闭数据库、环境等）
    """
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n收到 {sig_name} 信号，正在退出...")
        if not _cleanup_done:
            _cleanup_done = True
            if _shutdown_event:
                _shutdown_event.set()
        else:
            # 重复收到信号才强制退出
            print("强制退出...")
            sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被中断")
    except SystemExit:
        pass
    finally:
        print("模拟进程已退出")

