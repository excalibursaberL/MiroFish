"""Render the evidence for using round 6 as the provisional S1 baseline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1900
HEIGHT = 1120
SCALE = 2
SELECTED_ROUND = 6

BG = "#F5F7F8"
PANEL = "#FFFFFF"
INK = "#172126"
MUTED = "#627078"
GRID = "#DCE2E5"
TEAL = "#187C72"
CORAL = "#D45D45"
BLUE = "#356CA5"
AMBER = "#C88B24"
MARK = "#9D3D2F"


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=root
        / "Dataset"
        / "s1_round_selection_10rounds_k10_seed4004_v1"
        / "round_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "task" / "figs" / "s1_round6_baseline_evidence.png",
    )
    return parser.parse_args()


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size * SCALE)
    raise FileNotFoundError("No suitable TrueType font found")


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric: list[dict[str, float]] = []
    for row in rows:
        numeric.append({key: float(value) if value not in {"", None} else 0.0 for key, value in row.items()})
    return numeric


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font: ImageFont.FreeTypeFont, fill: str = INK) -> None:
    draw.text((xy[0] * SCALE, xy[1] * SCALE), value, font=font, fill=fill)


def rounded_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(tuple(value * SCALE for value in box), radius=radius * SCALE, fill=fill, outline=outline, width=1 * SCALE if outline else 1)


def line(draw: ImageDraw.ImageDraw, points: Sequence[tuple[float, float]], fill: str, width: int = 2) -> None:
    draw.line([(int(x * SCALE), int(y * SCALE)) for x, y in points], fill=fill, width=width * SCALE, joint="curve")


def circle(draw: ImageDraw.ImageDraw, center: tuple[float, float], radius: int, fill: str, outline: str | None = None) -> None:
    x, y = center
    draw.ellipse(
        ((x - radius) * SCALE, (y - radius) * SCALE, (x + radius) * SCALE, (y + radius) * SCALE),
        fill=fill,
        outline=outline,
        width=1 * SCALE if outline else 1,
    )


def panel_frame(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title_value: str, subtitle: str, fonts: dict[str, ImageFont.FreeTypeFont]) -> tuple[int, int, int, int]:
    rounded_rect(draw, box, 8, PANEL, GRID)
    x0, y0, x1, y1 = box
    text(draw, (x0 + 24, y0 + 22), title_value, fonts["panel"], INK)
    text(draw, (x0 + 24, y0 + 58), subtitle, fonts["small"], MUTED)
    return x0 + 64, y0 + 112, x1 - 30, y1 - 68


def chart_axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    y_ticks: Sequence[float],
    fonts: dict[str, ImageFont.FreeTypeFont],
    formatter,
) -> tuple[callable, callable]:
    x0, y0, x1, y1 = plot
    for tick in y_ticks:
        py = y1 - (tick - y_min) / (y_max - y_min) * (y1 - y0)
        line(draw, [(x0, py), (x1, py)], GRID, 1)
        label = formatter(tick)
        bbox = draw.textbbox((0, 0), label, font=fonts["axis"])
        label_width = (bbox[2] - bbox[0]) / SCALE
        text(draw, (int(x0 - label_width - 10), int(py - 9)), label, fonts["axis"], MUTED)
    for round_number in range(0, 11, 2):
        px = x0 + round_number / 10 * (x1 - x0)
        text(draw, (int(px - 6), y1 + 13), str(round_number), fonts["axis"], MUTED)
    line(draw, [(x0, y1), (x1, y1)], MUTED, 1)

    def x_map(value: float) -> float:
        return x0 + value / 10 * (x1 - x0)

    def y_map(value: float) -> float:
        return y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)

    selected_x = x_map(SELECTED_ROUND)
    line(draw, [(selected_x, y0), (selected_x, y1)], MARK, 2)
    text(draw, (int(selected_x - 28), y0 - 27), "第 6 轮", fonts["axis_bold"], MARK)
    return x_map, y_map


def legend(draw: ImageDraw.ImageDraw, x: int, y: int, items: Sequence[tuple[str, str]], fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    cursor = x
    for color, label in items:
        line(draw, [(cursor, y + 8), (cursor + 24, y + 8)], color, 3)
        text(draw, (cursor + 32, y - 2), label, fonts["legend"], MUTED)
        bbox = draw.textbbox((0, 0), label, font=fonts["legend"])
        cursor += 52 + int((bbox[2] - bbox[0]) / SCALE)


def render(metrics_path: Path, output_path: Path) -> None:
    rows = read_metrics(metrics_path)
    canvas = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
    draw = ImageDraw.Draw(canvas)
    fonts = {
        "title": load_font(34, bold=True),
        "subtitle": load_font(17),
        "stat": load_font(27, bold=True),
        "stat_label": load_font(14),
        "panel": load_font(21, bold=True),
        "small": load_font(13),
        "axis": load_font(12),
        "axis_bold": load_font(12, bold=True),
        "legend": load_font(12),
        "footer": load_font(14),
    }

    text(draw, (55, 40), "为什么暂定第 6 轮作为 S1 互动基准", fonts["title"])
    text(draw, (56, 91), "18 个匿名 A 股场景 · 每场景 10 个 Agent · 10 轮社会互动 · 本地随机种子 4004", fonts["subtitle"], MUTED)

    selected = rows[SELECTED_ROUND]
    final = rows[10]
    stats = [
        ("6", "暂定基准轮次"),
        (f"{int(selected['majority_correct_scenarios'])}/18", "场景多数方向正确"),
        (f"{selected['transition_mean_js']:.4f}", "相邻轮信念 JS（bits）"),
        (f"{int(selected['new_content_count'])}", "第 6 轮新增内容"),
    ]
    stat_x = [55, 320, 640, 1010]
    stat_w = [235, 290, 340, 290]
    for x, width, (value, label) in zip(stat_x, stat_w, stats):
        rounded_rect(draw, (x, 132, x + width, 224), 7, PANEL, GRID)
        text(draw, (x + 20, 148), value, fonts["stat"], TEAL if x != 1010 else AMBER)
        text(draw, (x + 20, 193), label, fonts["stat_label"], MUTED)
    rounded_rect(draw, (1330, 132, 1845, 224), 7, "#FFF7F3", "#EBC9BE")
    text(draw, (1350, 151), "暂定基准，不代表已经收敛", fonts["panel"], MARK)
    text(draw, (1350, 191), "第 7-10 轮仍有反弹，需用新种子复核", fonts["stat_label"], MUTED)

    boxes = [(55, 255, 640, 940), (657, 255, 1242, 940), (1259, 255, 1845, 940)]

    # Panel 1: prediction performance.
    plot = panel_frame(draw, boxes[0], "1  预测表现", "准确率没有单调上升，第 6 轮的场景多数结果最好", fonts)
    x_map, y_map = chart_axes(draw, plot, 0, 55, [0, 10, 20, 30, 40, 50], fonts, lambda v: f"{int(v)}%")
    individual = [(x_map(row["round"]), y_map(row["individual_direction_accuracy"] * 100)) for row in rows]
    majority = [(x_map(row["round"]), y_map(row["majority_correct_scenarios"] / 18 * 100)) for row in rows]
    baseline_y = y_map(8 / 18 * 100)
    for start in range(int(plot[0]), int(plot[2]), 14):
        line(draw, [(start, baseline_y), (min(start + 7, plot[2]), baseline_y)], MUTED, 1)
    line(draw, individual, TEAL, 3)
    line(draw, majority, CORAL, 3)
    for px, py in individual:
        circle(draw, (px, py), 3, TEAL)
    for px, py in majority:
        circle(draw, (px, py), 3, CORAL)
    circle(draw, individual[SELECTED_ROUND], 6, PANEL, TEAL)
    circle(draw, majority[SELECTED_ROUND], 6, PANEL, CORAL)
    legend(draw, boxes[0][0] + 26, boxes[0][3] - 47, [(TEAL, "个体准确率"), (CORAL, "场景多数正确/18"), (MUTED, "恒定下跌基线")], fonts)

    # Panel 2: transition JS with bars for newly authored content.
    plot = panel_frame(draw, boxes[1], "2  互动变化", "第 6 轮信念变化较低且新内容稀少，随后出现反弹", fonts)
    x_map, y_map = chart_axes(draw, plot, 0, 0.024, [0, 0.006, 0.012, 0.018, 0.024], fonts, lambda v: f"{v:.3f}")
    max_content = max(row["new_content_count"] for row in rows[1:])
    bar_width = (plot[2] - plot[0]) / 11 * 0.55
    for row in rows[1:]:
        x = x_map(row["round"])
        height = row["new_content_count"] / max_content * (plot[3] - plot[1]) * 0.48
        draw.rectangle(
            ((x - bar_width / 2) * SCALE, (plot[3] - height) * SCALE, (x + bar_width / 2) * SCALE, plot[3] * SCALE),
            fill="#E8C990" if int(row["round"]) != SELECTED_ROUND else AMBER,
        )
    js_points = [(x_map(row["round"]), y_map(row["transition_mean_js"])) for row in rows[1:]]
    line(draw, js_points, BLUE, 3)
    for px, py in js_points:
        circle(draw, (px, py), 3, BLUE)
    circle(draw, js_points[SELECTED_ROUND - 1], 6, PANEL, BLUE)
    legend(draw, boxes[1][0] + 26, boxes[1][3] - 47, [(BLUE, "相邻轮 JS"), (AMBER, "新增内容（相对柱高）")], fonts)

    # Panel 3: cumulative token scale.
    plot = panel_frame(draw, boxes[2], "3  计算规模", "第 10 轮累计 token 约为第 6 轮的 3.6 倍", fonts)
    max_tokens = max(row["cumulative_total_tokens"] for row in rows) / 1_000_000
    y_top = math.ceil(max_tokens / 50) * 50
    x_map, y_map = chart_axes(draw, plot, 0, y_top, [0, 50, 100, 150, 200, 250], fonts, lambda v: f"{int(v)}M")
    token_points = [(x_map(row["round"]), y_map(row["cumulative_total_tokens"] / 1_000_000)) for row in rows]
    polygon = [(token_points[0][0], plot[3]), *token_points, (token_points[-1][0], plot[3])]
    draw.polygon([(int(x * SCALE), int(y * SCALE)) for x, y in polygon], fill="#DDE8F2")
    line(draw, token_points, BLUE, 3)
    for px, py in token_points:
        circle(draw, (px, py), 3, BLUE)
    circle(draw, token_points[SELECTED_ROUND], 6, PANEL, MARK)
    text(draw, (int(token_points[SELECTED_ROUND][0] - 30), int(token_points[SELECTED_ROUND][1] - 38)), f"{selected['cumulative_total_tokens'] / 1_000_000:.1f}M", fonts["axis_bold"], MARK)
    text(draw, (int(token_points[10][0] - 57), int(token_points[10][1] - 36)), f"{final['cumulative_total_tokens'] / 1_000_000:.1f}M", fonts["axis_bold"], BLUE)
    legend(draw, boxes[2][0] + 26, boxes[2][3] - 47, [(BLUE, "累计 provider-reported tokens")], fonts)

    rounded_rect(draw, (55, 972, 1845, 1074), 7, "#EEF3F1", "#CBD8D3")
    text(draw, (78, 992), "读图结论", fonts["panel"], TEAL)
    text(draw, (225, 992), "第 6 轮兼顾了群体结果、新信息衰减和计算规模；第 9 轮虽有最高个体准确率，但没有稳定的群体优势。", fonts["footer"], INK)
    text(draw, (225, 1028), "限制：只有 1 个随机种子，18 个场景；后续必须在新种子和留出场景上验证，而不是继续按同一批结果挑轮次。", fonts["footer"], MUTED)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(output_path, optimize=True)


def main() -> None:
    args = parse_args()
    render(args.metrics.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
