from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import cv2
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps
from pptx import Presentation
from pptx.util import Inches


ROOT = Path(__file__).resolve().parents[2]
PPT_DIR = ROOT / "毕设冲刺" / "ppt"
OUT_DIR = PPT_DIR / "slides_png"
ASSET_DIR = PPT_DIR / "assets"
GEN_DIR = PPT_DIR / "generated"
FIG_DIR = PPT_DIR
THESIS_FIG = ROOT / "论文" / "figures"
TEMPLATE_LOGO = PPT_DIR / "template_reference" / "media" / "image1.png"
COVER_BG = GEN_DIR / "imagegen_cover_lite3_redwhite.png"
TEMPLATE_WORDMARK = PPT_DIR / "template_reference" / "media" / "image17.png"

W, H = 1920, 1080
SAFE = (76, 58, 1844, 1012)

COLORS = {
    "bg": "#F8F6F5",
    "paper": "#FFFFFF",
    "ink": "#171717",
    "muted": "#666666",
    "subtle": "#F4ECEA",
    "line": "#E8D6D4",
    "teal": "#B5121B",
    "cyan": "#C8242F",
    "green": "#A43A35",
    "orange": "#C87A2D",
    "red": "#B5121B",
    "navy": "#2B2424",
    "purple": "#7E1E25",
    "gold": "#B28A44",
    "deep_red": "#7D0008",
    "black": "#101010",
    "warm_gray": "#EFEAE8",
}

FONT_DIR = Path("C:/Windows/Fonts")
FONT_REG = FONT_DIR / "Noto Sans SC (TrueType).otf"
FONT_MED = FONT_DIR / "Noto Sans SC Medium (TrueType).otf"
FONT_BOLD = FONT_DIR / "Noto Sans SC Bold (TrueType).otf"
FONT_FALLBACK = FONT_DIR / "msyh.ttc"


def setup_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path = FONT_REG
    if weight == "bold":
        path = FONT_BOLD
    elif weight == "medium":
        path = FONT_MED
    if not path.exists():
        path = FONT_FALLBACK
    return ImageFont.truetype(str(path), size)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def with_alpha(value: str, alpha: int) -> tuple[int, int, int, int]:
    return (*hex_to_rgb(value), alpha)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_w: int, start: int, min_size: int = 16, weight: str = "regular") -> ImageFont.FreeTypeFont:
    size = start
    while size >= min_size:
        fnt = font(size, weight)
        if text_size(draw, text, fnt)[0] <= max_w:
            return fnt
        size -= 1
    return font(min_size, weight)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            lines.append("")
            continue
        cur = ""
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+\-_/=.]*|[，。；：、,.!?！？]+|\s+|.", raw)
        for token in tokens:
            if token.isspace() and not cur:
                continue
            trial = cur + token
            if text_size(draw, trial, fnt)[0] <= width or not cur:
                cur = trial
            elif text_size(draw, token, fnt)[0] <= width:
                lines.append(cur.rstrip())
                cur = token.lstrip()
            else:
                for ch in token:
                    trial = cur + ch
                    if text_size(draw, trial, fnt)[0] <= width or not cur:
                        cur = trial
                    else:
                        lines.append(cur.rstrip())
                        cur = ch
        if cur:
            lines.append(cur.rstrip())
    return lines


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    size: int,
    color: str = COLORS["ink"],
    weight: str = "regular",
    align: str = "left",
    line_spacing: float = 1.26,
    max_lines: int | None = None,
) -> int:
    x, y, w, h = box
    cur_size = size
    while cur_size >= 14:
        fnt = font(cur_size, weight)
        lines = wrap_text(draw, text, fnt, w)
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]
        line_h = int(cur_size * line_spacing)
        total_h = max(line_h, len(lines) * line_h)
        if total_h <= h:
            break
        cur_size -= 1
    fnt = font(cur_size, weight)
    lines = wrap_text(draw, text, fnt, w)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
    line_h = int(cur_size * line_spacing)
    cy = y
    for line in lines:
        tw, _ = text_size(draw, line, fnt)
        tx = x if align == "left" else x + (w - tw) // 2 if align == "center" else x + w - tw
        draw.text((tx, cy), line, font=fnt, fill=color)
        cy += line_h
    return cy


def rounded_rect(
    img: Image.Image,
    box: tuple[int, int, int, int],
    fill: str | tuple[int, int, int, int] = COLORS["paper"],
    outline: str | None = COLORS["line"],
    radius: int = 8,
    width: int = 2,
    shadow: bool = True,
) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x, y, w, h = box
    xy = (x, y, x + w, y + h)
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        sd.rounded_rectangle((x + 8, y + 12, x + w + 8, y + h + 12), radius=radius, fill=(33, 52, 64, 22))
        sh = sh.filter(ImageFilter.GaussianBlur(12))
        img.alpha_composite(sh)
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    img.alpha_composite(layer)


def draw_chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, color: str = "#FFFFFF", px: int = 18, py: int = 8, size: int = 22) -> tuple[int, int, int, int]:
    fnt = font(size, "medium")
    tw, th = text_size(draw, text, fnt)
    x, y = xy
    box = (x, y, x + tw + 2 * px, y + th + 2 * py)
    draw.rounded_rectangle(box, radius=6, fill=fill)
    draw.text((x + px, y + py - 2), text, font=fnt, fill=color)
    return box


def add_header(draw: ImageDraw.ImageDraw, page: int, title: str, section: str = "", accent: str = COLORS["teal"]) -> None:
    x1, y1, x2, _ = SAFE
    draw.rectangle((x1, 52, x1 + 54, 96), fill=accent)
    draw.text((x1 + 10, 59), f"{page:02d}", font=font(25, "bold"), fill="#FFFFFF")
    draw.rectangle((x1 + 72, 66, x1 + 160, 75), fill=accent)
    draw.text((x1 + 184, 49), title, font=font(32, "bold"), fill=COLORS["ink"])
    if section:
        fnt = font(20, "medium")
        tw, _ = text_size(draw, section, fnt)
        draw.text((x2 - tw, 58), section, font=fnt, fill=accent)
    draw.line((x1, 112, x2, 112), fill=COLORS["line"], width=2)


def add_footer(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle((76, 1022, 1844, 1026), fill=COLORS["deep_red"])
    draw.text((76, 1034), "基于动作扩散策略的四足机器人视觉目标导航", font=font(17, "regular"), fill="#8B7774")
    right = "陈样 · 上海交通大学 · 2026"
    fnt = font(18, "regular")
    tw, _ = text_size(draw, right, fnt)
    draw.text((1844 - tw, 1034), right, font=fnt, fill="#8B7774")


def new_slide(accent: str = COLORS["teal"]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (W, H), COLORS["bg"])
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, H), fill=COLORS["bg"])
    d.rectangle((0, 0, W, 18), fill=accent)
    d.polygon([(0, 18), (350, 18), (0, 330)], fill="#FFFFFF")
    d.line((35, 160, 220, 18), fill=with_alpha(accent, 90), width=2)
    d.line((1630, 0, 1450, 280), fill=with_alpha(accent, 70), width=2)
    d.rectangle((0, H - 18, W, H), fill="#FFFFFF")
    return img, d


def paste_image(
    base: Image.Image,
    path: Path,
    box: tuple[int, int, int, int],
    mode: str = "contain",
    radius: int = 8,
    border: bool = True,
    bg: str = "#FFFFFF",
) -> bool:
    if not path.exists():
        return False
    try:
        src = Image.open(path).convert("RGBA")
    except Exception:
        return False
    x, y, w, h = box
    if mode == "cover":
        src = ImageOps.fit(src, (w, h), method=Image.Resampling.LANCZOS)
    else:
        src.thumbnail((w - 18, h - 18), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (w, h), bg)
        canvas.alpha_composite(src, ((w - src.width) // 2, (h - src.height) // 2))
        src = canvas
    if radius:
        mask = Image.new("L", (w, h), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
        alpha = ImageChops.multiply(src.getchannel("A"), mask)
        src.putalpha(alpha)
        base.alpha_composite(src, (x, y))
    else:
        base.alpha_composite(src, (x, y))
    if border:
        ImageDraw.Draw(base).rounded_rectangle((x, y, x + w, y + h), radius=radius, outline=COLORS["line"], width=2)
    return True


def draw_metric(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], value: str, label: str, accent: str, sub: str = "") -> None:
    x, y, w, h = box
    draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill="#FFFFFF", outline=COLORS["line"], width=2)
    draw.rectangle((x, y, x + 9, y + h), fill=accent)
    draw.text((x + 28, y + 18), value, font=fit_font(draw, value, w - 56, 42, 26, "bold"), fill=accent)
    draw_text_box(draw, (x + 28, y + 70, w - 52, 58), label, 22, COLORS["ink"], "medium", line_spacing=1.16)
    if sub:
        draw_text_box(draw, (x + 28, y + 126, w - 52, h - 132), sub, 17, COLORS["muted"], line_spacing=1.18)


def draw_meta_box(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    accent: str = COLORS["red"],
    title_size: int = 26,
    body_size: int = 20,
    tag: str | None = None,
) -> None:
    """Draw a template-style title-plus-content text module."""
    rounded_rect(img, box, fill="#FFFFFF", outline=accent, shadow=True)
    x, y, w, h = box
    draw.rectangle((x, y, x + 8, y + h), fill=accent)
    tx = x + 28
    if tag:
        draw.rounded_rectangle((tx, y + 24, tx + 82, y + 58), radius=4, fill=accent)
        draw.text((tx + 19, y + 27), tag, font=font(18, "medium"), fill="#FFFFFF")
        tx += 104
    draw.text((tx, y + 22), title, font=font(title_size, "bold"), fill=accent)
    draw_text_box(draw, (x + 28, y + 70, w - 56, h - 90), body, body_size, COLORS["muted"], line_spacing=1.18)


def draw_bullets(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], items: Iterable[str], accent: str, size: int = 24) -> None:
    x, y, w, h = box
    cy = y
    for item in items:
        draw.rounded_rectangle((x, cy + 8, x + 12, cy + 20), radius=3, fill=accent)
        cy = draw_text_box(draw, (x + 28, cy, w - 28, h - (cy - y)), item, size, COLORS["ink"], "regular", line_spacing=1.18)
        cy += 18


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str, width: int = 5) -> None:
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    l = 20
    a = 0.48
    p1 = (x2 - l * math.cos(ang - a), y2 - l * math.sin(ang - a))
    p2 = (x2 - l * math.cos(ang + a), y2 - l * math.sin(ang + a))
    draw.polygon((end, p1, p2), fill=color)


def draw_video_placeholder(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    caption: str,
    thumb: Path | None = None,
    accent: str = COLORS["orange"],
) -> None:
    x, y, w, h = box
    rounded_rect(base, box, fill="#101820", outline=accent, shadow=True)
    if thumb and thumb.exists():
        paste_image(base, thumb, (x + 10, y + 10, w - 20, h - 74), mode="cover", radius=7, border=False)
        overlay = Image.new("RGBA", (w - 20, h - 74), (0, 0, 0, 52))
        base.alpha_composite(overlay, (x + 10, y + 10))
    else:
        draw.rectangle((x + 10, y + 10, x + w - 10, y + h - 74), fill="#23313B")
        for i in range(0, w, 44):
            draw.line((x + i, y + 10, x + i - 90, y + h - 74), fill="#2E414D", width=2)
    cx, cy = x + w // 2, y + (h - 74) // 2
    draw.ellipse((cx - 48, cy - 48, cx + 48, cy + 48), fill=with_alpha("#FFFFFF", 218))
    draw.polygon(((cx - 12, cy - 26), (cx - 12, cy + 26), (cx + 34, cy)), fill=accent)
    draw.rectangle((x, y + h - 64, x + w, y + h), fill="#FFFFFF")
    draw.text((x + 22, y + h - 55), title, font=font(22, "bold"), fill=COLORS["ink"])
    draw.text((x + 22, y + h - 27), caption, font=font(16, "regular"), fill=COLORS["muted"])
    draw.rounded_rectangle((x + w - 126, y + h - 45, x + w - 20, y + h - 16), radius=5, fill=accent)
    draw.text((x + w - 108, y + h - 43), "视频占位", font=font(15, "medium"), fill="#FFFFFF")


def extract_video_thumb(video_path: Path, out_path: Path, frame_ratio: float = 0.35) -> bool:
    if out_path.exists():
        return True
    if not video_path.exists():
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, int(total * frame_ratio))))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(frame).save(out_path)
    return True


def make_placeholder_image(path: Path, title: str, accent: str = COLORS["teal"]) -> None:
    img = Image.new("RGBA", (960, 540), "#E7EEF2")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((24, 24, 936, 516), radius=8, fill="#FFFFFF", outline=COLORS["line"], width=3)
    d.text((60, 60), title, font=font(42, "bold"), fill=accent)
    d.text((60, 128), "素材占位", font=font(26, "medium"), fill=COLORS["muted"])
    for i in range(5):
        y = 210 + i * 42
        d.rounded_rectangle((80, y, 880 - i * 38, y + 18), radius=4, fill="#C9D8DF")
    img.save(path)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def get_assets() -> dict[str, Path]:
    curated = PPT_DIR / "figures"

    def first_existing(*candidates: Path) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    paths = {
        "theme": GEN_DIR / "imagegen_theme_base.png",
        "cover": COVER_BG,
        "logo": TEMPLATE_LOGO,
        "intro": first_existing(curated / "intro_background.png", THESIS_FIG / "intro-background-placeholder.png"),
        "structure": first_existing(curated / "chapter_structure.png", THESIS_FIG / "structure.png"),
        "nomad": first_existing(curated / "algo_overview.png", THESIS_FIG / "algo-overview-placeholder.png", THESIS_FIG / "nomad.png"),
        "ddim": first_existing(curated / "ddim_comparison.png", THESIS_FIG / "ddim_comparison.png"),
        "tts": first_existing(curated / "tts_tradeoff.png", THESIS_FIG / "tts_combo_tradeoff_20260421.png"),
        "layered": first_existing(curated / "layered_system.png", THESIS_FIG / "fenceng.png"),
        "mujoco": first_existing(curated / "mujoco.png", THESIS_FIG / "mujoco.png"),
        "dual": first_existing(curated / "dual_chain.png", THESIS_FIG / "shuanglianlu.png"),
        "map": first_existing(curated / "topomap_overview.png", THESIS_FIG / "map.png"),
        "real": first_existing(curated / "real_robot.png", THESIS_FIG / "real-robot-eg.png"),
        "outdoor": first_existing(curated / "outdoor.png", THESIS_FIG / "shiwai.png"),
        "map_start": first_existing(curated / "map_lab_start.png", ROOT / "毕设冲刺" / "maps" / "lab_cori" / "000.png"),
        "map_mid": first_existing(curated / "map_lab_mid.png", ROOT / "毕设冲刺" / "maps" / "lab_cori" / "020.png"),
        "map_goal": first_existing(curated / "map_lab_goal.png", ROOT / "毕设冲刺" / "maps" / "lab_cori" / "039.png"),
        "map_outdoor": first_existing(curated / "map_outdoor_mid.png", ROOT / "毕设冲刺" / "maps" / "082.png"),
        "video_fast": ROOT / "毕设冲刺" / "video" / "ddim2cfg0tts8" / "videos" / "19700101_084701_lite3_real_interactive_navigate" / "navigation_record.mp4",
        "video_slope": ROOT / "毕设冲刺" / "video" / "二号楼楼下斜坡1" / "videos" / "20260426_154600_lite3_real_interactive_navigate" / "navigation_record.mp4",
        "video_grass": ROOT / "毕设冲刺" / "video" / "电草，失败" / "videos" / "20260426_161031_lite3_real_interactive_navigate" / "navigation_record.mp4",
    }
    for key in ["intro", "structure", "nomad", "ddim", "tts", "layered", "mujoco", "dual", "map", "real", "outdoor"]:
        if not paths[key].exists():
            make_placeholder_image(ASSET_DIR / f"placeholder_{key}.png", key, COLORS["teal"])
            paths[key] = ASSET_DIR / f"placeholder_{key}.png"
    thumbs = {
        "thumb_fast": (paths["video_fast"], ASSET_DIR / "video_thumb_fast.png", 0.40),
        "thumb_slope": (paths["video_slope"], ASSET_DIR / "video_thumb_slope.png", 0.32),
        "thumb_grass": (paths["video_grass"], ASSET_DIR / "video_thumb_grass.png", 0.28),
    }
    for key, (video, out, ratio) in thumbs.items():
        ok = extract_video_thumb(video, out, ratio)
        if not ok:
            fallback = paths.get("real" if key != "thumb_grass" else "outdoor")
            if fallback and fallback.exists():
                try:
                    img = Image.open(fallback).convert("RGB")
                    img = ImageOps.fit(img, (960, 540), method=Image.Resampling.LANCZOS)
                    img.save(out)
                    ok = True
                except Exception:
                    pass
        paths[key] = out if ok else Path()
    return paths


def slide_01(assets: dict[str, Path]) -> Image.Image:
    img = Image.new("RGBA", (W, H), "#F8F6F5")
    d = ImageDraw.Draw(img)
    if assets["cover"].exists():
        paste_image(img, assets["cover"], (0, 0, W, H), mode="cover", radius=0, border=False)
    else:
        paste_image(img, assets["intro"], (0, 0, W, H), mode="cover", radius=0, border=False)
        img.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, 168)))
    d.rectangle((0, 0, W, 18), fill=COLORS["red"])
    d.rectangle((76, 92, 950, 960), fill=with_alpha("#FFFFFF", 232))
    d.rectangle((76, 92, 90, 960), fill=COLORS["red"])
    d.line((114, 128, 900, 128), fill=with_alpha(COLORS["red"], 120), width=2)
    if assets["logo"].exists():
        paste_image(img, assets["logo"], (128, 132, 84, 84), mode="contain", radius=0, border=False, bg=(255, 255, 255, 0))
    d.text((236, 134), "上海交通大学本科毕业设计答辩", font=font(32, "medium"), fill=COLORS["deep_red"])
    d.text((128, 292), "基于动作扩散策略的", font=font(55, "bold"), fill=COLORS["ink"])
    d.text((128, 366), "四足机器人视觉目标导航", font=font(66, "bold"), fill=COLORS["red"])
    d.rectangle((132, 480, 755, 490), fill=COLORS["gold"])
    meta = [
        ("答辩人", "陈样"),
        ("学号", "522031910056"),
        ("指导教师", "雷华明"),
        ("专业", "智能感知工程"),
    ]
    x, y = 132, 560
    for i, (k, v) in enumerate(meta):
        yy = y + i * 70
        d.text((x, yy), k, font=font(24, "medium"), fill=COLORS["muted"])
        d.text((x + 140, yy - 4), v, font=font(31, "bold"), fill=COLORS["ink"])
    d.rounded_rectangle((132, 870, 340, 920), radius=4, fill=COLORS["red"])
    d.text((168, 879), "2026年5月", font=font(24, "medium"), fill="#FFFFFF")
    tag = "NoMaD · DDIM · TTS · Lite3"
    tag_font = font(30, "bold")
    tag_w, _ = text_size(d, tag, tag_font)
    d.text((1844 - tag_w, 952), tag, font=tag_font, fill=COLORS["deep_red"])
    return img


def slide_02(assets: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["cyan"])
    add_header(d, 2, "研究背景：视觉目标导航需要从环境中闭环决策", "背景与问题", COLORS["cyan"])
    paste_image(img, assets["intro"], (1090, 154, 700, 430), mode="contain", radius=8)
    d.text((118, 166), "从“看见目标”到“走到目标”", font=font(46, "bold"), fill=COLORS["ink"])
    draw_text_box(
        d,
        (120, 235, 820, 96),
        "服务机器人、巡检机器人与灾害救援场景都要求机器人在无显式地图或弱地图条件下，利用第一视角视觉持续估计下一步动作。",
        27,
        COLORS["muted"],
        line_spacing=1.24,
    )
    nodes = [
        ((156, 430, 360, 170), "视觉输入", "RGB 图像、目标图像、历史观测", COLORS["cyan"]),
        ((585, 370, 360, 170), "策略推理", "从数据中学习动作分布，而非手工几何规划", COLORS["teal"]),
        ((360, 640, 385, 170), "机器人执行", "速度指令、姿态稳定、地面约束", COLORS["orange"]),
    ]
    centers = []
    for box, title, body, color in nodes:
        x, y, w, h = box
        draw_meta_box(img, d, box, title, body, color, 29, 20)
        centers.append((x + w // 2, y + h // 2))
    arrow(d, (centers[0][0] + 180, centers[0][1] - 18), (centers[1][0] - 188, centers[1][1] + 12), COLORS["line"], 4)
    arrow(d, (centers[1][0] - 42, centers[1][1] + 92), (centers[2][0] + 120, centers[2][1] - 78), COLORS["line"], 4)
    arrow(d, (centers[2][0] - 170, centers[2][1] - 70), (centers[0][0] - 70, centers[0][1] + 100), COLORS["line"], 4)
    draw_meta_box(
        img,
        d,
        (1070, 650, 720, 222),
        "本文关注的核心",
        "以 NoMaD 为高层视觉策略生成局部动作；在四足机器人 Lite3 上构建可运行闭环。",
        COLORS["red"],
        30,
        24,
    )
    add_footer(d)
    return img


def slide_03(_: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["orange"])
    add_header(d, 3, "问题定义：把通用视觉导航策略接到真实四足平台", "研究挑战", COLORS["orange"])
    d.text((150, 166), "核心问题", font=font(33, "medium"), fill=COLORS["muted"])
    d.text((150, 214), "如何在算力受限、动力学复杂、环境变化明显的条件下，", font=font(42, "bold"), fill=COLORS["ink"])
    d.text((150, 274), "让四足机器人依据目标图像稳定完成视觉导航？", font=font(42, "bold"), fill=COLORS["orange"])
    cards = [
        ((118, 430, 500, 260), "算法侧", "扩散策略采样次数多；目标引导强度与实时性存在张力；候选动作需要快速筛选。", "01", COLORS["cyan"]),
        ((710, 382, 500, 260), "系统侧", "视觉模型输出为局部轨迹，四足底层接口需要速度与姿态指令，必须补齐中间控制链。", "02", COLORS["teal"]),
        ((1302, 430, 500, 260), "实验侧", "仿真能够验证闭环接口，但真实走廊、室外光照和地形才暴露泛化边界。", "03", COLORS["orange"]),
    ]
    for box, title, body, idx, color in cards:
        draw_meta_box(img, d, box, title, body, color, 32, 23, idx)
    y = 775
    d.rounded_rectangle((144, y, 1776, y + 112), radius=8, fill="#FFFFFF", outline=COLORS["line"], width=2)
    stages = [("目标图像", COLORS["cyan"]), ("历史观测", COLORS["cyan"]), ("扩散动作", COLORS["teal"]), ("PD映射", COLORS["orange"]), ("Lite3执行", COLORS["green"])]
    x = 220
    for i, (label, color) in enumerate(stages):
        d.rounded_rectangle((x, y + 31, x + 205, y + 81), radius=6, fill=color)
        d.text((x + 40, y + 42), label, font=font(22, "medium"), fill="#FFFFFF")
        if i < len(stages) - 1:
            arrow(d, (x + 218, y + 56), (x + 282, y + 56), COLORS["line"], 4)
        x += 310
    d.text((144, 918), "研究目标：在不改变 Lite3 底层稳定控制的前提下，构建可解释、可替换、可实测的高层视觉导航系统。", font=font(27, "medium"), fill=COLORS["navy"])
    add_footer(d)
    return img


def slide_04(assets: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["teal"])
    add_header(d, 4, "技术路线：算法优化、系统闭环与真实验证递进展开", "总体方案", COLORS["teal"])
    rounded_rect(img, (96, 148, 1015, 690), fill="#FFFFFF", outline=COLORS["line"])
    paste_image(img, assets["structure"], (126, 176, 955, 630), mode="contain", radius=6, border=False)
    right_x = 1170
    steps = [
        ("理论基础", "视觉目标导航、扩散模型、NoMaD、Lite3平台"),
        ("算法设计", "DDIM、CFG、TTS、视觉编码器替换"),
        ("仿真系统", "四层闭环架构与 MuJoCo 健康检查"),
        ("真实机器人", "Orin-Lite3 双链路、拓扑图导航、室内外验证"),
    ]
    y = 160
    for i, (title, body) in enumerate(steps):
        color = [COLORS["cyan"], COLORS["teal"], COLORS["orange"], COLORS["green"]][i]
        d.ellipse((right_x, y + 10, right_x + 54, y + 64), fill=color)
        d.text((right_x + 15, y + 17), str(i + 1), font=font(26, "bold"), fill="#FFFFFF")
        d.text((right_x + 80, y), title, font=font(30, "bold"), fill=COLORS["ink"])
        draw_text_box(d, (right_x + 82, y + 44, 570, 72), body, 22, COLORS["muted"], line_spacing=1.18)
        if i < len(steps) - 1:
            d.line((right_x + 27, y + 72, right_x + 27, y + 137), fill=COLORS["line"], width=4)
        y += 170
    d.rounded_rectangle((120, 875, 1800, 950), radius=8, fill="#FFFFFF", outline=COLORS["teal"], width=2)
    d.text((160, 895), "叙事主线", font=font(25, "bold"), fill=COLORS["teal"])
    d.text((300, 895), "先解决扩散策略实时性，再解决策略到四足平台的控制接口，最终用真实走廊与室外场景验证可用性与边界。", font=font(25, "medium"), fill=COLORS["ink"])
    add_footer(d)
    return img


def slide_05(assets: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["purple"])
    add_header(d, 5, "算法基线：NoMaD 将目标图像导航建模为动作扩散", "算法基础", COLORS["purple"])
    rounded_rect(img, (100, 148, 1080, 615), fill="#FFFFFF", outline=COLORS["line"])
    paste_image(img, assets["nomad"], (130, 178, 1020, 555), mode="contain", radius=6, border=False)
    side = (1245, 156, 555, 610)
    rounded_rect(img, side, fill="#FFFFFF", outline=COLORS["purple"])
    d.text((1286, 192), "输入与输出", font=font(33, "bold"), fill=COLORS["purple"])
    draw_bullets(
        d,
        (1288, 252, 476, 250),
        ["输入：历史观测图像、目标图像、目标掩码", "输出：局部动作轨迹与距离预测", "目标：沿拓扑图逐步逼近视觉目标"],
        COLORS["purple"],
        23,
    )
    draw_metric(d, (1288, 560, 220, 142), "19.05M", "模型参数量", COLORS["purple"], "可在边缘计算平台部署")
    draw_metric(d, (1532, 560, 220, 142), "8动作", "默认候选数", COLORS["teal"], "TTS 选择最优轨迹")
    bottom = [
        ((130, 820, 390, 124), "距离预测", "为拓扑节点切换提供局部进度信号", COLORS["cyan"]),
        ((580, 820, 390, 124), "扩散去噪", "从噪声轨迹迭代生成可执行动作", COLORS["purple"]),
        ((1030, 820, 390, 124), "目标掩码", "支持有目标导航与无目标探索两种模式", COLORS["orange"]),
        ((1480, 820, 290, 124), "四足适配", "仅保留前向与偏航命令", COLORS["green"]),
    ]
    for box, title, body, color in bottom:
        draw_meta_box(img, d, box, title, body, color, 24, 18)
    add_footer(d)
    return img


def slide_06(_: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["teal"])
    add_header(d, 6, "算法改进：统一推理模块支持速度、引导与候选选择", "核心优化", COLORS["teal"])
    rounded_rect(img, (740, 312, 445, 230), fill="#FFFFFF", outline=COLORS["teal"])
    d.text((810, 360), "统一推理模块", font=font(39, "bold"), fill=COLORS["teal"])
    draw_text_box(d, (790, 425, 340, 64), "以相同接口切换采样器、引导权重、候选数量与视觉编码器", 22, COLORS["muted"], align="center")
    cards = [
        ((118, 168, 548, 238), "DDIM 加速", "将随机多步采样改为确定性少步采样；默认 DDIM-2 显著压缩纯采样时间。", COLORS["cyan"], "速度"),
        ((1256, 168, 548, 238), "CFG 目标引导", "无条件与有条件分支组合，调节目标图像对动作生成的影响强度。", COLORS["orange"], "引导"),
        ((118, 612, 548, 238), "TTS 候选选择", "一次生成多条轨迹，用距离预测器挑选最接近目标的候选，低成本提高稳定性。", COLORS["green"], "选择"),
        ((1256, 612, 548, 238), "视觉编码器替换", "抽象 EfficientNet、DINOv2、ConvNeXt、ResNet 接口，比较速度与泛化潜力。", COLORS["purple"], "表征"),
    ]
    for box, title, body, color, tag in cards:
        draw_meta_box(img, d, box, title, body, color, 31, 21, tag)
    arrow(d, (666, 285), (740, 373), COLORS["line"], 4)
    arrow(d, (1256, 285), (1185, 373), COLORS["line"], 4)
    arrow(d, (666, 733), (740, 490), COLORS["line"], 4)
    arrow(d, (1256, 733), (1185, 490), COLORS["line"], 4)
    d.rounded_rectangle((308, 905, 1612, 956), radius=8, fill="#FFFFFF", outline=COLORS["teal"], width=2)
    d.text((356, 916), "最终默认配置：EfficientNet-B0 + DDIM-2 + CFG=0 + TTS-8", font=font(27, "bold"), fill=COLORS["teal"])
    add_footer(d)
    return img


def slide_07(assets: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["cyan"])
    add_header(d, 7, "离线消融：DDIM 给出主要实时性收益，TTS 提供低成本筛选", "算法实验", COLORS["cyan"])
    rounded_rect(img, (95, 148, 825, 450), fill="#FFFFFF", outline=COLORS["line"])
    paste_image(img, assets["ddim"], (120, 172, 775, 400), mode="contain", radius=6, border=False)
    rounded_rect(img, (965, 148, 825, 450), fill="#FFFFFF", outline=COLORS["line"])
    paste_image(img, assets["tts"], (990, 172, 775, 400), mode="contain", radius=6, border=False)
    draw_metric(d, (118, 650, 350, 170), "4.84×", "DDIM-2 纯采样加速", COLORS["cyan"], "相对 DDPM-10；采样约由 40ms 降至 8ms 量级")
    draw_metric(d, (518, 650, 350, 170), "6.80Hz", "真实机器人最快高层频率", COLORS["teal"], "默认配置 DDIM-2 / CFG=0 / TTS=8")
    draw_metric(d, (918, 650, 350, 170), "CFG≈2×", "引导分支推理开销", COLORS["orange"], "收益不稳定，真实系统默认关闭")
    draw_metric(d, (1318, 650, 350, 170), "TTS-8", "候选筛选折中点", COLORS["green"], "保留多样性，同时避免过高延迟")
    d.rounded_rectangle((140, 886, 1780, 952), radius=8, fill="#FFFFFF", outline=COLORS["cyan"], width=2)
    d.text((184, 903), "结论", font=font(27, "bold"), fill=COLORS["cyan"])
    d.text((282, 903), "速度瓶颈首先来自扩散采样；在真实平台上，简单、稳定、低延迟的默认组合优先于更强但更慢的引导。", font=font(25, "medium"), fill=COLORS["ink"])
    add_footer(d)
    return img


def slide_08(assets: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["teal"])
    add_header(d, 8, "闭环框架：四层结构把视觉动作转成四足速度指令", "系统实现", COLORS["teal"])
    rounded_rect(img, (122, 156, 1120, 570), fill="#FFFFFF", outline=COLORS["line"])
    paste_image(img, assets["layered"], (150, 184, 1064, 514), mode="contain", radius=6, border=False)
    layers = [
        ((1320, 165, 430, 112), "推理层", "NoMaD 输出局部轨迹与距离", COLORS["cyan"]),
        ((1320, 315, 430, 112), "任务组织层", "状态机维护目标节点与超时逻辑", COLORS["teal"]),
        ((1320, 465, 430, 112), "中层控制层", "PD 映射到线速度与偏航速度", COLORS["orange"]),
        ((1320, 615, 430, 112), "平台执行层", "MuJoCo 或 Lite3 UDP 接口执行", COLORS["green"]),
    ]
    for i, (box, title, body, color) in enumerate(layers):
        rounded_rect(img, box, fill="#FFFFFF", outline=color)
        x, y, w, h = box
        d.text((x + 24, y + 18), title, font=font(28, "bold"), fill=color)
        d.text((x + 24, y + 58), body, font=font(20, "regular"), fill=COLORS["muted"])
        if i:
            arrow(d, (x + w // 2, y - 32), (x + w // 2, y - 6), COLORS["line"], 4)
    bottom = (122, 790, 1628, 126)
    rounded_rect(img, bottom, fill="#FFFFFF", outline=COLORS["teal"], shadow=False)
    d.text((160, 818), "关键接口约束", font=font(27, "bold"), fill=COLORS["teal"])
    draw_bullets(
        d,
        (380, 810, 1320, 82),
        ["局部轨迹只提供高层期望，底层稳定仍由 Lite3 控制器负责", "同一任务组织层可切换仿真接口与真实 UDP 接口，便于逐级验证"],
        COLORS["teal"],
        22,
    )
    add_footer(d)
    return img


def slide_09(assets: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["orange"])
    add_header(d, 9, "仿真验证：先证明闭环链路可跑通，再进入真实机器人", "MuJoCo", COLORS["orange"])
    rounded_rect(img, (100, 150, 1120, 610), fill="#FFFFFF", outline=COLORS["line"])
    paste_image(img, assets["mujoco"], (128, 178, 1064, 554), mode="contain", radius=6, border=False)
    d.rounded_rectangle((1270, 168, 1745, 218), radius=6, fill=COLORS["orange"])
    d.text((1302, 177), "健康检查任务", font=font(26, "bold"), fill="#FFFFFF")
    rows = [
        ("easy", "直行与轻微转向", "到达"),
        ("medium", "更长路径", "到达"),
        ("hard", "复杂目标", "到达"),
    ]
    y = 260
    for i, (case, desc, result) in enumerate(rows):
        color = [COLORS["green"], COLORS["teal"], COLORS["cyan"]][i]
        rounded_rect(img, (1270, y, 475, 106), fill="#FFFFFF", outline=color, shadow=False)
        d.text((1295, y + 21), case, font=font(30, "bold"), fill=color)
        d.text((1435, y + 24), desc, font=font(22, "regular"), fill=COLORS["ink"])
        d.rounded_rectangle((1627, y + 31, 1712, y + 69), radius=5, fill=color)
        d.text((1644, y + 35), result, font=font(19, "medium"), fill="#FFFFFF")
        y += 134
    rounded_rect(img, (1270, 690, 475, 170), fill="#FFFFFF", outline=COLORS["orange"])
    d.text((1300, 724), "仿真定位", font=font(28, "bold"), fill=COLORS["orange"])
    draw_text_box(d, (1300, 774, 415, 60), "用于验证状态机、PD 映射与平台接口，不替代真实机器人实验结论。", 22, COLORS["muted"], line_spacing=1.18)
    d.rounded_rectangle((150, 885, 1770, 950), radius=8, fill="#FFFFFF", outline=COLORS["orange"], width=2)
    d.text((198, 902), "过渡逻辑：仿真通过后，保持高层策略与任务组织不变，仅将平台执行层切换为 Lite3 UDP 链路。", font=font(25, "medium"), fill=COLORS["ink"])
    add_footer(d)
    return img


def slide_10(assets: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["green"])
    add_header(d, 10, "真实部署：Orin 与 Lite3 之间形成视觉推理、运动执行双链路", "平台部署", COLORS["green"])
    rounded_rect(img, (105, 150, 1110, 620), fill="#FFFFFF", outline=COLORS["line"])
    paste_image(img, assets["dual"], (132, 178, 1056, 564), mode="contain", radius=6, border=False)
    chain = [
        ((1280, 170, 480, 210), "水平推理链", ["相机图像", "NoMaD 推理", "PD 速度映射"], COLORS["teal"]),
        ((1280, 430, 480, 210), "垂直控制链", ["UDP 桥接", "Lite3 高层接口", "步态与姿态稳定"], COLORS["orange"]),
    ]
    for box, title, items, color in chain:
        rounded_rect(img, box, fill="#FFFFFF", outline=color)
        x, y, w, h = box
        d.text((x + 26, y + 22), title, font=font(29, "bold"), fill=color)
        xx = x + 34
        yy = y + 90
        for i, item in enumerate(items):
            d.rounded_rectangle((xx + i * 138, yy, xx + i * 138 + 110, yy + 46), radius=5, fill=color)
            d.text((xx + i * 138 + 13, yy + 10), item, font=font(17, "medium"), fill="#FFFFFF")
            if i < len(items) - 1:
                arrow(d, (xx + i * 138 + 112, yy + 23), (xx + i * 138 + 135, yy + 23), COLORS["line"], 3)
    metrics = [
        ("25Hz", "速度刷新"),
        ("5Hz", "心跳保持"),
        ("0.12m/s", "最大线速度"),
        ("0.35rad/s", "最大偏航"),
    ]
    x = 190
    for value, label in metrics:
        draw_metric(d, (x, 820, 310, 126), value, label, COLORS["green"])
        x += 400
    add_footer(d)
    return img


def slide_11(assets: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["teal"])
    add_header(d, 11, "拓扑图导航：用局部窗口让目标逐步向前推进", "任务组织", COLORS["teal"])
    rounded_rect(img, (100, 150, 1020, 570), fill="#FFFFFF", outline=COLORS["line"])
    paste_image(img, assets["map"], (130, 178, 960, 512), mode="contain", radius=6, border=False)
    thumbs = [
        (assets["map_start"], "起点节点 000", COLORS["cyan"]),
        (assets["map_mid"], "中间节点 020", COLORS["teal"]),
        (assets["map_goal"], "目标节点 039", COLORS["orange"]),
    ]
    x = 1190
    y = 165
    for path, label, color in thumbs:
        rounded_rect(img, (x, y, 520, 150), fill="#FFFFFF", outline=color, shadow=False)
        paste_image(img, path, (x + 14, y + 14, 190, 122), mode="cover", radius=6, border=False)
        d.text((x + 230, y + 34), label, font=font(25, "bold"), fill=color)
        d.text((x + 230, y + 76), "作为拓扑序列中的视觉锚点", font=font(19, "regular"), fill=COLORS["muted"])
        y += 180
    d.rounded_rectangle((190, 795, 1730, 895), radius=8, fill="#FFFFFF", outline=COLORS["teal"], width=2)
    d.text((232, 824), "滑动窗口半径 ±4", font=font(28, "bold"), fill=COLORS["teal"])
    axis_x, axis_y = 610, 842
    d.line((axis_x, axis_y, axis_x + 850, axis_y), fill=COLORS["line"], width=5)
    for i in range(9):
        cx = axis_x + i * 96
        color = COLORS["orange"] if i == 4 else COLORS["cyan"] if i < 4 else COLORS["green"]
        d.ellipse((cx - 19, axis_y - 19, cx + 19, axis_y + 19), fill=color)
        d.text((cx - 18, axis_y + 32), str(i - 4), font=font(17, "medium"), fill=COLORS["muted"])
    d.text((1490, 824), "距离预测器选择局部目标", font=font(22, "medium"), fill=COLORS["ink"])
    d.text((1490, 856), "避免一次跳到远端节点", font=font(18, "regular"), fill=COLORS["muted"])
    add_footer(d)
    return img


def slide_12(assets: dict[str, Path], manifest: list[dict]) -> Image.Image:
    img, d = new_slide(COLORS["green"])
    add_header(d, 12, "真实机器人结果：默认配置在走廊任务中取得最高频率", "实验结果", COLORS["green"])
    table = (105, 150, 930, 520)
    rounded_rect(img, table, fill="#FFFFFF", outline=COLORS["line"])
    x, y, w, _ = table
    d.text((x + 34, y + 28), "室内走廊消融结果", font=font(31, "bold"), fill=COLORS["green"])
    headers = ["配置", "成功率", "高层频率", "结论"]
    col = [x + 34, x + 438, x + 594, x + 744]
    yy = y + 92
    d.rectangle((x + 28, yy, x + w - 28, yy + 48), fill=COLORS["subtle"])
    for i, htxt in enumerate(headers):
        d.text((col[i], yy + 10), htxt, font=font(18, "bold"), fill=COLORS["ink"])
    rows = [
        ("DDPM", "100%", "2.89 Hz", "基线可用但慢"),
        ("DDIM-2 / CFG=0 / TTS=8", "100%", "6.80 Hz", "默认推荐"),
        ("DDIM-2 / CFG=2 / TTS=8", "100%", "3.43 Hz", "引导变慢"),
        ("DDIM-3 / CFG=0 / 无TTS", "100%", "5.90 Hz", "更快但筛选弱"),
    ]
    yy += 62
    for i, row in enumerate(rows):
        if i == 1:
            d.rounded_rectangle((x + 25, yy - 6, x + w - 25, yy + 47), radius=5, fill="#E7F6F1")
        for j, txt in enumerate(row):
            fnt = font(18, "bold" if (i == 1 or j == 2) else "regular")
            color = COLORS["green"] if i == 1 and j in (0, 2) else COLORS["ink"]
            d.text((col[j], yy + 5), txt, font=fnt, fill=color)
        yy += 64
    chart = (110, 705, 930, 210)
    rounded_rect(img, chart, fill="#FFFFFF", outline=COLORS["line"], shadow=False)
    d.text((chart[0] + 28, chart[1] + 22), "频率对比", font=font(25, "bold"), fill=COLORS["ink"])
    vals = [2.89, 6.80, 3.43, 5.90]
    labels = ["DDPM", "默认", "CFG=2", "DDIM-3"]
    maxv = 7.0
    bx = chart[0] + 175
    for i, (v, lab) in enumerate(zip(vals, labels)):
        bh = int(118 * v / maxv)
        cx = bx + i * 160
        color = COLORS["green"] if lab == "默认" else COLORS["cyan"] if i in (0, 3) else COLORS["orange"]
        d.rounded_rectangle((cx, chart[1] + 160 - bh, cx + 74, chart[1] + 160), radius=5, fill=color)
        d.text((cx - 5, chart[1] + 168), lab, font=font(17, "medium"), fill=COLORS["muted"])
        d.text((cx, chart[1] + 132 - bh), f"{v:.2f}", font=font(17, "bold"), fill=color)
    draw_video_placeholder(
        img,
        d,
        (1110, 168, 680, 420),
        "真实走廊导航演示",
        "ddim2cfg0tts8 / navigation_record.mp4",
        assets.get("thumb_fast"),
        COLORS["green"],
    )
    manifest.append(
        {
            "slide": 12,
            "placeholder": "真实走廊导航演示视频",
            "box_px": [1110, 168, 680, 420],
            "video": rel(assets["video_fast"]),
        }
    )
    rounded_rect(img, (1110, 640, 680, 238), fill="#FFFFFF", outline=COLORS["green"])
    d.text((1150, 675), "现场讲述重点", font=font(28, "bold"), fill=COLORS["green"])
    draw_bullets(
        d,
        (1152, 728, 600, 110),
        ["所有配置在测试走廊中均完成到达", "默认组合在成功率不变时显著提高闭环频率"],
        COLORS["green"],
        23,
    )
    add_footer(d)
    return img


def slide_13(assets: dict[str, Path], manifest: list[dict]) -> Image.Image:
    img, d = new_slide(COLORS["orange"])
    add_header(d, 13, "探索与室外泛化：能力边界来自光照、地形与视觉分布差异", "扩展实验", COLORS["orange"])
    rounded_rect(img, (100, 150, 790, 475), fill="#FFFFFF", outline=COLORS["line"])
    paste_image(img, assets["outdoor"], (126, 178, 738, 419), mode="contain", radius=6, border=False)
    cards = [
        ((935, 160, 380, 160), "无目标探索", "目标掩码置空，策略依据历史观测生成前进动作。", COLORS["teal"]),
        ((1395, 160, 380, 160), "室外零样本", "直接迁移到楼下斜坡、草地、强光环境。", COLORS["orange"]),
        ((935, 370, 380, 160), "可行场景", "纹理连续、坡度较小、光照不过曝时可前进。", COLORS["green"]),
        ((1395, 370, 380, 160), "失效边界", "草地、空旷区域和强反光使视觉表征不稳定。", COLORS["red"]),
    ]
    for box, title, body, color in cards:
        draw_meta_box(img, d, box, title, body, color, 25, 19)
    draw_video_placeholder(
        img,
        d,
        (165, 700, 710, 225),
        "室外斜坡演示",
        "二号楼楼下斜坡1 / navigation_record.mp4",
        assets.get("thumb_slope"),
        COLORS["green"],
    )
    draw_video_placeholder(
        img,
        d,
        (1015, 700, 710, 225),
        "草地失败案例",
        "电草，失败 / navigation_record.mp4",
        assets.get("thumb_grass"),
        COLORS["red"],
    )
    manifest.extend(
        [
            {"slide": 13, "placeholder": "室外斜坡演示视频", "box_px": [165, 700, 710, 225], "video": rel(assets["video_slope"])},
            {"slide": 13, "placeholder": "草地失败案例视频", "box_px": [1015, 700, 710, 225], "video": rel(assets["video_grass"])},
        ]
    )
    add_footer(d)
    return img


def slide_14(_: dict[str, Path]) -> Image.Image:
    img, d = new_slide(COLORS["teal"])
    add_header(d, 14, "总结与展望：从视觉扩散策略到四足机器人闭环导航", "总结", COLORS["teal"])
    d.text((126, 160), "主要工作", font=font(38, "bold"), fill=COLORS["ink"])
    work = [
        ("算法", "构建统一 NoMaD 推理模块，系统比较 DDIM、CFG、TTS 与编码器替换。", COLORS["cyan"]),
        ("系统", "搭建四层闭环架构，实现仿真与 Lite3 真实平台的接口复用。", COLORS["teal"]),
        ("实验", "完成 MuJoCo、室内走廊、无目标探索与室外泛化验证。", COLORS["green"]),
    ]
    x = 126
    for tag, body, color in work:
        draw_meta_box(img, d, (x, 235, 510, 190), tag, body, color, 26, 21)
        x += 600
    d.text((126, 520), "不足与改进方向", font=font(38, "bold"), fill=COLORS["ink"])
    future = [
        ((126, 600, 780, 118), "训练数据", "加入四足机器人视角和室外数据，降低分布偏移。"),
        ((1014, 600, 780, 118), "控制融合", "将局部轨迹与更强的速度/足端控制约束联合优化。"),
        ((126, 760, 780, 118), "语义记忆", "拓扑图结合语义、深度或占据记忆，提高长距离导航能力。"),
        ((1014, 760, 780, 118), "实时部署", "继续压缩模型并评估 TensorRT/量化部署收益。"),
    ]
    for i, (box, title, body) in enumerate(future):
        color = [COLORS["orange"], COLORS["purple"], COLORS["cyan"], COLORS["green"]][i]
        draw_meta_box(img, d, box, title, body, color, 25, 19)
    d.rounded_rectangle((690, 925, 1230, 985), radius=4, fill=COLORS["red"])
    d.text((805, 934), "谢谢各位老师", font=font(31, "bold"), fill="#FFFFFF")
    add_footer(d)
    return img


def save_slide(img: Image.Image, name: str) -> Path:
    out = OUT_DIR / name
    img.convert("RGB").save(out, quality=95)
    return out


SPEAKER_NOTES = [
    "各位老师好，我是陈样，我的毕业设计题目是《基于动作扩散策略的四足机器人视觉目标导航》。本课题围绕一个问题展开：如何让四足机器人只依靠相机观测和目标图像，在真实走廊与室外环境中稳定地向目标移动。接下来我会按照研究背景、算法设计、系统实现、仿真验证和真实机器人实验的顺序汇报。",
    "首先是研究背景。视觉目标导航的目标不是给机器人一张精确地图，而是给它目标图像和当前视觉观测，让它在闭环中持续判断下一步怎么走。这类能力适用于服务、巡检和救援等场景。对四足机器人来说，它既要理解视觉目标，又要在真实平台上满足运动执行约束，因此本文关注的是从视觉策略到机器人闭环执行的完整链路。",
    "本课题面临三类挑战。第一是算法侧，动作扩散策略生成效果好，但原始采样步数多，推理延迟较高。第二是系统侧，NoMaD 输出的是局部动作轨迹，而 Lite3 接收的是速度和控制指令，需要中间映射与状态机。第三是实验侧，仿真只能验证链路是否通畅，真实走廊、光照和室外地面才会暴露泛化问题。",
    "这一页是全文技术路线。论文先介绍视觉导航、扩散模型、NoMaD 和 Lite3 平台基础；然后设计统一推理模块，比较 DDIM、CFG、TTS 和编码器替换；再搭建 MuJoCo 闭环验证系统；最后迁移到 Orin 加 Lite3 的真实平台。整体逻辑是先解决实时性，再解决系统闭环，最后通过真实场景验证可用性和边界。",
    "算法基线采用 NoMaD。它输入历史观测图像、目标图像和目标掩码，经过视觉编码器后，一方面预测到拓扑候选目标的距离，另一方面通过扩散策略生成局部动作轨迹。本文保留这种高层策略框架，同时面向四足机器人做推理加速、候选筛选和速度指令适配。默认实验中模型参数量约为 19.05M。",
    "围绕 NoMaD，我做了四类改进。DDIM 用少步确定性采样替代原始多步随机采样，是主要的实时性来源。CFG 用有条件和无条件分支调节目标引导强度，但会引入额外推理开销。TTS 一次生成多条候选轨迹，再用距离预测器筛选。编码器替换则用于评估不同视觉表征的速度和泛化潜力。最终默认配置是 EfficientNet-B0、DDIM-2、CFG=0、TTS=8。",
    "离线消融结果说明，DDIM-2 相比 DDPM-10 在纯扩散采样上带来约 4.84 倍加速，采样延迟从约 40 毫秒降到 8 毫秒量级。CFG 虽然能增强目标引导，但双分支推理会带来接近两倍开销，真实平台收益不稳定。TTS-8 在候选多样性和延迟之间较平衡，所以后续真实机器人默认采用 DDIM-2、CFG=0、TTS=8。",
    "系统实现采用四层闭环结构。第一层是推理层，由 NoMaD 输出局部轨迹和距离；第二层是任务组织层，维护拓扑节点、导航状态和超时逻辑；第三层是中层控制层，把局部轨迹通过 PD 规则映射为线速度和偏航速度；第四层是平台执行层，可以接 MuJoCo，也可以接 Lite3 UDP 接口。这样同一套任务逻辑可以先仿真、再上真机。",
    "在进入真实机器人之前，我先用 MuJoCo 做健康检查。这里的目的不是证明最终性能，而是验证状态机、PD 映射和平台接口是否形成闭环。easy、medium 和 hard 三类任务都能到达目标，说明系统链路可运行。之后迁移到真实机器人时，高层策略和任务组织基本保持一致，只替换底层执行接口。",
    "真实部署采用 Orin 加 Lite3 的双链路结构。水平推理链从相机图像进入 NoMaD，得到局部动作，再映射为速度指令；垂直控制链负责 UDP 桥接、Lite3 高层接口和底层步态稳定。为保证安全，速度刷新为 25Hz，心跳为 5Hz，最大线速度限制在 0.12 米每秒，最大偏航速度限制在 0.35 弧度每秒。",
    "真实导航依赖预先采集的拓扑图。系统不会一次选择很远的目标节点，而是在当前节点附近使用半径为正负 4 的滑动窗口。距离预测器在窗口内选择局部目标，机器人每前进一段就更新窗口。这样可以把长距离目标拆成连续的局部视觉目标，降低单次目标跳转带来的不稳定性。",
    "这一页是真实走廊实验结果。四组配置在测试走廊中都完成到达，说明系统闭环是可用的。其中 DDPM 基线高层频率约 2.89Hz，默认 DDIM-2、CFG=0、TTS=8 达到约 6.80Hz，是最高频率。打开 CFG 后频率下降到约 3.43Hz，说明在真实平台上，稳定低延迟的默认组合更适合部署。",
    "除了有目标导航，我还测试了无目标探索和室外零样本迁移。无目标探索通过目标掩码置空，让策略依据历史观测生成前进动作。室外实验显示，在坡度较小、纹理连续、光照不过曝的场景下机器人可以前进；但草地、强光和空旷区域会造成视觉表征不稳定。这说明系统具有一定迁移能力，但仍受训练分布影响。",
    "最后总结一下。本文的主要工作包括：构建统一 NoMaD 推理模块并完成 DDIM、CFG、TTS 等消融；搭建从推理到四足执行的四层闭环系统；完成 MuJoCo、室内走廊、无目标探索和室外泛化实验。不足主要在训练数据分布、控制融合和长距离语义记忆方面。后续可以加入更多四足视角数据，并结合深度、语义或占据记忆提升泛化能力。我的汇报结束，谢谢各位老师。",
]


def build_ppt(slide_paths: list[Path]) -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    for idx, path in enumerate(slide_paths):
        slide = prs.slides.add_slide(blank)
        slide.shapes.add_picture(str(path), 0, 0, width=prs.slide_width, height=prs.slide_height)
        if idx < len(SPEAKER_NOTES):
            slide.notes_slide.notes_text_frame.text = SPEAKER_NOTES[idx]
    out = PPT_DIR / "visualnav_defense.pptx"
    prs.save(out)
    return out


def write_notes(manifest: list[dict]) -> None:
    notes = [
        "# 答辩PPT页说明",
        "",
        "本文件由 `build_defense_ppt.py` 生成。PPT 中每页为一张完整排版图片，视频采用占位框；实际答辩时可在对应页面将占位框替换为视频。",
        "",
        "## 视频占位",
    ]
    for item in manifest:
        if "slide" not in item:
            continue
        notes.append(f"- 第 {item['slide']} 页：{item['placeholder']}，位置 px={item['box_px']}，视频：`{item['video']}`")
    notes.extend(
        [
            "",
            "## 每页讲述要点",
            "- 01 标题：点出动作扩散策略、四足机器人、视觉目标导航三个关键词。",
            "- 02 背景：说明视觉闭环导航的应用需求。",
            "- 03 问题：强调算法实时性、系统接口、真实环境泛化三类挑战。",
            "- 04 技术路线：按理论、算法、仿真、真实机器人递进。",
            "- 05 NoMaD：解释输入输出和动作扩散基线。",
            "- 06 优化模块：DDIM、CFG、TTS、编码器替换。",
            "- 07 离线实验：说明默认配置的选择依据。",
            "- 08 系统架构：四层闭环与接口复用。",
            "- 09 仿真：MuJoCo 作为健康检查。",
            "- 10 部署：Orin-Lite3 双链路与安全速度约束。",
            "- 11 拓扑图：局部窗口和距离预测推动节点前进。",
            "- 12 真实结果：默认配置成功且最高频。",
            "- 13 扩展实验：探索、室外可行性与失败边界。",
            "- 14 总结：贡献、不足、未来工作。",
        ]
    )
    (PPT_DIR / "slide_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    setup_dirs()
    assets = get_assets()
    manifest: list[dict] = [
        {"canvas_px": [W, H], "safe_box_px": list(SAFE), "rule": "所有文本、图表、视频占位框均位于 safe_box_px 内。"}
    ]
    builders = [
        ("01_title.png", lambda: slide_01(assets)),
        ("02_background.png", lambda: slide_02(assets)),
        ("03_problem.png", lambda: slide_03(assets)),
        ("04_route.png", lambda: slide_04(assets)),
        ("05_nomad.png", lambda: slide_05(assets)),
        ("06_optimizations.png", lambda: slide_06(assets)),
        ("07_offline_results.png", lambda: slide_07(assets)),
        ("08_system_framework.png", lambda: slide_08(assets)),
        ("09_mujoco.png", lambda: slide_09(assets)),
        ("10_real_deploy.png", lambda: slide_10(assets)),
        ("11_topomap.png", lambda: slide_11(assets)),
        ("12_real_robot_results.png", lambda: slide_12(assets, manifest)),
        ("13_exploration_outdoor.png", lambda: slide_13(assets, manifest)),
        ("14_summary.png", lambda: slide_14(assets)),
    ]
    slide_paths = [save_slide(builder(), name) for name, builder in builders]
    ppt = build_ppt(slide_paths)
    write_notes(manifest)
    (PPT_DIR / "slide_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"slides={len(slide_paths)}")
    print(f"ppt={ppt}")
    print(f"png_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
