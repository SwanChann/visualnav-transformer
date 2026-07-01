const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "visualnav-transformer";
pptx.subject = "基于动作扩散策略的四足机器人视觉目标导航";
pptx.title = "基于动作扩散策略的四足机器人视觉目标导航";
pptx.company = "Shanghai Jiao Tong University";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  lang: "zh-CN",
};

const OUT = path.join(__dirname, "output", "test.pptx");

const W = 13.333;
const H = 7.5;
const C = {
  bg: "F5F8FA",
  paper: "FFFFFF",
  ink: "17252D",
  muted: "5C6B75",
  subtle: "EDF3F6",
  line: "D7E2E7",
  teal: "008C95",
  cyan: "18A9C9",
  green: "3AA77A",
  orange: "F08B2E",
  navy: "243B53",
};

function addBase(slide, page, title, accent = C.teal) {
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: 0.12,
    fill: { color: accent },
    line: { color: accent },
  });

  slide.addText(String(page).padStart(2, "0"), {
    x: 0.55,
    y: 0.33,
    w: 0.36,
    h: 0.25,
    fontFace: "Microsoft YaHei",
    fontSize: 13,
    bold: true,
    color: accent,
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.95,
    y: 0.45,
    w: 0.6,
    h: 0.06,
    rectRadius: 0.01,
    fill: { color: accent },
    line: { color: accent },
  });
  slide.addText(title, {
    x: 1.7,
    y: 0.28,
    w: 8.2,
    h: 0.32,
    fontFace: "Microsoft YaHei",
    fontSize: 16,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55,
    y: 0.78,
    w: 12.25,
    h: 0,
    line: { color: C.line, width: 1 },
  });
}

function addFooter(slide) {
  slide.addText("基于动作扩散策略的四足机器人视觉目标导航", {
    x: 0.55,
    y: 7.08,
    w: 5.6,
    h: 0.2,
    fontFace: "Microsoft YaHei",
    fontSize: 8.5,
    color: "8A98A1",
    margin: 0,
  });
}

function addCard(slide, x, y, w, h, outline = C.line) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.08,
    fill: { color: C.paper },
    line: { color: outline, width: 1.2 },
    shadow: {
      type: "outer",
      color: "213440",
      opacity: 0.1,
      blur: 1,
      angle: 45,
      distance: 1,
    },
  });
}

function addBullet(slide, text, x, y, accent = C.teal) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y: y + 0.08,
    w: 0.12,
    h: 0.12,
    rectRadius: 0.02,
    fill: { color: accent },
    line: { color: accent },
  });
  slide.addText(text, {
    x: x + 0.28,
    y,
    w: 10.2,
    h: 0.36,
    fontFace: "Microsoft YaHei",
    fontSize: 18,
    color: C.ink,
    breakLine: false,
    margin: 0,
  });
}

function addModule(slide, label, x, y, w, h, accent) {
  addCard(slide, x, y, w, h, accent);
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w: 0.08,
    h,
    fill: { color: accent },
    line: { color: accent },
  });
  slide.addText(label, {
    x: x + 0.18,
    y: y + 0.22,
    w: w - 0.34,
    h: h - 0.3,
    fontFace: "Microsoft YaHei",
    fontSize: label.length > 15 ? 12.5 : 14.5,
    bold: true,
    color: C.ink,
    align: "center",
    valign: "mid",
    fit: "shrink",
    margin: 0.03,
  });
}

function addArrow(slide, x1, y1, x2, y2, color = C.teal) {
  slide.addShape(pptx.ShapeType.line, {
    x: x1,
    y: y1,
    w: x2 - x1,
    h: y2 - y1,
    line: {
      color,
      width: 2,
      endArrowType: "triangle",
    },
  });
}

function addSlide1() {
  const slide = pptx.addSlide();
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: 0.14,
    fill: { color: C.teal },
    line: { color: C.teal },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.72,
    y: 0.62,
    w: 6.25,
    h: 6.02,
    fill: { color: C.paper, transparency: 6 },
    line: { color: C.paper },
    shadow: {
      type: "outer",
      color: "213440",
      opacity: 0.12,
      blur: 2,
      angle: 45,
      distance: 1,
    },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0.72,
    y: 0.62,
    w: 0.1,
    h: 6.02,
    fill: { color: C.teal },
    line: { color: C.teal },
  });
  slide.addText("上海交通大学本科毕业设计答辩", {
    x: 1.22,
    y: 1.0,
    w: 5.1,
    h: 0.35,
    fontFace: "Microsoft YaHei",
    fontSize: 17,
    color: C.navy,
    margin: 0,
  });
  slide.addText("基于动作扩散策略的", {
    x: 1.05,
    y: 2.16,
    w: 5.3,
    h: 0.55,
    fontFace: "Microsoft YaHei",
    fontSize: 26,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  slide.addText("四足机器人视觉目标导航", {
    x: 1.05,
    y: 2.75,
    w: 5.55,
    h: 0.7,
    fontFace: "Microsoft YaHei",
    fontSize: 29,
    bold: true,
    color: C.teal,
    fit: "shrink",
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 1.08,
    y: 3.62,
    w: 4.25,
    h: 0.07,
    rectRadius: 0.03,
    fill: { color: C.orange },
    line: { color: C.orange },
  });
  slide.addText("Action Diffusion Policy for Visual Goal Navigation", {
    x: 1.06,
    y: 3.92,
    w: 5.1,
    h: 0.34,
    fontFace: "Aptos",
    fontSize: 14,
    color: C.muted,
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 1.08,
    y: 6.0,
    w: 1.45,
    h: 0.38,
    rectRadius: 0.05,
    fill: { color: C.teal },
    line: { color: C.teal },
  });
  slide.addText("2026年5月", {
    x: 1.25,
    y: 6.08,
    w: 1.1,
    h: 0.2,
    fontFace: "Microsoft YaHei",
    fontSize: 11,
    bold: true,
    color: C.paper,
    align: "center",
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 7.35,
    y: 0.92,
    w: 4.85,
    h: 5.45,
    fill: { color: C.subtle },
    line: { color: C.line, width: 1 },
  });
  slide.addText("VISUAL NAVIGATION", {
    x: 8.0,
    y: 1.68,
    w: 3.5,
    h: 0.34,
    fontFace: "Aptos",
    fontSize: 17,
    bold: true,
    color: C.teal,
    charSpace: 1,
    align: "center",
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 8.15,
    y: 3.55,
    w: 3.0,
    h: -0.7,
    line: { color: C.teal, width: 2, endArrowType: "triangle" },
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 8.15,
    y: 3.55,
    w: 2.9,
    h: 0.8,
    line: { color: C.orange, width: 2, endArrowType: "triangle" },
  });
  slide.addShape(pptx.ShapeType.ellipse, {
    x: 7.82,
    y: 3.18,
    w: 0.62,
    h: 0.62,
    fill: { color: C.teal },
    line: { color: C.teal },
  });
  slide.addShape(pptx.ShapeType.ellipse, {
    x: 10.85,
    y: 2.66,
    w: 0.46,
    h: 0.46,
    fill: { color: C.cyan },
    line: { color: C.cyan },
  });
  slide.addShape(pptx.ShapeType.ellipse, {
    x: 10.85,
    y: 4.25,
    w: 0.46,
    h: 0.46,
    fill: { color: C.orange },
    line: { color: C.orange },
  });
}

function addSlide2() {
  const slide = pptx.addSlide();
  addBase(slide, 2, "研究背景", C.cyan);
  slide.addText("研究背景", {
    x: 0.8,
    y: 1.18,
    w: 4.2,
    h: 0.55,
    fontFace: "Microsoft YaHei",
    fontSize: 26,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  slide.addText("从“看见目标”到“走到目标”", {
    x: 0.82,
    y: 1.78,
    w: 5.6,
    h: 0.4,
    fontFace: "Microsoft YaHei",
    fontSize: 17,
    bold: true,
    color: C.cyan,
    margin: 0,
  });

  addCard(slide, 0.8, 2.65, 11.7, 2.35, C.cyan);
  addBullet(slide, "四足机器人需要在非结构化环境中完成自主移动，视觉目标导航能直接利用相机观测降低建图与定位依赖。", 1.12, 3.05, C.cyan);
  addBullet(slide, "传统导航方法通常依赖显式地图、规划器和手工控制接口，在光照变化、地形扰动和算力受限平台上部署成本较高。", 1.12, 3.72, C.teal);
  addBullet(slide, "动作扩散策略可从目标图像和当前观测中生成连续动作序列，为端到端视觉导航提供更稳定、可迭代优化的策略表达。", 1.12, 4.39, C.orange);

  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.8,
    y: 5.55,
    w: 11.7,
    h: 0.62,
    rectRadius: 0.06,
    fill: { color: C.paper },
    line: { color: C.line },
  });
  slide.addText("核心关注：在真实 Lite3 平台上实现低延迟、可验证、可替换的视觉导航系统。", {
    x: 1.15,
    y: 5.72,
    w: 10.8,
    h: 0.25,
    fontFace: "Microsoft YaHei",
    fontSize: 14,
    bold: true,
    color: C.navy,
    margin: 0,
  });
  addFooter(slide);
}

function addSlide3() {
  const slide = pptx.addSlide();
  addBase(slide, 3, "系统框架", C.teal);
  slide.addText("系统框架", {
    x: 0.8,
    y: 1.13,
    w: 3.2,
    h: 0.5,
    fontFace: "Microsoft YaHei",
    fontSize: 25,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  slide.addText("视觉观测经过编码与距离估计后，由扩散策略生成高层动作，再映射到底层 PD 控制并驱动 Lite3。", {
    x: 0.82,
    y: 1.72,
    w: 11.0,
    h: 0.32,
    fontFace: "Microsoft YaHei",
    fontSize: 13.5,
    color: C.muted,
    margin: 0,
  });

  const y = 3.25;
  const w = 1.72;
  const h = 0.86;
  const modules = [
    ["Camera", 0.82, C.cyan],
    ["Visual Encoder", 2.87, C.teal],
    ["Distance Predictor", 4.92, C.orange],
    ["Diffusion Policy", 6.97, C.green],
    ["PD Control", 9.02, C.navy],
    ["Lite3", 11.07, C.teal],
  ];

  modules.forEach(([label, x, accent]) => addModule(slide, label, x, y, w, h, accent));
  for (let i = 0; i < modules.length - 1; i += 1) {
    const [, x1] = modules[i];
    const [, x2] = modules[i + 1];
    addArrow(slide, x1 + w + 0.1, y + h / 2, x2 - 0.1, y + h / 2, C.teal);
  }

  addCard(slide, 1.05, 5.15, 11.25, 0.86, C.line);
  slide.addText("模块职责", {
    x: 1.38,
    y: 5.38,
    w: 1.25,
    h: 0.24,
    fontFace: "Microsoft YaHei",
    fontSize: 13,
    bold: true,
    color: C.teal,
    margin: 0,
  });
  slide.addText("感知输入 | 视觉特征 | 局部目标选择 | 动作序列采样 | 关节/速度控制 | 真实四足平台执行", {
    x: 2.58,
    y: 5.38,
    w: 8.95,
    h: 0.24,
    fontFace: "Microsoft YaHei",
    fontSize: 12.5,
    color: C.ink,
    margin: 0,
  });
  addFooter(slide);
}

async function main() {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  addSlide1();
  addSlide2();
  addSlide3();
  await pptx.writeFile({ fileName: OUT });
  console.log(`Created ${OUT}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
