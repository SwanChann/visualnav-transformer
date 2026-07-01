# PPT 自动生成项目

本目录是一个用于自动生成答辩 PPT 的项目工作区。项目通过脚本、素材图片、模板参考文件和生成结果文件组织 PPT 生产流程，便于后续维护、批量生成和复用版式。

## 目录结构

```text
.
|-- assets/                 # PPT 使用的补充素材，如视频缩略图
|-- figures/                # 论文、实验和系统相关图表素材
|-- generated/              # 自动生成的中间视觉资源
|-- node_modules/           # Node.js 依赖目录
|-- output/                 # 生成输出目录
|-- slides_png/             # 已导出的幻灯片预览图
|-- template_reference/     # PPT 模板解析出的参考资源
|-- build_defense_ppt.py    # 自动生成答辩 PPT 的主要脚本
|-- package.json            # Node.js 项目依赖配置
|-- package-lock.json       # Node.js 依赖锁定文件
|-- ppt模版.pptx             # PPT 模板文件
|-- slide_manifest.json     # 幻灯片结构或素材清单
|-- slide_notes.md          # 幻灯片备注与讲稿要点
`-- visualnav_defense.pptx  # 已生成的答辩 PPT
```

## 项目说明

- `build_defense_ppt.py` 是当前生成 PPT 的核心脚本。
- `figures/`、`assets/` 和 `generated/` 存放生成 PPT 所需的图片资源。
- `template_reference/` 用于保留模板中的版式、主题和媒体参考。
- `slides_png/` 可用于快速检查每页幻灯片的视觉效果。
- `visualnav_defense.pptx` 是已经生成的 PPT 成品。

## 依赖

当前 Node.js 依赖记录在 `package.json` 中，主要包括：

- `pptxgenjs`
- `openai`

如需重新安装 Node.js 依赖，可运行：

```powershell
npm install
```

## 使用

在本目录下运行生成脚本：

```powershell
python build_defense_ppt.py
```

生成结果通常会写入当前目录或 `output/` 目录，具体以脚本配置为准。
