# Orin 相机标定与输入对齐说明

## 1. 结论

1. 相机原始输出不需要手动改成视觉编码器输入尺寸，当前代码会在进入 NoMaD 之前自动缩放到配置文件里的 `image_size`。
2. 相机标定不是所有真机实验的硬前置条件。普通 USB 相机、畸变不明显时，可以先不标定，直接完成阶段一独立调试。
3. 如果使用广角镜头、鱼眼镜头，或者画面边缘畸变明显，建议先完成标定，再在桥接层开启去畸变。

## 2. 当前代码行为

### 2.1 输入尺寸对齐

当前部署链里的输入尺寸对齐已经由代码完成：

1. `scripts/shared/nomad_inference.py`
   `pil_to_tensor()` 会按当前策略配置中的 `image_size` 自动执行 `resize(...)` 和归一化。
2. `deployment/src/utils.py`
   旧部署链中的 `transform_images()` 也会使用同样的 `image_size` 做缩放。

这意味着，真正需要保证的是“真机预处理与训练时一致”，而不是要求相机原始输出本身就等于 `96x96` 或 `98x98`。

当前几种常用配置对应的输入尺寸如下：

1. `nomad_encoder_efficientnet_b0.yaml` 使用 `image_size: [96, 96]`
2. `nomad_encoder_convnext_tiny.yaml` 使用 `image_size: [96, 96]`
3. `nomad_encoder_resnet50.yaml` 使用 `image_size: [96, 96]`
4. `nomad_encoder_dinov2_small.yaml` 使用 `image_size: [98, 98]`

### 2.2 相机标定

当前真机桥接层已支持可选标定文件：

1. `scripts/deployment/lite3_real_bridge.py`
   `OrinCamera` 新增了 `calibration_path` 和 `undistort_alpha`。
2. `scripts/deployment/orin_standalone_test.py`
   阶段一独立测试新增了 `--camera-calibration-path` 和 `--undistort-alpha`。
3. `scripts/configs/navigation_host/lite3_real_bridge_config.json`
   已增加 `camera_calibration_path` 和 `undistort_alpha` 字段。

如果提供标定文件，桥接层会在每次读帧后执行 OpenCV 去畸变，再进入后续推理流程。

## 3. 标定文件格式

仓库里已提供模板：

```bash
ls scripts/configs/navigation_host/camera_calibration_template.json
```

模板内容如下：

```json
{
  "camera_matrix": [
    [525.0, 0.0, 319.5],
    [0.0, 525.0, 239.5],
    [0.0, 0.0, 1.0]
  ],
  "dist_coeffs": [-0.12, 0.08, 0.0, 0.0, 0.0]
}
```

建议复制为自己的文件：

```bash
cp scripts/configs/navigation_host/camera_calibration_template.json \
   scripts/configs/navigation_host/my_camera_calibration.json
```

然后把真实相机内参与畸变参数填进去。

## 4. 启用方式

### 4.1 阶段一独立测试时启用

```bash
python scripts/deployment/orin_standalone_test.py \
  --test camera \
  --camera-device 0 \
  --camera-calibration-path scripts/configs/navigation_host/my_camera_calibration.json \
  --undistort-alpha 0.0
```

### 4.2 真机桥接配置里长期启用

在 `scripts/configs/navigation_host/lite3_real_bridge_config.json` 中填写：

```json
"camera_calibration_path": "scripts/configs/navigation_host/my_camera_calibration.json",
"undistort_alpha": 0.0
```

其中：

1. `camera_calibration_path`
   指向标定 JSON 文件。
2. `undistort_alpha`
   对应 OpenCV 去畸变时保留视场的参数。
   `0.0` 表示尽量裁掉黑边；
   `1.0` 表示尽量保留视场，但更可能出现边缘黑边。

## 5. 推荐策略

1. 普通镜头、畸变不明显：先不标定，保持最小可运行链路。
2. 广角镜头、边缘直线明显弯曲：先标定，再开启桥接层去畸变。
3. 无论是否标定，都不要手动去改模型输入尺寸；应始终以策略配置中的 `image_size` 为准。

## 6. 预期结果

1. 不启用标定时，阶段一 `camera / model / benchmark / pipeline` 仍应正常通过。
2. 启用标定时，终端会额外打印：

```text
[OrinCamera] 已加载相机标定文件: ...
```

3. 去畸变会增加少量前处理开销，但一般不会改变“是否能在 Orin 上部署”的结论。
