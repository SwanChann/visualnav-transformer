# recon raw pilot content audit

- Passed: True
- Artifact: `/home/yifei/codespace/visualnav-transformer/nomad_dataset/recon_raw/recon_release/jackal_2019-12-19-14-24-48_0_r00.hdf5`
- Bytes: 143826
- SHA-256: `d850e8cf3b6a3613d7fce54312f19109412edf816031791fdf0838febbcffbb6`
- Conversion/model/training/evaluation/simulation executed: False

## Content

```json
{
  "required_nodes": [
    "jackal/position",
    "jackal/yaw",
    "images/rgb_left"
  ],
  "missing_nodes": [],
  "stream_shapes": {
    "position": [
      2,
      3
    ],
    "yaw": [
      2
    ],
    "rgb_left": [
      2
    ]
  },
  "stream_dtypes": {
    "position": "float64",
    "yaw": "float64",
    "rgb_left": "|S28573"
  },
  "counts": {
    "position": 2,
    "yaw": 2,
    "rgb_left": 2
  },
  "sample_indices": [
    0,
    1
  ],
  "decoded_images": [
    {
      "index": 0,
      "format": "JPEG",
      "mode": "RGB",
      "size": [
        640,
        480
      ],
      "encoded_bytes": 28573
    },
    {
      "index": 1,
      "format": "JPEG",
      "mode": "RGB",
      "size": [
        640,
        480
      ],
      "encoded_bytes": 28296
    }
  ],
  "finite_pose_samples": true,
  "errors": [],
  "passed": true
}
```

## Evidence boundary

Raw pilot content/readability audit only. It does not validate the full dataset, metric scale, collection-session completeness, conversion output, model behavior, training, evaluation, or simulation.
