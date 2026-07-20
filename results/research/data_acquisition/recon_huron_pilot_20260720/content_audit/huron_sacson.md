# huron_sacson raw pilot content audit

- Passed: True
- Artifact: `/home/yifei/codespace/visualnav-transformer/nomad_dataset/huron_raw/Dec-09-2022-bww8/00000010.bag`
- Bytes: 11044659
- SHA-256: `71ac9844157bee7e90f9a6ab1d91af10ebad7e39073fd2885ccabc6d66034397`
- Conversion/model/training/evaluation/simulation executed: False

## Content

```json
{
  "required_topics": {
    "/fisheye_image/compressed": "sensor_msgs/CompressedImage",
    "/odometry": "nav_msgs/Odometry"
  },
  "all_topics": {
    "/ar_marker": {
      "message_type": "ar_track_alvar_msgs/AlvarMarkers",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.916270414470688
    },
    "/boundary_boxes": {
      "message_type": "darknet_ros_msgs/BoundingBoxes",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.890138051185652
    },
    "/bumper": {
      "message_type": "create_msgs/Bumper",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.916298825685017
    },
    "/depth_spherical_image/compressed": {
      "message_type": "sensor_msgs/CompressedImage",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.902735104054567
    },
    "/fisheye_image/compressed": {
      "message_type": "sensor_msgs/CompressedImage",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.889713938550532
    },
    "/laserscan": {
      "message_type": "sensor_msgs/LaserScan",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.91757748341268
    },
    "/odometry": {
      "message_type": "nav_msgs/Odometry",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.919822962770112
    },
    "/panorama_image/compressed": {
      "message_type": "sensor_msgs/CompressedImage",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.891099495472757
    },
    "/pedestrians_pose": {
      "message_type": "visualization_msgs/MarkerArray",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.898400696366163
    },
    "/spherical_image/compressed": {
      "message_type": "sensor_msgs/CompressedImage",
      "message_count": 198,
      "connections": 1,
      "frequency_hz": 10.905541558438182
    }
  },
  "start_time_s": 1678499944.8839028,
  "end_time_s": 1678499964.194526,
  "duration_s": 19.310623168945312,
  "message_count": 1980,
  "image_sample": {
    "stamp_s": 1678499944.8929641,
    "format_field": "jpeg",
    "decoded_format": "JPEG",
    "mode": "RGB",
    "size": [
      160,
      120
    ],
    "encoded_bytes": 10180
  },
  "odometry_sample": {
    "stamp_s": 1678499944.8908525,
    "frame_id": "odom",
    "child_frame_id": "base_footprint",
    "finite_pose": true
  },
  "errors": [],
  "passed": true
}
```

## Evidence boundary

Raw pilot content/readability audit only. It does not validate the full dataset, metric scale, collection-session completeness, conversion output, model behavior, training, evaluation, or simulation.
