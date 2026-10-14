# demo_slam

A minimal Python pipeline for **3D point cloud reconstruction from video** using sparse feature tracking, structure from motion, bundle adjustment, dense stereo, and voxel filtering.

👉 **Interactive 3D Demo:**  
[https://maratzakirov.github.io/demo_slam/](https://maratzakirov.github.io/demo_slam/)

(it renders file myscene.ply made from kitchen.mp4)

*Controls: Left Click to rotate, Right Click to pan, Scroll to zoom.*

---

## Features

- Video loading and frame preprocessing.
- Sparse feature tracking with **Lucas–Kanade optical flow**.
- Camera pose estimation via **Essential Matrix + RANSAC**.
- Scale correction using previously triangulated 3D points.
- Track management through `TrackManager`: observations `(cam, u, v)`, 3D points in world frame, colors.
- **Bundle Adjustment** with analytic sparse Jacobian and Huber loss.
- Dense stereo for selected frame pairs:
  - `stereoRectify`
  - `remap`
  - **HitNet** disparity via ONNX Runtime
  - `reprojectImageTo3D`
- Voxel voting and filtering of the dense point cloud.
- Binary **PLY** export.
- Debug visualizations: optical flow, matches, 3D scene.

---

## Dependencies

```bash
pip install numpy opencv-python matplotlib scipy scikit-image onnxruntime
```

## Note

The maim.py will process the video stream and export the final 3D point cloud directly to dense.ply and dense_voxel.ply.
To view these point clouds you may use online tool like https://swyvl.io/tools/point-cloud-viewer/

## Future development

- 3D landmark reconstruction
- 3D pose relocalization (finding pose of random camera shot in a processed map)