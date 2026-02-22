# ERRORFIX

## 2026-02-22 10:20:30 UTC - Unity 中 `.splat4d` 点云整体偏移/歪倒

### 现象
- Unity 导入 `.splat4d` 后,高斯点云的整体位置偏移,并且看起来像被整体旋转(歪倒).
- 当 Unity 侧同时使用原始 COLMAP 相机位姿(或以 COLMAP 坐标为参考)时,两者对不上.

### 根因
- 训练代码 `FreeTimeParser(normalize=True)` 会对 COLMAP 原始空间应用归一化 transform:
  - 相机中心做平移+统一缩放(similarity)
  - 点云做 PCA 对齐
- ckpt 中的 `means/velocities/scales/quats` 都处于 "train normalized 空间".
- exporter 旧实现直接把 ckpt 坐标写到 `.splat4d`,导致 Unity 侧拿 COLMAP 原始空间做参考时出现整体错位.

### 修复
- `tools/exportor/export_splat4d.py` 新增:
  - `--output-space train|colmap`(默认 train)
  - `--colmap-dir <sparse/0>`(当 output-space=colmap 时必填)
- 当 `--output-space=colmap` 时:
  - 读取 COLMAP 的 `cameras/images/points3D`.
  - 复现训练侧的 `colmap->train` transform.
  - 对导出 record 的 `(position, velocity, scale, rotation)` 统一应用 `T^{-1}`,导出回 COLMAP 原始空间.
- `.splat4d v2` 额外写入 `XFRM` section(64B,16xf32)记录 `colmap->train` transform,用于离线 debug.

### 验证
- 重新导出并检查二进制头/sections:
  - 输出: `results/bar_release_full/out_0_61/exports/ckpt_29999_v2_sh3_seg50_k512_f16_colmap.splat4d`
  - magic=`SPL4DV02`
  - section table 包含 `XFRM`
