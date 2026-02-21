# FreeTimeGsVanilla checkpoint -> `.sog4d` exporter 手册(数据流映射清单)

> 适用对象: 你要在 `https://github.com/OpsiClear/FreeTimeGsVanilla` 的改版里做导出器(exporter).
> 导出目标: 本仓库(Unity UPM 包 `wu.yize.gsplat`)可以直接导入和播放的 `.sog4d`.
>
> 这份文档不依赖你“记得某次口头约定”.
> 我把所有关键假设都写成可核对的规则,方便 code agent 逐条实现.

---

## 0. 先回答你的问题: 这些特点还存在吗?

结论: 在 `OpsiClear/FreeTimeGsVanilla` 的公开版本(main 分支)里,这些特点**仍然存在**.
并且它们的语义和你描述的一致:

### 0.1 checkpoint 确实包含每个 Gaussian 的 4D 参数

公开仓库的 `src/simple_trainer_freetime_4d_pure_relocation.py` 在 `load_checkpoint()` 里,直接从 `ckpt["splats"]` 读取并重建如下字段:

- `means` `[N,3]`
- `scales` `[N,3]`(log scale)
- `quats` `[N,4]`(四元数)
- `opacities` `[N]`(logit)
- `sh0` `[N,1,3]`
- `shN` `[N,K,3]`
- `times` `[N,1]`
- `durations` `[N,1]`(log sigma)
- `velocities` `[N,3]`

你可以在公开仓库里直接定位到这段代码(我这里用 URL 形式给出,避免描述歧义):

```text
https://raw.githubusercontent.com/OpsiClear/FreeTimeGsVanilla/main/src/simple_trainer_freetime_4d_pure_relocation.py
```

### 0.2 temporal opacity 的高斯核语义仍然是: sigma(t) = exp(-0.5 * ((t - mu_t)/s)^2)

同一个公开仓库里,`compute_temporal_opacity()` 的实现是:

- `mu_t = times`
- `s = exp(durations)`(并 clamp 最小值,默认 0.02)
- `temporal_opacity(t) = exp(-0.5 * ((t - mu_t) / (s + eps))^2)`

并且 combined opacity 是:

- `base_opacity = sigmoid(opacities_logit)`
- `opacity(t) = base_opacity * temporal_opacity(t)`

对应实现同样可在公开仓库找到:

```text
https://raw.githubusercontent.com/OpsiClear/FreeTimeGsVanilla/main/src/simple_trainer_freetime_4d_pure_relocation.py
```

### 0.3 为什么这让导出更“顺手”

因为你不需要再像 4DGaussians 那样:
- 先对每个时间点跑 deformation network 导出 PLY
- 再用 PLY 序列差分去拟合 velocity/time/duration

FreeTimeGS checkpoint 已经给了你:
- 运动模型的闭式表达式(线性): `pos(t) = means + velocities * (t - times)`
- 时间权重函数的闭式表达式(高斯): `opacity(t) = sigmoid(opacities) * temporal_opacity(t)`

所以 exporter 的主工作变成:
- 选一组你要导出的 `t_frame[]`
- 在这些 t 上把 `pos(t)` 和 `opacity(t)` 直接“采样烘焙”成逐帧 keyframe 数据

而 `.sog4d` 恰好就是“逐帧 keyframe + 全属性可插值”的格式.

---

## 1. `.sog4d` 的本质(你要导出的是什么)

`.sog4d` 是一个 ZIP bundle.
根目录必须包含 `meta.json`.
其它文件主要是:
- per-frame 的 lossless WebP “数据图”(不是贴图).
- (可选) SH rest 的 palette(centroids.bin) + labels(WebP 或 delta-v1).

更完整的规范,以本仓库 OpenSpec 为准:
- `openspec/specs/sog4d-container/spec.md`
- `openspec/specs/sog4d-sequence-encoding/spec.md`
- `openspec/specs/sog4d-unity-importer/spec.md`

Unity 侧读者实现(最终权威,因为它决定能不能导入):
- `Editor/GsplatSog4DImporter.cs`

---

## 2. exporter 的数据流映射清单(从 checkpoint 到 `.sog4d`)

我把整个 exporter 拆成 3 个层次:
- A. 从 checkpoint 得到“静态参数”(不随帧变)
- B. 从 4D 参数得到“逐帧动态参数”(pos/opacity)
- C. 把逐帧参数编码成 `.sog4d` streams(量化纹理 + palette + meta.json)

### 2.1 A层: checkpoint -> 静态参数(一次性)

输入:
- `ckpt = torch.load(ckpt_path)["splats"]`

输出(建议统一转成 float32,放 CPU numpy,方便后续写 WebP/zip):
- `means0`: `[N,3]` float32
- `vel`: `[N,3]` float32
- `mu_t`: `[N,1]` float32
- `log_sigma`: `[N,1]` float32
- `base_opacity`: `[N]` float32 = `sigmoid(opacities_logit)`
- `scale_log`: `[N,3]` float32(来自 ckpt 的 `scales`)
- `scale_lin`: `[N,3]` float32 = `exp(scale_log)`
- `quat_wxyz`: `[N,4]` float32(归一化 + hemisphere 规范化)
- `f_dc`: `[N,3]` float32 = `sh0.squeeze(1)`
- `sh_rest`: `[N,restCoeffCount,3]` float32 = `shN` reshape/拷贝后得到

注意事项(这是最常见的坑):
- `opacities` 是 logit,必须 sigmoid.
- `scales` 是 log scale,必须 exp.
- `durations` 是 log(sigma),必须 exp.
- `quats` 分量顺序是 (w,x,y,z),并且必须归一化.

### 2.2 B层: 4D 参数 -> 逐帧采样(pos/opacity)

你需要先决定:
- `frameCount = M`
- `timeMapping`:
  - uniform: `t_i = i/(M-1)`
  - explicit: `t_i` 用外部列表(仍然必须在 [0,1],且单调非递减)

对每一帧时间 `t`:

1) 位置(线性运动):
- `pos(t) = means0 + vel * (t - mu_t)`

2) temporal opacity(高斯核):
- `sigma = exp(log_sigma)`
- 建议保持与 FreeTimeGsVanilla 一致的防塌缩 clamp:
  - `sigma = clamp(sigma, minSigma)`(默认 `minSigma=0.02`)
- `temporal_opacity(t) = exp(-0.5 * ((t - mu_t) / (sigma + eps))^2)`

3) combined opacity(写入 `.sog4d` 的 alpha):
- `opacity(t) = base_opacity * temporal_opacity(t)`

4) 可选的“显式不可见”裁剪(让数据更干净,也更接近 Unity 侧阈值):
- Unity 运行时的硬阈值是 `opacity < 1/255 => 0`.
- 所以 exporter 可以做:
  - `if opacity(t) < 1/255: opacity(t) = 0`

这一层的输出就是:
- 每帧一份 `pos_f: [N,3]` float32
- 每帧一份 `opacity_f: [N]` float32

其它参数(scale/rot/SH)如果你不做 time-dependent 外观,可以认为是静态.

### 2.3 C层: 逐帧参数 -> `.sog4d` streams(编码)

下面是**一一对应**的映射表.
这部分你只要照着实现,Unity importer 就能读.

#### 2.3.1 `streams.position`(u16 量化,hi/lo 两张 WebP)

每帧你需要算:
- `rangeMin_f = pos_f.min(axis=0)`
- `rangeMax_f = pos_f.max(axis=0)`

然后对每个 splat 的每个分量做 u16 量化:

```text
t = clamp01((x - min) / (max - min))
q_u16 = round(t * 65535)
hi = q_u16 >> 8
lo = q_u16 & 0xFF
```

写两张 RGBA8 WebP:
- `frames/{frame}/position_hi.webp`
  - RGB = (hi_x, hi_y, hi_z)
  - A = 255
- `frames/{frame}/position_lo.webp`
  - RGB = (lo_x, lo_y, lo_z)
  - A = 255

#### 2.3.2 `streams.scale`(codebook + indices.webp)

`.sog4d` 运行时希望 scale 是线性值.
但 exporter 为了更稳的量化,建议仍然在 log-domain 拟合 codebook:

- 拟合输入: `log(scale_lin)`(也就是 checkpoint 的 `scale_log`)
- 得到 `scale_codebook_log[K,3]`
- 存入 meta.json 的 codebook 要写线性:
  - `scale_codebook = exp(scale_codebook_log)`

每帧写 `frames/{frame}/scale_indices.webp`:
- 这是 u16 index map.
- 由于 scale 不随时间变化,你可以:
  - 先算一份 `idx_scale[N]`
  - 然后所有帧复用同一个 idx_scale 写成 WebP(内容相同,但路径不同)

WebP 像素格式:
- `index = R + (G<<8)`(小端)
- B=0,A=255

#### 2.3.3 `streams.rotation`(RGBA8 量化四元数)

四元数处理流程(建议完全对齐本仓库 pack 工具):
- normalize
- hemisphere 规范化: `if w < 0: q = -q`
- 量化:
  - `byte = round(clamp(q,-1,1) * 128 + 128)`

每帧写 `frames/{frame}/rotation.webp`:
- RGBA 对应 (w,x,y,z) 的 byte

同样因为 rotation 通常是静态的:
- 你可以先算一份 `quat_u8[N,4]`
- 每帧写同内容.

#### 2.3.4 `streams.sh` 的 SH0 与 opacity(`sh0.webp`)

`.sog4d` 的 `sh0.webp` 同时承载:
- `f_dc`(DC SH 系数): RGB
- `opacity`: A

你需要先生成 `sh0Codebook[256]`(float).
推荐做法(与本仓库 pack 工具一致):
- 把所有 `f_dc` 的标量样本收集起来(每个 splat 贡献 3 个标量).
- 用:
  - quantile(更快,更稳),或
  - kmeans(误差更小,但依赖 sklearn)

然后每帧生成 `frames/{frame}/sh0.webp`:
- RGB 三个 byte 是 `f_dc.r/g/b` 在 codebook 里的索引(0..255)
- A 是 `opacity(t)` 量化到 byte:
  - `A = round(clamp01(opacity(t)) * 255)`

注意:
- 这意味着 `.sog4d` 可以非常自然地承接 FreeTimeGS 的 temporal opacity.
- 你只要把 `opacity(t)` 烘焙进每一帧的 alpha 就行.

#### 2.3.5 SH rest: 两种版本(v1/v2),你二选一

你的 exporter 需要在 “实现复杂度” 和 “压缩效果” 之间做选择.

##### 版本v1(meta.version=1): 单一 palette(shN)

适用:
- 你想先跑通链路.
- 或者你不想处理 sh1/sh2/sh3 拆分.

做法:
- `bands` 决定 `restCoeffCount = (bands+1)^2 - 1`
- 把 `sh_rest` 展平到 `[N, restCoeffCount*3]` 做 kmeans:
  - 得到 `centroids[shNCount, restCoeffCount*3]`
  - centroids 写入 `shN_centroids.bin`(little-endian,f16/f32)
- 每帧写 labels:
  - full: `frames/{frame}/shN_labels.webp`
  - delta-v1: `frames/{startFrame}/shN_labels.webp` + `sh/delta_*.bin`

##### 版本v2(meta.version=2): 按 band 拆分(sh1/sh2/sh3)

适用(推荐):
- 你更关心质量/码率.
- 你希望更贴近 DualGS/DynGsplat 一类的“多 codebook”思路.

做法:
- sh1 使用 rest 的前 3 个 coeff,维度 D=3*3=9
- sh2 使用接下来的 5 个 coeff,维度 D=5*3=15
- sh3 使用接下来的 7 个 coeff,维度 D=7*3=21
- 分别拟合三套 kmeans:
  - `sh1_centroids.bin`
  - `sh2_centroids.bin`
  - `sh3_centroids.bin`
- labels 同理分三套:
  - full: `frames/{frame}/sh1_labels.webp` 等
  - delta-v1: `frames/{startFrame}/sh1_labels.webp` + `sh/sh1_delta_*.bin` 等

#### 2.3.6 delta-v1 的“最顺手用法”(结合 FreeTimeGS 的静态 SH)

FreeTimeGS 的 SH 系数通常是静态的(不随时间变).
这会产生一个非常好的性质:
- 每一帧的 SH labels 都是一样的.

因此你用 delta-v1 时:
- segment 的 baseLabels 写一次
- segment 内每帧的 updateCount 都是 0

这会让 delta 文件非常小.
实现也很简单:
- 你只要按格式写 header
- 然后循环写 `(frameCount-1)` 个 `u32(0)`

delta-v1 二进制格式要点(必须严格遵守,否则 Unity importer 会 fail-fast):
- Header:
  - magic: 8 bytes = "SOG4DLB1"
  - version: u32 = 1
  - segmentStartFrame: u32
  - segmentFrameCount: u32
  - splatCount: u32
  - labelCount: u32(=shNCount 或 band.count)
- Body:
  - 对 segment 内每个后续帧,写:
    - updateCount(u32)
    - updateCount 个条目:
      - splatId(u32)
      - newLabel(u16)
      - reserved(u16,必须为 0)
  - 同一帧(block)内 splatId 必须严格递增.

---

## 3. 推荐的 exporter 形态(给 code agent 的“施工图”)

你可以用两条路线实现.
我建议先选路线1跑通,再决定是否需要路线2.

### 路线1(最快能用): checkpoint -> per-frame PLY -> 调用本仓库 pack 工具

逻辑:
1) 你在 FreeTimeGsVanilla 里写一个脚本:
   - 读 checkpoint
   - 选 `t_frame[]`
   - 计算每帧 `pos(t)` 和 `opacity(t)`(把 temporal opacity 乘进去)
   - 其它参数(scale/rot/SH)直接从 checkpoint 拿(静态)
   - 用 `gsplat.exporter.export_splats` 写成每帧标准 PLY
2) 然后直接调用本仓库现成的 pack:
   - `python3 Tools~/Sog4D/ply_sequence_to_sog4d.py pack ...`

优点:
- pack/validate/zip/webp/codebook/palette/delta 这一大坨你不用再写一遍.
- 出错时,你可以用 `validate` 做快速定位.

缺点:
- 会产生一堆 PLY 中间文件,IO 比较大.

### 路线2(最终形态): checkpoint -> 直接写 `.sog4d`

逻辑:
- 直接把本仓库的 `Tools~/Sog4D/ply_sequence_to_sog4d.py` 当作参考实现.
- 把“读 PLY”那部分替换成“读 checkpoint + 逐帧采样”.
- 其它部分(量化,WebP lossless,delta-v1)尽量保持一致.

优点:
- 没有 PLY 中间文件.
- 更快,也更容易控制内存峰值(可以 chunk).

缺点:
- 需要你在 exporter 里维护更多格式细节.

---

## 4. 参数建议(让脚本可调,避免硬编码)

以当前仓库已经落地的脚本为准,参数建议如下(可以直接对照 `--help`):

### 4.1 `.sog4d` exporter: `tools/exportor/export_sog4d.py`

基础:
- `--ckpt-path`
- `--output-path`
- `--frame-count`(默认 61)
- `--time-mapping uniform|explicit`
- `--frame-times path_or_csv`(仅 explicit)
- `--layout-width`
- `--min-sigma`(默认 0.02,对齐训练/viewer 的 clamp)
- `--overwrite`

全局裁剪(必须对所有帧一致,否则会破坏 splat identity):
- `--base-opacity-threshold`(按 base_opacity=sigmoid(logit)过滤)
- `--max-splats`(按 base_opacity top-k 全局裁剪)

每帧 alpha 的“硬置零”(对齐 Unity 运行时的 1/255 阈值):
- `--alpha-zero-threshold`(默认 1/255)

编码/体积:
- `--webp-method`(默认 0,最快)
- `--webp-quality`(默认 100,lossless)
- `--zip-compression stored|deflated`(默认 stored)

scale codebook:
- `--scale-codebook-size`(默认 256)
- `--scale-codebook-sample`(默认 200000)
- `--assign-chunk`(默认 200000)

SH0 codebook:
- `--sh0-codebook-sample`(默认 0,使用全量 3*N 标量)

SH rest(当前只实现 v1: 单一 shN palette,输出 `meta.json.version=1`):
- `--sh-bands 0..3`(0=仅 sh0+opacity; 1..3=额外导出 shN)
- `--shn-count`(默认 512,越大质量越好,但 kmeans 更慢)
- `--shn-centroids-type f16|f32`(默认 f16)
- `--shn-labels-encoding full|delta-v1`(默认 delta-v1,SH 静态时几乎零成本)
- `--shn-codebook-sample`(默认 100000)
- `--shn-assign-chunk`(默认 50000)
- `--shn-kmeans-iters`(默认 10)

未实现(需要后续增量开发,不要按这些参数去调用当前脚本):
- `meta.json.version=2` 的 per-band palette(`sh1/sh2/sh3`)导出.
- 可配置的 delta segment length(当前 delta-v1 固定写一个 segment 覆盖 `[0,frameCount)`).

### 4.2 `.splat4d` exporter: `tools/exportor/export_splat4d.py`

基础:
- `--ckpt`
- `--output`
- `--base-opacity-threshold`
- `--chunk-size`
- `--min-sigma`

版本语义:
- `--splat4d-version 1|2`
  - v1: hard-window(旧语义,用 `--temporal-threshold` 近似时间高斯核)
  - v2: gaussian(新语义,time=mu_t,duration=sigma,更贴近 FreeTimeGS checkpoint)

v1 专用:
- `--temporal-threshold`(默认 0.01)

---

## 5. 最小验证清单(你写完 exporter 后怎么快速判定“对了”)

1) 先导出一个最小配置:
- `.sog4d`:
  - `--sh-bands 0`(只做 sh0+opacity)
  - `--frame-count 5`
  - 建议加 `--max-splats 50000` 做快速冒烟
- `.splat4d`:
  - `--splat4d-version 2`(如果你的 importer 支持 v2)

2) 用 Unity importer 验证:
- 能导入(不会在 Console 报 import error)
- 能播放(拖动 TimeNormalized 时物体会动,并且会淡入淡出)

3) 如果你采用路线1(PLY->pack):
- 先跑 `pack --self-check`
- 再跑 `validate`

---

## 6. 额外建议(我认为你后面一定会用到)

- `.sog4d` 是“逐帧 keyframe”格式.
  如果你最终目标是“连续 4D 运动”,`.splat4d` 反而更直接.
  你可以同时输出两份:
  - `.splat4d`: 用于 runtime 4DGS 主后端(速度+时间核)
    - v1: hard window(旧语义,time0+duration).
    - v2: gaussian(新语义,time=mu_t,duration=sigma),更贴近 FreeTimeGS checkpoint.
    - exporter: `python tools/exportor/export_splat4d.py --splat4d-version 2 ...`
  - `.sog4d`: 用于逐帧对照,或用于需要全属性插值的工作流

- 由于 FreeTimeGS 的 SH 通常是静态的:
  - 建议优先用 `delta-v1`.
  - 你几乎免费就能获得很好的压缩比(大量 updateCount=0).
