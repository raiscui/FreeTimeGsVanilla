# Capability: sog4d-sequence-encoding

## Purpose
定义 `.sog4d` 的逐帧序列编码方式(streams/layout/time mapping/SH palette 与 labels).
目标是在保持运行时解码高性能的同时,让所有核心属性逐帧可插值.

## Requirements

### Requirement: `meta.json.format` MUST be `"sog4d"`
为了让读者实现(gsplat-unity)能 fail-fast 识别 bundle 类型,`meta.json` 顶层 MUST 包含:
- `format`: MUST 为 `"sog4d"`

备注:
- 早期 exporter 若缺少该字段,会导致离线 `validate` 失败,并阻塞 Unity importer.

### Requirement: Frame-to-frame splat identity MUST be stable
系统 MUST 保证 frame-to-frame 的 splat identity 稳定,以支持逐帧 keyframe 插值.
因此系统 MUST 满足以下恒等约束:
- `splatId` 的定义在所有帧中 MUST 一致.
- `splatCount` 在所有帧中 MUST 一致,且等于 `meta.json.splatCount`.
- 每帧的属性图(layout) MUST 使用相同的 `layout` 参数,并遵循相同的 `splatId -> pixel` 映射.

#### Scenario: frameCount 变化导致无法插值
- **WHEN** 某一帧的属性图像素数量不足以覆盖 `splatCount`
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含 frameIndex 与 `splatCount` 信息)

### Requirement: `timeMapping` MUST support both uniform and explicit modes
`meta.json.timeMapping.type` MUST 为以下之一:
- `"uniform"`: 帧时间均匀分布在 `[0,1]`
- `"explicit"`: 帧时间由 `frameTimesNormalized[]` 显式给出

当 `type="uniform"` 时:
- 当 `frameCount > 1` 时,系统 MUST 将第 `i` 帧的时间定义为 `t_i = i / (frameCount - 1)`
- 当 `frameCount == 1` 时,系统 MUST 将唯一帧时间定义为 `t_0 = 0.0`

当 `type="explicit"` 时:
- `meta.json.timeMapping.frameTimesNormalized` MUST 存在
- `frameTimesNormalized` 长度 MUST 等于 `frameCount`
- `frameTimesNormalized[i]` MUST 位于 `[0,1]`
- `frameTimesNormalized` MUST 单调非递减

#### Scenario: uniform time mapping 的帧时间定义
- **WHEN** `frameCount=5` 且 `timeMapping.type="uniform"`
- **THEN** 系统 MUST 解释帧时间为 `[0.0, 0.25, 0.5, 0.75, 1.0]`

#### Scenario: explicit time mapping 的合法性校验
- **WHEN** `timeMapping.type="explicit"` 且 `frameTimesNormalized` 存在越界值(例如 `-0.1`)
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含越界值与索引)

### Requirement: `layout` MUST define a deterministic splatId-to-pixel mapping
系统 MUST 通过 `layout` 定义确定性的 splatId-to-pixel 映射.
系统 MUST 以"数据图"的方式读取像素,而不是以"颜色图"的方式读取.

`meta.json.layout` MUST 包含:
- `type`: MUST 为 `"row-major"`
- `width`: MUST 为正整数
- `height`: MUST 为正整数

并且 MUST 定义映射:
- `pixelIndex = splatId`
- `x = pixelIndex % width`
- `y = floor(pixelIndex / width)`
- 仅当 `splatId < splatCount` 时该像素有效

#### Scenario: layout 的尺寸不足
- **WHEN** `width * height < splatCount`
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含 `width`, `height`, `splatCount`)

### Requirement: Per-frame path templates MUST interpret `{frame}` as a zero-padded decimal frame index
系统 MUST 以确定性的方式把 per-frame 路径模板中的 `{frame}` 替换为 frameIndex 字符串.

对于任何 per-frame 路径模板(例如 `position.hiPath`, `scale.indicesPath`, `rotation.path`, `sh.sh0Path`, `sh.shNLabelsPath`):
- 模板字符串 MUST 包含 `{frame}`
- 系统 MUST 将 `{frame}` 替换为十进制 frameIndex,并左侧补零到至少 5 位
  - 例如 frameIndex=7 时,替换结果 MUST 为 `"00007"`
  - 当 frameIndex>=100000 时,替换结果 MUST 允许自然扩展为更多位数(不截断)

#### Scenario: `{frame}` 的零填充替换
- **WHEN** `frameIndex=3`,且某个模板为 `"frames/{frame}/sh0.webp"`
- **THEN** 系统 MUST 解析到实际路径 `"frames/00003/sh0.webp"`

### Requirement: Streams MUST be per-frame and interpolable for position, scale, rotation, and SH
用户已确认: position, scale, rotation, opacity, SH MUST 逐帧可插值.
因此 `meta.json.streams` MUST 声明并提供以下 streams,且它们 MUST 为 per-frame:
- `position`
- `scale`
- `rotation`
- `sh`

其中:
- opacity MUST 作为 `sh0.webp` 的 alpha 被承载(见下文 `streams.sh`).

并且系统 MUST 能对上述属性在相邻两帧之间进行插值(插值语义由 `4dgs-keyframe-motion` 定义).

#### Scenario: 缺少必需 stream
- **WHEN** `meta.json.streams` 缺少 `sh`
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含缺失 stream 名)

### Requirement: Position stream MUST use u16 quantization with per-frame range
`streams.position` MUST 使用 16-bit 量化表示 position.
系统 MUST 支持 per-frame range,以减少动态序列的量化误差.

`streams.position` MUST 包含:
- `rangeMin`: float3 数组,长度 MUST 等于 `frameCount`
- `rangeMax`: float3 数组,长度 MUST 等于 `frameCount`
- `hiPath`: 字符串模板,例如 `"frames/{frame}/position_hi.webp"`
- `loPath`: 字符串模板,例如 `"frames/{frame}/position_lo.webp"`

JSON 形态约定:
- float3 MUST 序列化为 object 形态: `{"x": <float>, "y": <float>, "z": <float>}`
- 原因: Unity `JsonUtility` 解析 `Vector3` 需要该形态,不能写成 `[[x,y,z], ...]` 的 list-of-3.

并且每帧 MUST 存在两张 WebP 图:
- `position_hi.webp`: 每个像素的 RGB 为 x,y,z 的高 8-bit
- `position_lo.webp`: 每个像素的 RGB 为 x,y,z 的低 8-bit

对每个分量,解码规则 MUST 为:
- `q = hi * 256 + lo` (范围 `[0, 65535]`)
- `x = rangeMin.x + (q / 65535) * (rangeMax.x - rangeMin.x)` (y,z 同理)

#### Scenario: position range 数组长度不匹配
- **WHEN** `streams.position.rangeMin` 长度不等于 `frameCount`
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含 `frameCount` 与实际长度)

### Requirement: Scale stream MUST use a codebook plus per-frame indices
`streams.scale` MUST 使用 codebook + index map 的编码方式.
`streams.scale` MUST 包含:
- `codebook`: float3 数组,每个元素表示一个候选 scale(对象空间)
- `indicesPath`: 字符串模板,例如 `"frames/{frame}/scale_indices.webp"`

JSON 形态约定:
- `codebook` 内的 float3 MUST 序列化为 object 形态: `{"x": <float>, "y": <float>, "z": <float>}`
- 原因同上(Unity `JsonUtility`).

每帧 MUST 存在一张 `indicesPath` 指向的 WebP 图像作为 index map(例如 `scale_indices.webp`):
- 每个像素的 RG 表示一个 u16 index(小端): `index = r + (g << 8)`
- 解码时 `scale = codebook[index]`

#### Scenario: scale index 越界
- **WHEN** `scale_indices.webp` 中出现 `index >= len(codebook)`
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含 frameIndex 与 index 值)

### Requirement: Rotation stream MUST use per-frame quantized quaternions
`streams.rotation` MUST 以 per-frame 的量化 quaternion 表示旋转.
`streams.rotation` MUST 包含:
- `path`: 字符串模板,例如 `"frames/{frame}/rotation.webp"`

每帧 MUST 存在一张 `path` 指向的 WebP 图像(例如 `rotation.webp`).
每个像素的 RGBA MUST 表示 `(w, x, y, z)` 四元数的 8-bit 量化分量.

解码规则 MUST 与 `.splat4d` 保持一致:
- `v = (byte - 128) / 128`,得到 `[-1, 1]` 的近似值
- 解码后 MUST 归一化 quaternion
- 为了稳定插值路径,系统 MUST 对 quaternion 做半球规范化: `if w < 0 then q = -q`

#### Scenario: rotation 解码后归一化
- **WHEN** 某个像素解码得到的 quaternion 长度不为 1
- **THEN** 系统 MUST 在归一化后再进入后续插值与渲染流程

### Requirement: SH stream MUST be per-frame and support interpolation
`streams.sh` MUST 支持 SH0 与高阶 SH(rest coefficients)的逐帧表达,并可插值.

`streams.sh` MUST 包含:
- `bands`: 整数,范围 MUST 为 `[0, 3]`
- `sh0Path`: 字符串模板,例如 `"frames/{frame}/sh0.webp"`
- `sh0Codebook`: float 数组,长度 MUST 为 `256`

每帧 MUST 存在一张 `sh0Path` 指向的 WebP 图像(例如 `sh0.webp`).
该图像 MUST 为 RGBA8 数据图.
对每个像素:
- `f_dc.r = sh0Codebook[R]`
- `f_dc.g = sh0Codebook[G]`
- `f_dc.b = sh0Codebook[B]`
- `opacity = A / 255`

当 `bands > 0` 时,SH rest 的编码方式 MUST 由 `meta.json.version` 决定:
- `version=1`: 使用单一 SHN palette + labels(兼容路径).
- `version=2`: SH rest 按 band 拆分为 `sh1`/`sh2`/`sh3` 三套 palette + labels(推荐).

#### Encoding v1(meta.json.version == 1): SHN palette + labels
当 `meta.json.version == 1` 时,`streams.sh` 还 MUST 包含:
- `shNCount`: 整数,范围 MUST 为 `[1, 65535]`
- `shNCentroidsType`: 字符串,值 MUST 为 `"f16"` 或 `"f32"`
- `shNCentroidsPath`: 字符串,例如 `"shN_centroids.bin"`
- `shNLabelsEncoding`: 字符串,值 MUST 为 `"full"` 或 `"delta-v1"`

若 `shNLabelsEncoding` 缺失,系统 MUST 将其视为 `"full"`.

当 `bands > 0` 时,系统 MUST 定义:
- `restCoeffCount = (bands + 1)^2 - 1`

当 `shNLabelsEncoding="full"` 时,`streams.sh` 还 MUST 包含:
- `shNLabelsPath`: 字符串模板,例如 `"frames/{frame}/shN_labels.webp"`

并且每帧 MUST 存在一张 `shNLabelsPath` 指向的 WebP 图像(例如 `shN_labels.webp`).
该图像的每个像素 RG MUST 表示一个 u16 label(小端):
- `label = r + (g << 8)`
- `label` MUST 满足 `label < shNCount`

当 `shNLabelsEncoding="delta-v1"` 时,`streams.sh` 还 MUST 包含:
- `shNDeltaSegments`: 数组,按 `startFrame` 升序排列,覆盖 `[0, frameCount)` 的所有帧

并且:
- `streams.sh` MUST NOT 包含 `shNLabelsPath`(避免 "full" 与 "delta" 的路径定义发生歧义).
- `shNDeltaSegments` 的每个 segment MUST 包含:
  - `startFrame`: 整数,范围 MUST 为 `[0, frameCount-1]`
  - `frameCount`: 正整数
  - `baseLabelsPath`: 字符串,指向该 segment 首帧的 labels WebP(格式同 `shN_labels.webp`)
  - `deltaPath`: 字符串,指向该 segment 的 delta 二进制文件
- segments MUST 满足:
  - 第 0 个 segment 的 `startFrame` MUST 为 `0`
  - 相邻 segments MUST 连续:
    - `segments[i+1].startFrame == segments[i].startFrame + segments[i].frameCount`
  - 所有 segments 的 `frameCount` 总和 MUST 等于 `frameCount`

`baseLabelsPath` 指向的 WebP 图像 MUST 满足:
- 每个像素 RG 表示一个 u16 label(小端): `label = r + (g << 8)`
- `label` MUST 满足 `label < shNCount`

`deltaPath` 指向的二进制文件 MUST 为 little-endian,并使用格式 `labelDeltaV1`:
- Header:
  - `magic`: 8 bytes ASCII,值 MUST 为 `"SOG4DLB1"`
  - `version`: u32,值 MUST 为 `1`
  - `segmentStartFrame`: u32,值 MUST 等于该 segment 的 `startFrame`
  - `segmentFrameCount`: u32,值 MUST 等于该 segment 的 `frameCount`
  - `splatCount`: u32,值 MUST 等于 `meta.json.splatCount`
  - `labelCount`: u32,值 MUST 等于 `streams.sh.shNCount`

delta body MUST 为一组按 frame 顺序的 block.
对于 segment 内每个 frame(从 `startFrame+1` 到 `startFrame+frameCount-1`):
- block MUST 以 `updateCount`(u32)开头
- 随后 MUST 紧跟 `updateCount` 个 update 条目
- 每个 update MUST 为 `(splatId, newLabel, reserved)`:
  - `splatId`: u32,范围 MUST 为 `[0, splatCount-1]`
  - `newLabel`: u16,范围 MUST 为 `[0, labelCount-1]`
  - `reserved`: u16,值 MUST 为 `0`
- 同一个 block 内的 update 条目 MUST 按 `splatId` 严格递增(避免重复或乱序应用).

#### Scenario: shNLabelsEncoding="delta-v1" 时缺少 segments
- **WHEN** `shNLabelsEncoding="delta-v1"` 但 `shNDeltaSegments` 缺失
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含缺失字段名)

#### Scenario: shN base labels 越界
- **WHEN** `baseLabelsPath` 对应的 WebP 中出现 `label >= shNCount`
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含 frameIndex 与 label 值)

#### Scenario: shN delta 中 newLabel 越界
- **WHEN** delta 文件中出现 `newLabel >= labelCount`
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含 frameIndex 与 newLabel 值)

#### Encoding v2(meta.json.version == 2): per-band palettes(sh1/sh2/sh3)
当 `meta.json.version == 2` 且 `bands > 0` 时:
- 系统 MUST 使用 per-band 的 palette + labels,并在 `streams.sh` 下声明:
  - `sh1`: 当 `bands >= 1` 时 MUST 存在,表示 l=1 的 3 个 rest coefficients.
  - `sh2`: 当 `bands >= 2` 时 MUST 存在,表示 l=2 的 5 个 rest coefficients.
  - `sh3`: 当 `bands >= 3` 时 MUST 存在,表示 l=3 的 7 个 rest coefficients.

每个 band stream MUST 为对象,并 MUST 包含:
- `count`: 整数,范围 MUST 为 `[1, 65535]`
- `centroidsType`: 字符串,值 MUST 为 `"f16"` 或 `"f32"`
- `centroidsPath`: 字符串,指向该 band 的 centroids binary 文件
- `labelsEncoding`: 字符串,值 MUST 为 `"full"` 或 `"delta-v1"`(缺失时视为 `"full"`)

当 `labelsEncoding="full"` 时,该 band stream 还 MUST 包含:
- `labelsPath`: per-frame 模板,例如 `"frames/{frame}/sh1_labels.webp"`

当 `labelsEncoding="delta-v1"` 时,该 band stream 还 MUST 包含:
- `deltaSegments`: 数组,按 `startFrame` 升序排列,覆盖 `[0, frameCount)` 的所有帧
- 并且该 band stream MUST NOT 包含 `labelsPath`

`deltaSegments` 的 segment schema 与 `shNDeltaSegments` 一致:
- `startFrame`: 整数
- `frameCount`: 正整数
- `baseLabelsPath`: 字符串,指向该 segment 首帧的 labels WebP
- `deltaPath`: 字符串,指向该 segment 的 delta 二进制文件

v2 的 delta 文件 MUST 复用同一个 `labelDeltaV1` 二进制布局:
- `labelCount` MUST 等于对应 band 的 `count`
- update 的 `newLabel` MUST 小于该 band 的 `count`

### Requirement: SH centroids MUST be stored in a binary file for efficiency
当 `bands > 0` 时,系统 MUST 通过二进制文件承载 SH rest 的 centroids(palette).
这样做的目的是避免 JSON 解析开销,并降低 bundle 体积.

当 `meta.json.version == 1` 时:
- `streams.sh.shNCentroidsPath` MUST 指向一个 binary 文件(默认 `"shN_centroids.bin"`).
- 该文件 MUST 为 little-endian.
- 当 `shNCentroidsType="f16"` 时,该文件 MUST 以 `float16` 存储.
- 当 `shNCentroidsType="f32"` 时,该文件 MUST 以 `float32` 存储.
- 并且该文件 MUST 满足 size:
  - `size == shNCount * restCoeffCount * 3 * sizeof(floatType)`
  - `restCoeffCount = (bands + 1)^2 - 1`
  - `3` 表示 RGB 三通道

当 `meta.json.version == 2` 时:
- 对每个存在的 band stream(`sh1`/`sh2`/`sh3`),其 `centroidsPath` MUST 指向对应的 binary 文件.
- 该文件 MUST 为 little-endian.
- 当 `centroidsType="f16"` 时,该文件 MUST 以 `float16` 存储.
- 当 `centroidsType="f32"` 时,该文件 MUST 以 `float32` 存储.
- 并且每个文件 MUST 满足 size:
  - `size == count * coeffCount * 3 * sizeof(floatType)`
  - `coeffCount` 固定为: `sh1=3`,`sh2=5`,`sh3=7`

#### Scenario: centroids binary 尺寸不匹配
- **WHEN** 任意一个被 `streams.sh.*.centroidsPath` 引用的 centroids binary 文件大小不满足对应公式
- **THEN** 导入器 MUST 失败,并输出明确的 error(包含期望 size 与实际 size)
