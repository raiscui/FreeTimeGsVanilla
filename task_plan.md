# 任务计划: FreeTimeGS -> `.splat4d v1/v2` 映射规格文档

## 目标
在 `specs/` 增加一份短而可执行的规格.
用论文原文+公式,把 FreeTimeGS 的(µx,µt,s,v,σ(t))映射到`.splat4d`字段.
同时说明 Unity runtime 的解释方式,避免后续反复口头对齐.

## 阶段
- [x] 阶段1: 计划和设置
- [x] 阶段2: 研究/收集信息(论文+代码)
- [x] 阶段3: 执行/构建(写 specs 文档)
- [x] 阶段4: 审查和交付(校验 Mermaid,回写四文件)

## 方案方向(至少二选一)

### 方向A: 不惜代价,最佳方案(更长期稳定)
- 思路: 规格文档中同时给出:
  - 论文原文短引述(不超过 25 words/段) + 页码.
  - trainer/exporter/Unity shader 的公式对照.
  - 1 张 flowchart + 1 张 sequenceDiagram.
  - 一张“字段/单位/语义”对照表.
- 代价: 文档更长,但后续沟通成本最低.
- 收益: 规范化,可直接作为 exporter/runtime 的契约.

### 方向B: 先能用,后面再优雅(本次先做)
- 思路: 文档保持短.
  - 只保留最关键的 2-3 条原文短引述.
  - 以表格+一张 flowchart 为主.
  - 把更细节的实现指向代码路径.
- 代价: 细节需要点开代码看.
- 收益: 交付快,先把语义钉住.

## 关键问题
1. FreeTimeGS 的`duration`在论文里是`sigma`还是`log(sigma)`? 我们的 ckpt 里存的是什么?
2. `.splat4d-version`(timeModel)与`.splat4d-format-version`(header+sections)是正交概念,如何避免误用?
3. Unity runtime 对 timeModel=1(window)与 timeModel=2(gaussian)分别如何裁剪/加权/更新 position?

## 做出的决定
- [2026-02-23] 选择方向B 先交付短文档,后续若需要再扩展为方向A.
- [2026-02-23] 由于旧 `task_plan.md` 已达 1000 行,已续档为历史文件,并启动 continuous-learning.

## 状态
**目前在阶段4**: 已完成交付.规格已落盘,Mermaid 已校验,四文件已回写.
