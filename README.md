# 数据质量评分校准工具（DQ Score Calibration）

可版本化的数据质量评分校准台：规则权重、通过阈值、豁免窗口随**校准版本**冻结；
每次评分拆到规则与样本并生成不可变运行记录；历史评分在版本升级后原样保留，
告警与发布门禁始终按运行所钉的版本策略执行；在规则冲突或样本不足时**拒绝输出
假精确分数**。

## 核心设计

### 四态运行模型（无假精确）

评分引擎是纯函数（`app/domain/scoring.py`，无 IO），输出四种互斥状态，
只有 `scored` 携带数值总分，其余状态 `overall_score = null` 且门禁自动阻断：

| 状态 | 触发条件 | 总分 | 门禁 |
|---|---|---|---|
| `scored` | 所有规则达到各自 `min_samples`，批次达到 `min_batch_samples`，冲突组未超限 | 加权分（0–1） | 按版本 gate_policy 判定 |
| `empty` | 批次无样本，或全部观测被豁免窗口覆盖、无评估证据 | `null` | 阻断 |
| `insufficient_samples` | 批次样本不足或某规则评估样本低于 `min_samples` | `null` | 阻断 |
| `rule_conflict` | 静态冲突组内规则在共享样本上的不一致率超过容忍度 | `null` | 阻断 |

状态优先级：empty → rule_conflict → insufficient_samples → scored。

### 版本化与不可变性

- 校准版本生命周期 `draft → active → archived`（`POST /api/versions/{id}/activate`），
  同一时刻只有一个 active；配置入库后永不修改，激活只翻转状态行。
- 每个版本对配置做规范化 JSON 的 **sha256 校验和**；每条运行记录同时钉住
  `version_id` 与 `config_checksum`，历史分数可永久解释。
- 运行记录（含提交快照、逐样本证据、豁免原因、门禁结论、告警）只插入不更新。
- 回补（backfill）按 `(version_id, dataset, batch_id)` **幂等**：重复提交返回
  既有运行；跨版本重评（`rescore_of_run_id`）生成一条新的不可变运行，
  原运行的分数与门禁结论保持不变。

### 评分口径

- 规则分 = 评估通过样本数 / 评估样本数（豁免样本不计入分母）。
- 总分按规则权重加权，全程高精度计算、仅对展示值保留 4 位小数，避免舍入累积。
- `min_samples=0` 的规则若被豁免到零评估样本，其权重在剩余有证据的规则间
  重新归一化，**不会把"没证据"记 0 分**，规则分本身仍为 `null`。
- 豁免窗口为半开区间 `[start_at, end_at)`，按观测时间（优先 observation 级
  时间戳）匹配 dataset + rule。
- 冲突组按组内规则的共享样本统计不一致率，豁免观测不携带判定、不参与冲突计算。

## 目录结构

```
app/
  domain/models.py      # pydantic 模型：校准配置、批次、评分结果、版本对照
  domain/scoring.py     # 纯函数评分引擎（四态、豁免、冲突、门禁、告警）
  storage.py            # SQLite：版本表/不可变 runs 表/alerts 表 + 校验和
  main.py               # FastAPI 路由
  demo.py               # 幂等演示种子（覆盖全部四态、空批次、回补/重评）
  static/index.html     # 单页 UI（无外部依赖）
tests/                  # 25 个 pytest 用例（引擎单测 + 存储/API 端到端）
```

## 运行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
# 浏览器打开 http://127.0.0.1:8000/ ，点「载入演示数据」
# 数据库路径可用环境变量 DQ_CALIBRATION_DB 覆盖（默认 app/data/dq.db）
```

测试：

```bash
.venv/bin/python -m pytest tests/ -q
```

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/versions` | 创建 draft 版本（权重/阈值/豁免窗口/冲突组/门禁全在 body） |
| POST | `/api/versions/{id}/activate` | 激活版本（旧 active 自动 archived） |
| GET | `/api/versions` / `/active` / `/{id}` | 版本列表/详情（含 checksum） |
| GET | `/api/versions/{from}/diff/{to}` | 版本对照（权重、阈值、豁免窗口、门禁等差异） |
| POST | `/api/runs?version_id=` | 提交批次评分；省略版本用 active；重复 `(version,dataset,batch)` 幂等返回 |
| GET | `/api/runs` / `/api/runs/{id}` | 运行列表 / 详情（含规则拆解与样本证据快照） |
| GET | `/api/datasets/{ds}/batches/{b}/runs` | 同一批次跨版本的全部运行（重评历史） |
| GET | `/api/alerts` | 告警（绑定 run 与版本） |
| POST | `/api/demo/seed` | 幂等写入演示数据 |

## 页面

单页应用（`/ui/index.html`，原生 JS，无构建步骤）包含五个页签：

1. **运行记录**：状态/总分/门禁/版本一览，点进详情看评分构成（逐规则权重与
   规则分）、冲突分析、门禁阻断原因、逐样本 evidence 与豁免标记；
2. **提交批次**：可直接提交 `samples: []` 验证空批次，也支持指定版本与 backfill；
3. **校准版本**：版本列表、激活、配置查看、新建 draft，以及两版本差异对照；
4. **发布门禁**：所有运行按各自所钉版本的策略呈现放行/阻断；
5. **告警**：按运行与版本归档的告警流。

## 演示数据覆盖

种子数据（`POST /api/demo/seed`，可重复调用）包含：健康批次（scored 放行）、
单规则失败（scored 但门禁阻断）、规则冲突、样本不足、空批次、全豁免批次，
以及同一批次在 v1.0/v2.0 下的回补重评（历史 v1 结论不变、v2 按新策略阻断）。
