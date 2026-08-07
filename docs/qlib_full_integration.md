# Qlib 全参考融合与 QuantAgent 治理映射

> 核验基准：2026-08-07。用户给出的 `qlib.org.cn/en/latest/` 文档树被逐项登记；
> 当前 PyPI 稳定版为 **pyqlib 0.9.7**。QuantAgent 采用“上游原生 Qlib
> 运行时 + QuantAgent PIT/泄漏/滚动 OOS 治理 + 可审计桥接”的方式融合，
> 不 fork Qlib，也不把 Qlib 的研究便利功能当成最终晋级规则。

## 1. 为什么原仓库不是“Qlib 全融合”

原仓库已经有：

- `QlibProvider.daily_ohlcv`：本地 Qlib 日线 OHLCV；
- Qlib CN bootstrap 与 market panel；
- 独立的 `available_at` / PIT / leakage audit；
- QuantAgent 自研回测、执行、RL 与模型比较。

但此前没有一个覆盖 Qlib 全文档树的契约，也没有统一桥接：

- qrun-style Workflow；
- DatasetH / DataHandlerLP / StaticDataLoader；
- Model/自定义 Model；
- Recorder / Experiment / Record templates；
- Strategy / Backtest；
- OnlineManager；
- TrainerR / DelayTrainerR / TrainerRM / TaskManager；
- Nested Decision 高频执行；
- MetaTask / MetaDataset / MetaModel；
- Qlib RL；
- 公式化 Alpha / Alpha158 / Alpha360；
- Serialization / server mode；
- 上游 API/FAQ/changelog 兼容性审计。

因此不能把“安装了 pyqlib + 能读日线”称为完整融合。

## 2. 现在的融合边界

### QuantAgent 保留控制权

Qlib 不覆盖或不能替代下列 QuantAgent 规则：

1. 每条历史输入必须满足 `available_at <= decision_time`；
2. 财务数据以披露/可获得时间为准，不以报告期末伪造可用性；
3. 学习型预处理只在训练段拟合；
4. purged walk-forward + embargo 由 QuantAgent 先生成受治理的 segment；
5. final holdout 只验收一次，不用 Qlib 的 Record/回测反复选参；
6. benchmark 必须显式指定；
7. 交易成本、涨跌停、T+1、停牌和容量约束不得因 Qlib baseline 而弱化；
8. PBO / DSR / SPA 等晋级闸门仍属于 QuantAgent；
9. live/paper execution 必须经过 QuantAgent operating mode / risk gates。

### Qlib 负责它最擅长的部分

- 数据表达式、Dataset/DataHandler/Processor；
- 上游模型和自定义 Model 接口；
- qrun/task workflow；
- Recorder/Experiment/Record；
- TopkDropout 等策略 baseline 与 backtest；
- signal/portfolio analysis；
- OnlineManager 与 rolling task；
- Nested Decision / 高频执行研究；
- Meta learning；
- RL research/execution components；
- task management；
- serialization/server mode。

这不是重复造一套 Qlib，而是让 QuantAgent 能稳定调用完整上游能力。

## 3. 27 个参考页面完整映射

机器清单：`src/quantagent/qlib/catalog.py`。任何 ID 数量漂移都会使测试失败。

| # | 参考 | QuantAgent 融合 |
|---:|---|---|
| 1 | Introduction | 架构/设计治理参考 |
| 2 | Quick Start | `qlib-*` CLI 首次运行语义 |
| 3 | Installation | `pyqlib>=0.9.7,<0.10` research extra |
| 4 | Initialization | `QlibRuntime.initialize` |
| 5 | Get Data | `calendar/instruments/features` runtime bridge |
| 6 | Custom Model Integration | 任意 Qlib Model config / custom Model |
| 7 | Workflow | `run_workflow_config`，qrun core |
| 8 | Data | QuantAgent PIT Parquet -> StaticDataLoader -> DataHandlerLP/DatasetH |
| 9 | Model | 上游模型/自定义模型 config bridge |
| 10 | Strategy | TopkDropout/PortAna baseline |
| 11 | Highfreq | Nested Decision config bridge + QuantAgent execution gates |
| 12 | Meta | MetaTask/MetaDataset/MetaModel config bridge |
| 13 | Recorder | Qlib `R` + Experiment/Recorder + Record artifacts |
| 14 | Report | SigAna/PortAna + evaluate helpers |
| 15 | Online | OnlineManager/OnlineStrategy config bridge |
| 16 | RL | Qlib RL config bridge + QuantAgent PIT RL |
| 17 | Formulaic Alpha | Qlib expression engine、Alpha158/Alpha360 benchmark |
| 18 | Server | 显式 local/client provider mode；不隐式联网 |
| 19 | Serialization | 上游 Serializable/Recorder artifact，保留版本与 provenance |
| 20 | Task Management | TrainerR/DelayTrainerR/TrainerRM；Mongo TaskManager opt-in |
| 21 | PIT | 与 QuantAgent `available_at`/披露时间语义对齐 |
| 22 | Code Standard | 兼容性/开发治理参考 |
| 23 | Development Guidance | 兼容性/开发治理参考 |
| 24 | Build Image | 部署参考，不强制 Docker |
| 25 | API | bridge API 兼容性参考 |
| 26 | FAQ | 运维/限制参考 |
| 27 | Changelog | 上游版本漂移审计 |

完整 URL 由 registry 固定，不靠本文件手抄计数。

## 4. Qlib 0.9.7 对 QuantAgent 的关键价值

Qlib v0.9.7 的上游发布包含：

- Parquet data support；
- `BaseDataHandler` / unified `fetch` 接口重构；
- MLflow 配置更新；
- `risk_analysis` geometric accumulation；
- horizon utility。

其中 Parquet 支持最关键：`StaticDataLoader` 在 0.9.7 可直接
`pd.read_parquet(..., engine="pyarrow")`，因此 QuantAgent 不需要把已经治理好的
silver/gold 数据重新 dump 成 Qlib bin 才能做模型实验。

QuantAgent 新增：

```text
Fuyao / TickFlow / PIT sources
           |
           v
QuantAgent raw -> silver/gold
           |
           | available_at + leakage guards
           v
prepare-qlib-parquet
           |
           v
Qlib StaticDataLoader
           |
           v
DataHandlerLP -> DatasetH
           |
           v
Model -> Recorder -> SignalRecord/SigAnaRecord/PortAnaRecord
           |
           v
Qlib baseline result
           |
           v
QuantAgent OOS / PBO / DSR / SPA / holdout promotion gates
```

## 5. PIT Parquet 规则

`prepare-qlib-parquet` 默认使用：

```text
Qlib datetime := QuantAgent available_at
Qlib instrument := SH600519 / SZ000001 / ...
```

列变为：

```text
(feature, quality)
(feature, momentum)
(label, forward_return_20d)
...
```

硬约束：

- feature 名中出现 `forward/future/lead/label/target` 类标签语义直接阻塞；
- feature 与 label 不允许重叠；
- `(datetime, instrument)` 必须唯一；
- `available_at` 不能缺失；
- label 可以包含未来收益，但只能在 Qlib `label` group，不能混入 inference feature；
- Dataset segment 仍必须由 QuantAgent purge/embargo 后再交给 Qlib。

示例：

```bash
quantagent prepare-qlib-parquet \
  --input data/v7/silver/market_panel/research_panel.parquet \
  --output data/qlib/research.parquet \
  --features quality,value,momentum,low_vol,liquidity \
  --labels forward_return_20d \
  --time-column available_at
```

## 6. 构建 Qlib task

模型 config 不被 QuantAgent 限死，可以使用 Qlib built-in 或用户自定义
`Model`：

```yaml
class: LGBModel
module_path: qlib.contrib.model.gbdt
kwargs:
  loss: mse
```

构建 task：

```bash
quantagent build-qlib-task \
  --parquet-path data/qlib/research.parquet \
  --model-config configs/qlib/lgb.yaml \
  --benchmark-symbol 000300.SH \
  --train-start 2016-01-01 --train-end 2021-12-31 \
  --valid-start 2022-03-01 --valid-end 2023-12-31 \
  --test-start 2024-03-01 --test-end 2026-07-31 \
  --minimum-gap-days 59 \
  --provider-uri ~/.qlib/qlib_data/cn_data \
  --output data/qlib/workflow.yaml
```

`minimum-gap-days` 是额外的保守 calendar gap 检查；真正按交易日标签 horizon
执行的 purge/embargo 应先由 QuantAgent splitter 完成。

运行：

```bash
quantagent run-qlib-workflow \
  --config data/qlib/workflow.yaml
```

底层使用 Qlib 0.9.7 的 `task_train` 语义：初始化 Model/Dataset、fit、保存
model/dataset，然后按 `record` 生成预测、信号分析和组合回测。

## 7. Recorder / Experiment

Qlib Recorder 用于：

- params；
- metrics；
- model/dataset objects；
- prediction；
- SignalRecord；
- SigAnaRecord；
- PortAnaRecord；
- tags/artifacts。

QuantAgent 仍保留自己的 DataManifest/evidence/holdout ledger。两者用途不同：
Recorder 是实验运行账本，QuantAgent manifest 是数据/决策治理账本。

## 8. Online / Task Management

`QlibRuntime.train_tasks` 支持：

- `mode=recorder` -> `TrainerR`；
- `mode=delay` -> `DelayTrainerR`；
- `mode=task_manager` -> `TrainerRM`。

`TrainerRM/TaskManager` 通常需要 MongoDB，因此默认不启用。不能为了“全融合”
偷偷引入一个常驻数据库。

OnlineManager/OnlineStrategy 通过 native config bridge 构建。它们可用于 rolling
model/prediction/update simulation，但真正的实盘/仿真委托仍必须经过 QuantAgent
execution/safety 层。

## 9. Highfreq Nested Decision

Qlib 的 Nested Decision Execution 解决“日频组合决策 + 更细粒度执行”联合回测。
QuantAgent 的融合原则：

- outer decision：QuantAgent portfolio/risk target；
- inner decision：Qlib nested executor 或 QuantAgent execution agent；
- exchange/tradability/cost：不得弱于 QuantAgent A 股约束；
- RL 只能优化 execution policy，不得读取未来成交/未来 bar；
- 高频结果必须与同一成本口径的非 RL baseline 对照。

## 10. Meta / RL

Meta 和 RL 不是默认晋级模型；它们属于研究 arm：

- MetaTask / MetaDataset / MetaModel 通过 `QlibRuntime.instantiate`；
- Qlib RL state/action/reward/simulator/vessel 同样通过 upstream config；
- QuantAgent PIT RL 环境继续保留；
- 两者共享相同 OOS/PIT/cost/tradability 约束后才比较；
- 不允许因复杂度更高而自动获得更高 ranking。

## 11. Formulaic Alpha

Qlib 表达式引擎与 Alpha158/Alpha360 用作：

1. 独立 baseline；
2. 因子候选生成器；
3. 与 QuantAgent 101 Formulaic Alphas/经典因子交叉验证的参考。

Qlib 自带示例 label 使用负方向 `Ref` 取得未来收益，因此 label expression
只能位于训练 label；任何相同语义被塞入 feature 都属于 leakage。

## 12. Live 文档/版本审计

静态覆盖：

```bash
quantagent qlib-capabilities
quantagent audit-qlib-coverage \
  --output data/qlib/coverage.json
```

显式联网核验 27 个文档 URL 和 PyPI stable：

```bash
quantagent audit-qlib-coverage \
  --live-docs \
  --allow-network \
  --output data/qlib/coverage-live.json
```

网络永远不是隐式依赖。将来 Qlib 进入 0.10/1.x 时 live audit 会把超出当前
tested 0.9.x series 视为需要重新兼容性验证，而不是静默接受。

## 13. 全融合不等于全启用

以下能力代码已可桥接，但默认关闭或仅 research mode：

- Mongo TaskManager；
- OnlineManager；
- nested high-frequency execution；
- Meta learning；
- Qlib RL；
- server/client provider mode。

原因不是“没融合”，而是这些能力改变运行基础设施或可能影响执行安全。
只有明确配置后才启用，这比把所有服务默认启动更符合量化研究的可复现性和
production safety。
