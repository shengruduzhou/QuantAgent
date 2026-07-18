# SHADOW_TEST_PROTOCOL — 盲化前向 7 交易日运营影子测试（H-029 PASS 后启动）

> 启动日 = 2026-07-15（day-1 = 首个完整链路运行：数据→打分→S1-S3 决策→模拟成交→加密→哈希链）。
> **影子集 = ledger 中自 day-1 起的前 7 个交易日记录**。cron 已装（工作日 16:30，绝对路径）。

## 每日自动检查（runner 内建，健康记录可见）

- data_status / prediction_status / order_generation_status / fill_status 全 OK
- failed_job_count = 0；schema_hash 稳定；weights_hash 逐日记录

## Day-7 验证清单（人工/脚本执行后方可宣布 schedule 正式运营）

1. `fresh_blind_status.py`：哈希链 VALID，7/7 交易日记录无缺；
2. 7 日 schema_hash 全等（面板 schema 无漂移）；
3. 每日 S1/S2/S3 weights_hash 存在且 order_logs/fill_logs 对应文件齐全；
4. encrypted_performance/ 每日 3 文件（S1-S3），**零解密读取**（访问=违规，留痕）；
5. 打分步逐日引用 passes=true 证书（cert hash 不变）；
6. cron.log 无未捕获异常；
7. S4 learner port 状态复核（PENDING 项：regime_weight_meta 日频移植，读日前完成或首读时批式构建 S4 书——freeze manifest 许可）。

## 已知限制（如实）

- 盲化为程序性（密钥本机；纪律=工具拒读+哈希链留痕）；
- backlog >1 交易日时 runner 数据步会 TIMEOUT fail-closed → 需手动 standalone 追补（day-1 即如此处理，2026-07-15 实测 9 日 backlog >3600s）；
- S4 决策暂缺（上行 PENDING 项）；S1-S3 决策自 day-1 起真实生成。
