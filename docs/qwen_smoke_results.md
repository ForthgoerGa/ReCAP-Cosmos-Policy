# Qwen 图像重排接入与试跑结果

实现已在 ROG 权威仓库的 feat/qwen-retrieval-rerank 分支完成，未提交或推送。
计算在 a2 / lyg2150 完成：GPU 1 用于索引、基线和 K=1；GPU 2 用于 K=32。

## 结果

| Seed | 原状态 | 原 coverage | Qwen K=32 | Qwen coverage |
|---|---|---:|---|---:|
| 42 | 成功 | 85.2% | 失败 | 34.4% |
| 43 | 成功 | 85.1% | 失败 | 0.0% |
| 44 | 成功 | 87.5% | 成功 | 88.8% |
| 45 | 失败 | 78.0% | 成功 | 85.4% |
| 合计 | 3/4 | 83.9% | 2/4 | 52.2% |

这是四种子的接入试跑，不是统计充分的优劣结论。
新策略改善 seed 45，但使 42、43 退化，当前配置不适合作为优于基线的默认方案。

逐种子视频页面：ROG 的 ../qwen_eval_results_20260909/index.html；RBS-PC 的 /home/CNS2026401561/ReCap/qwen_eval_results_20260909/index.html

## 验证与实际开销

- 7 项测试通过；覆盖旧实现对照、尾部填充对齐、重排、错误传播、日志去重。
- 完整示教池 200 条示教、12,951 个候选；200 个随机查询的新旧 payload 完全一致。
- 原状态四次 rollout 的 combined 视频与之前正式评估对应视频逐字节一致。
- K=1、seed 45：38 次检索的候选、图像哈希、动作/proprio、最终动作完全一致，combined 视频一致。
- 真正执行 Qwen 重排 124 次，其中 123 次改变原状态 Top-1。
- Qwen K=32 检索耗时 p50 57.8 ms、p95 63.0 ms；原状态约 1.5 ms。
- Qwen 自身峰值 PyTorch 分配约 4.0 GiB；同卡 Cosmos + Qwen 采样峰值 10,523 MiB。
- Qwen 四次 episode 无异常，所有运行已完成，GPU 进程已退出。
- 完整视频解码结果见 video_validation.json。
- BF16 单帧与批处理同图向量余弦一致性 0.999529，存在数值差异，索引批大小为 8。

## 代码入口

权威仓库：
/home/CNS2026401561/Projects/ReCAP/ReCAP-Cosmos-Policy （ROG Ubuntu）

- docs/qwen_retrieval.md：调用关系、环境、索引、运行命令。
- configs/retrieval/qwen_rerank.yaml：Top-K、模型与索引等参数。
- cosmos_policy/experiments/robot/pusht_ret/retrievers/：编码器、独立 worker/client、索引校验、重排策略。
- cosmos_policy/scripts/retrieval/：建索引、启动消融、汇总结果。
- tests/retrieval/test_qwen_rerank.py：兼容性与错误路径验证。
- 现有 retrieval.py 抽出状态候选与片段读取；run_eval.py 新增策略入口与可选诊断。
- 不改 Cosmos policy 输入布局、checkpoint、残差统计或训练路径；无连续性逻辑。

a2 运行配置：
/home/chenyuhong/Projects/ReCAP/qwen_setup/runtime_env.sh
独立 Qwen 环境：
/home/chenyuhong/.conda/envs/ReCap-Qwen
索引：
/home/chenyuhong/Projects/ReCAP/qwen_indices/base_2b_image_v1

模型固定为 Qwen/Qwen3-VL-Embedding-2B，
revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda。
权重、数据、源码和依赖版本均记录校验值。

