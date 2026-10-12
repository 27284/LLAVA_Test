# LLaVA 视觉信息保留与使用实验

统一入口：在项目根目录运行 `python -m experiments.run <stage>`。默认模型为
`llava-hf/llava-1.5-7b-hf`，CUDA 4-bit，默认仅读取本地缓存；允许下载时加
`--allow-download`。本地模型目录可通过 `--model` 指定。CPU 用 `--quantization none`。
推荐先 `--limit 2` 写入独立 smoke 输出目录，再运行完整实验。

## 1. 数据与 baseline

```powershell
python -m experiments.run prepare
python -m experiments.run baseline --output results/baseline
python -m experiments.run evaluate --subset data/vqav2/subset/qa_subset_300.json --predictions results/vqav2_baseline_300.csv --reclassify --output results/legacy_baseline_audit
```

新子集 `data/vqav2/subset/vqav2_validation_300.json` 每类 60 条，随机种子 42。
旧子集、旧 CSV 保留。五类采用互斥规则，yes/no 优先，再判断 color、count、spatial、object。
`count` 限定 how many，价格、年龄等数字答案不自动算计数。启发式类别仍需人工抽查；
旧 CSV 重新分类后的类别数可能不再均衡。

模型被要求只输出单词或短语。原始生成文本不会被自动截取成 GT，也不做 substring 匹配。
`baseline_evaluated.csv` 包含全部标识、类别、问题、GT、预测、soft accuracy、exact match、correct。
`baseline_summary.json` 给出类别统计、覆盖率和按图片聚类的 bootstrap 95% 区间。
`correct_subset.json` 是 S_correct，默认 soft accuracy >= 1；可调整 `--correct-threshold`。
soft accuracy 使用全部十个人工答案，按留一人工答案的方式平均，匹配 1/2/3/4 次通常为
0.3/0.6/0.9/1。保留官方 evaluator 的条件式文本规范化规则。
官方规则来源：https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
完整句子对短答案得零分是评估格式问题，不能据此直接宣称模型没有识别出颜色。

## 2–4. 局部干预、visual reliance 与 attention

```powershell
python -m experiments.run intervene --subset results/baseline/correct_subset.json --output results/intervention --grid 4 --top-k 3 --attention --plots
```

图像先将完整图片缩放到模型的方形输入尺寸；baseline、干预、特征提取使用相同预处理。
这是明确的实验选择，避免默认 center crop 导致原图区域与 visual patch 对不上；
与旧 baseline 的原始预处理可能不同，所以正式实验应重新运行 baseline。
网格按行优先编号，从 0 开始。遮挡填充默认原图均值，可用 `--fill gray` 做敏感性分析，
应写入新的输出目录。完整数据不是默认 Blank/Noise。

固定 Y 为 multiple_choice_answer，teacher forcing 累加答案 token 的条件 log probability，
不计 prompt、不计 EOS，不把 generation score 当概率。目标 tokenization 固定，
使用 Llama tokenizer 的默认词边界前缀，不手动添加额外空格 token。
同时报告 token 数和每 token 分数。每块 C_i = 原图 logP - 单块遮挡 logP。
按有符号 C_i 选固定 top-k（默认 3），联合遮挡后计算 VR，不将负值截成零。
另做默认 3 次同面积随机遮挡，记录 random VR 和 excess VR。
top-k 在同一目标上选择，存在选择效应；这些分数衡量遮挡操作下的行为，不能单凭它们证明
自然场景因果关系或语言 shortcut。可提高 `--random-controls`。

attention 使用 eager 后端，取所有 LLM 层、所有头、预测固定 GT 各 token 的 query 对
image tokens 的均值，包含预测首个答案 token 的 prompt 末尾 query。
这是 teacher-forced GT attention，不是自由生成答案 attention。
仅保存压缩后的 patch attention 和网格图，通过面积加权映射到同一干预网格。
输出 Pearson、带 ties 的 Spearman；常量图相关性写 null，不伪造零。
attention 的原始视觉质量保留，不额外归一化为总和 1。
`maps/<question_id>.json` 包含每区域分数、框坐标、top 区域及随机对照，`--plots` 生成 PNG。
若未安装 matplotlib，自动使用 Pillow 生成带逐格数值与颜色说明的图，不影响实验执行。
类别均值和置信区间见 `intervention_summary.json`，不预设颜色比 yes/no 更依赖视觉。

## 5. 逐层 linear probing

```powershell
python -m experiments.run features --output results/features --layers 2 8 16 24 32
python -m experiments.run probe --features results/features --output results/probe --interventions results/intervention
```

特征在全部 baseline 子集上提取，避免只在 S_correct 训练产生额外选择偏差。
按 1-based Transformer block 输出层号采集 visual token hidden states（最后一层为 block 输出，
不混用最终 RMSNorm 后输出），每层做 visual tokens 平均池化。
标签为规范化后的 GT 答案；颜色为颜色标签，其他类别是答案解码诊断，不应统一视为纯视觉属性。
LLaVA 1.5 image tokens 位于 question 之前，因果 mask 下这些状态不含后续 question；
因此均值池化并不提供问题所指物体的定位，复杂、多物体颜色样本可能不可解码。
后续 answer tokens 同样不能回流污染早期 visual states。

使用 NumPy ridge linear classifier，alpha 固定 10，5 折按 image_id 分组；
仅训练折标准化、确定类别与 majority baseline，同时提供随机打乱训练标签的对照。
测试折未见标签算错误并记录数量，样本或类别不足时标记跳过。默认 300 条只能用于探索；
可靠 preservation 结论需要更多带属性标注的样本、多次划分、ROI/patch probe 与控制实验。
`probe_predictions.csv` 是 out-of-fold 预测；join 文件与 VR 按 question_id 对齐。
探针答对/答错分组的 VR 统计供联合分析，不自动把单个 probe 错误判为 preservation failure。
若定义高 probe / 低 VR，阈值须事先设定并报告，不可从结果挑选阈值。

## 6. Language–Vision conflict

```powershell
python -m experiments.run conflict --manifest data/conflict/manifest.json --output results/conflict --attention --plots
```

需人工审核的 normal/conflict 配对 manifest，每对 question 与 category 相同；每条字段：

```json
{
  "question_id": 9000001,
  "image_id": 900001,
  "pair_id": "banana_pair_01",
  "condition": "conflict",
  "image_path": "images/green_banana.jpg",
  "category": "color",
  "question": "What color is the banana?",
  "answer": "green",
  "all_answers": ["green", "green", "green", "green", "green", "green", "green", "green", "green", "green"],
  "language_prior_answer": "yellow",
  "provenance": "Document actual source/edit process, human annotators, and how prior was established",
  "reviewed": true
}
```

同 pair_id 还需 normal 条目，question_id 全部唯一、image_id 为数值。
相对图片路径以 manifest 目录解析。十个答案必须来自真实人工标注，禁止将单个答案复制十次
充当 VQAv2 标注；上面仅示意字段形式。不得使用原图 GT 给改色图打分。
程序复用 baseline 和干预，输出 all-sample 与 correct-only 分组统计、配对 accuracy drop
与 VR change，bootstrap 按 pair_id 聚类。全部配对上的 VR 包含原图答错样本，
只能作为独立描述，主要解释使用 S_correct，并报告选择偏差。
程序提供执行路径，但不编造冲突图片、人工答案或实验结论。

## 运行、恢复与验证

每条推理后落盘，重跑同一命令会跳过已完成 question_id。配置/子集改变时拒绝混用旧输出，
请选新目录；使用 `--limit` 的 smoke 也应独立目录。模型/数据错误直接失败，不静默吞掉样本。
改模型本地权重或图片内容后也必须换目录（配置校验不能侦测原路径下内容替换）。
建议阶段 2–4 使用 S_correct；阶段 5 使用全子集；阶段 6 同时报告全配对与正确子集。
不能直接用原模型答错后的遮挡错误推断依赖低，也不能由低相关性直接断言语言 shortcut。

```powershell
python -m unittest discover -s tests -v
```

测试涵盖官方 soft score、类别、网格覆盖、attention 映射、相关性、image-disjoint probe，
以及完整阶段衔接、conflict 配对汇总与恢复运行。
以及随机初始化微型 LLaVA 的真实 forward：核对答案概率、attention query 索引、hook 清理及
visual states 不受后续 question 影响。微型模型不是预训练 7B 实验结果。

本次验证：9 项测试通过；本地缓存 7B 模型完成 1 条真实 baseline、4×4 遮挡、attention、
2/8/16/24/32 层特征提取，产物位于 `results/smoke_verified/`。
新平衡子集已生成；旧 CSV 审核位于 `results/legacy_baseline_audit/`。
完整 300 条新版模型实验及真实 conflict 数据实验尚未执行。
