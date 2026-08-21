// ============================================================
// src/papers.js —— 前沿论文库：通俗解析 + 原文链接
// ============================================================

export const PAPER_CATS = ['全部', '架构', '训练与缩放', '对齐', '推理加速', '多模态', '开源与服务商']

export const PAPERS = [
  // ---- 架构 ----
  { title: 'Attention Is All You Need', org: 'Google', year: 2017, cat: '架构',
    summary: 'Transformer 诞生之作——用"注意力"替代循环网络，所有 token 并行处理。今天所有大模型（包括你在本页训练的）都是它的后代。',
    url: 'https://arxiv.org/abs/1706.03762' },
  { title: 'RoFormer: RoPE 旋转位置编码', org: '苏剑林 · 追一科技', year: 2021, cat: '架构',
    summary: '把"位置"用旋转角度编码进向量——让模型既能知道字的位置，又天然支持外推到更长上下文。长上下文模型的基础。',
    url: 'https://arxiv.org/abs/2104.09864' },
  { title: 'Switch Transformer（MoE 混合专家）', org: 'Google', year: 2021, cat: '架构',
    summary: '把模型拆成上千个"专家"，每个 token 只激活少数几个——同样的参数量，算得更少。MoE 的奠基之作，DeepSeek 沿用此路。',
    url: 'https://arxiv.org/abs/2101.03961' },

  // ---- 训练与缩放 ----
  { title: 'Scaling Laws for Neural Language Models', org: 'OpenAI', year: 2020, cat: '训练与缩放',
    summary: '发现模型性能随"参数×数据×算力"按幂律增长——解释了大模型为什么越来越大，也指导了怎么分配算力。',
    url: 'https://arxiv.org/abs/2001.08361' },
  { title: 'Language Models are Unsupervised Multitask Learners（GPT-2）', org: 'OpenAI', year: 2019, cat: '训练与缩放',
    summary: '证明"预测下一个字"学出的模型能做翻译、问答、摘要等一切任务——零样本多任务能力，预训练范式的里程碑。',
    url: 'https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf' },
  { title: 'QLoRA（4-bit 量化微调）', org: '华盛顿大学', year: 2023, cat: '训练与缩放',
    summary: '把模型量化为 4 位再微调——单张消费级显卡就能微调 65B 大模型。让"个人微调大模型"成为可能。',
    url: 'https://arxiv.org/abs/2305.14314' },

  // ---- 对齐 ----
  { title: 'InstructGPT：RLHF 对齐', org: 'OpenAI', year: 2022, cat: '对齐',
    summary: '三阶段对齐的经典：预训练 → 监督微调 → 用人类反馈的强化学习（PPO）。让模型从"会续写"变成"会听话"。',
    url: 'https://arxiv.org/abs/2203.02155' },
  { title: 'DPO：Direct Preference Optimization', org: 'Stanford', year: 2023, cat: '对齐',
    summary: '数学上证明"偏好对齐可以变成一次简单分类训练"——不需要奖励模型和强化学习，两三行公式搞定对齐。',
    url: 'https://arxiv.org/abs/2305.18290' },
  { title: 'DeepSeek-R1：强化学习激发推理', org: 'DeepSeek', year: 2025, cat: '对齐',
    summary: '用纯强化学习（GRPO）让模型自发学会"深度思考"（长思维链）——数学/代码能力暴涨，还开源了完整技术，引爆全球。',
    url: 'https://arxiv.org/abs/2501.12948' },

  // ---- 推理加速 ----
  { title: 'FlashAttention', org: 'Stanford', year: 2022, cat: '推理加速',
    summary: '重写注意力计算：分块+避免把大矩阵写回内存——训练推理都快数倍，还能处理更长序列。',
    url: 'https://arxiv.org/abs/2205.14135' },
  { title: 'PagedAttention（vLLM 的 KV 缓存管理）', org: 'UC Berkeley', year: 2023, cat: '推理加速',
    summary: '把 KV 缓存像操作系统分页一样管理——内存利用率提升，服务吞吐翻倍。vLLM 推理引擎的核心。',
    url: 'https://arxiv.org/abs/2309.06180' },
  { title: 'Speculative Decoding（投机解码）', org: 'Google', year: 2023, cat: '推理加速',
    summary: '小模型先快速打草稿，大模型一次性验证多个候选——"猜得快，验得准"，生成提速 2~3 倍。',
    url: 'https://arxiv.org/abs/2211.17192' },

  // ---- 多模态 ----
  { title: 'DDPM：扩散模型', org: 'UC Berkeley', year: 2020, cat: '多模态',
    summary: '把图加噪成雪花、学去噪还原；生成时从噪声一步步出图。Stable Diffusion 等主流 AI 画图的原理。',
    url: 'https://arxiv.org/abs/2006.11239' },
  { title: 'CLIP：图文对齐', org: 'OpenAI', year: 2021, cat: '多模态',
    summary: '把图像和文字映射到同一个向量空间——"猫"的图和"猫"的文字靠在一起。跨模态理解的基石。',
    url: 'https://arxiv.org/abs/2103.00020' },
  { title: 'Whisper：语音识别', org: 'OpenAI', year: 2022, cat: '多模态',
    summary: '用互联网规模的多语言音频+文本训练语音识别——能转写 99 种语言，鲁棒性远超传统方法。',
    url: 'https://arxiv.org/abs/2212.04356' },
  { title: 'GPT-4 Technical Report', org: 'OpenAI', year: 2023, cat: '多模态',
    summary: '多模态大模型（看图+文本），展现出接近人类水平的考试与推理能力。技术细节保密，但方向明确：一个模型通吃。',
    url: 'https://arxiv.org/abs/2303.08774' },

  // ---- 开源与服务商 ----
  { title: 'Llama：开源大模型', org: 'Meta', year: 2023, cat: '开源与服务商',
    summary: '公开权重的大模型，让全世界能自己部署/微调/研究——开源生态的引爆点。',
    url: 'https://arxiv.org/abs/2302.13971' },
  { title: 'DeepSeek-V3：低成本 MoE', org: 'DeepSeek', year: 2024, cat: '开源与服务商',
    summary: '671B 参数、37B 激活的 MoE 模型，训练成本仅 550 万美元（GPT-4 的零头）——"低成本高性能"路线代表。',
    url: 'https://arxiv.org/abs/2412.19437' },
]