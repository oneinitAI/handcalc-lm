// ============================================================
// src/notes.js
// 手算LM —— 双层讲解内容（纸张翻面：正面直觉 / 背面公式）
// novice: 新手层（页边楷体批注）；expert: 进阶层（公式/实现细节）
// ============================================================

export const NOTES = {
  dataCard: {
    novice: '模型从这些字里学规律——它不识字义，只学"哪个字后面常跟哪个字"。',
    expert: '字符级 tokenizer + 自监督 next-token prediction：训练目标 P(w_t | w_<t)。',
  },
  modelCard: {
    novice: 'loss=猜得有多烂，训练就是让 loss 往下掉。学习率=步子大小：太大一步跨过头（训崩），太小走得慢。',
    expert: 'L = -Σ log P(w_t|w_<t)。Adam 更新：m=β₁m+(1-β₁)g, v=β₂v+(1-β₂)g², θ-=lr·m̂/(√v̂+ε)。',
  },
  sftCard: {
    novice: '换数据格式，模型从"接着写"变成"回答问题"。问答对写少了没效果，写多了只会背你写的。',
    expert: 'SFT：在预训练权重上，用 <u>/<a> 标记的问答对继续监督训练（small lr）。',
  },
  genCard: {
    novice: 'temperature：0.1=每次都选最稳的字（乏味），1.5=爱冒险（乱）。Attention 直播：每生成一个字，模型在看前文的哪些字。',
    expert: '采样：logits/T → softmax → 多项式采样。注意力 = softmax(QKᵀ/√d)，行和=1。',
  },
  dpoCard: {
    novice: '你告诉模型哪个回答更好，它学会偏向你的偏好——这正是 OpenAI 标注员做的真实工作。β 越小越接近原模型。',
    expert: 'L_DPO = -log σ(β·log πθ(yw)/πref(yw) - β·log πθ(yl)/πref(yl))（Rafailov 2023，Eq.7）。',
  },
}