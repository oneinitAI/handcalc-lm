// ============================================================
// src/microscope.js
// 手算LM —— 显微镜模式：用当前模型真实权重，逐步展示一个 token 的前向传播
// "给你权重集和纸笔，你可以把 LLM 的下一句话算出来"
// ============================================================

import { forward } from './model.js'
import { softmaxRows } from './matrix.js'

function argmax(arr) {
  let bi = 0
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[bi]) bi = i
  return bi
}

/**
 * 显微镜前向：对输入序列做解释性 forward，返回逐步数据。
 * idx: [T] token 数组；预测下一个字（T 位置之后）。
 * 返回 { context, steps, predicted }
 */
export function runMicroscope(params, idx, cfg) {
  const { logits, cache } = forward(params, idx, null, cfg)
  const T = idx.length
  const nEmb = cfg.n_embd
  const block = cache.blocks[0]
  const attn = block.attn.cache
  const mlp = block.mlp.cache

  const steps = []

  // ① 查词向量表（wte）
  steps.push({
    title: '① 查词向量表 wte',
    desc: '每个字查 wte 表，变成一个固定长度的数字向量',
    rows: idx.map((id, t) => ({
      token: `第${t + 1}字`,
      vec: cache.tokEmb[t].map((v) => +v.toFixed(3)),
    })),
  })

  // ② 叠加位置编码（wpe）
  steps.push({
    title: '② 叠加位置编码 wpe',
    desc: '每个位置加上 wpe 的位置向量，让模型知道字的先后',
    rows: idx.map((id, t) => ({
      token: `位置${t + 1}`,
      vec: cache.posEmb[t].map((v) => +v.toFixed(3)),
    })),
  })

  // ③ 自注意力（当前最后位置对所有前文的关注）
  const lastProb = []
  const nHead = cfg.n_head
  for (let s = 0; s < T; s++) {
    let acc = 0
    for (let h = 0; h < nHead; h++) acc += attn.attProbs[h][T - 1][s] / nHead
    lastProb.push(+acc.toFixed(3))
  }
  steps.push({
    title: '③ 自注意力',
    desc: '最后一个字对前面每个字做"注意力加权"——权重就是 attention',
    attnProb: lastProb,
    attnQ: attn.q[0][T - 1].map((v) => +v.toFixed(3)),
    attnK: attn.k[0].map((row) => row.map((v) => +v.toFixed(3))),
  })

  // ④ 前馈网络 FFN
  steps.push({
    title: '④ 前馈网络 FFN',
    desc: '对向量做两次线性变换，中间夹 GELU 激活',
    h1: mlp.h1[T - 1].map((v) => +v.toFixed(3)),
    act: mlp.act[T - 1].map((v) => +v.toFixed(3)),
  })

  // ⑤ 输出分数 logits
  steps.push({
    title: '⑤ 输出分数 logits',
    desc: '把向量乘到词表 wte 上，每个候选字得到一个分数',
    logits: logits[T - 1].map((v) => +v.toFixed(3)),
  })

  // ⑥ Softmax 概率
  const probs = softmaxRows([logits[T - 1]])[0]
  steps.push({
    title: '⑥ Softmax → 概率',
    desc: '把分数变成概率，概率最高的就是模型预测的下一个字',
    probs: probs.map((v) => +v.toFixed(4)),
    top: argmax(probs),
  })

  return { context: idx.slice(), steps, predicted: argmax(probs), vocab: cfg.vocab_size }
}
