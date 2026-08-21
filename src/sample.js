// ============================================================
// src/sample.js
// 手算LM —— 采样核心 + 从模型生成文本
// ============================================================

import { forward } from './model.js'
import { softmaxRows } from './matrix.js'

/**
 * 采样核心：temperature 缩放 → topK/topP 裁剪 → softmax。
 * 返回概率分布（数组）。opts: { temperature, topK, topP }
 */
export function sampleProbs(logits, opts = {}) {
  const temperature = opts.temperature ?? 1.0
  const topK = opts.topK ?? null
  const topP = opts.topP ?? null
  const scaled = logits.map((v) => v / temperature)

  // topK：只保留概率最高的前 k 个候选
  if (topK && topK > 0 && topK < scaled.length) {
    const sorted = scaled.slice().sort((a, b) => b - a)
    const thr = sorted[topK - 1]
    for (let j = 0; j < scaled.length; j++) if (scaled[j] < thr) scaled[j] = -Infinity
  }

  // topP（nucleus sampling）：按概率从高到低累积，保留累计概率达 p 的候选
  if (topP && topP > 0 && topP < 1) {
    const probs0 = softmaxRows([scaled])[0]
    const order = probs0.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])
    let cum = 0
    const keep = new Set()
    for (const [v, i] of order) {
      cum += v
      keep.add(i)
      if (cum >= topP) break
    }
    for (let j = 0; j < scaled.length; j++) if (!keep.has(j)) scaled[j] = -Infinity
  }

  return softmaxRows([scaled])[0]
}

/** 按概率采样一个索引 */
export function sampleFrom(probs) {
  let r = Math.random()
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i]
    if (r <= 0) return i
  }
  return probs.length - 1
}

/**
 * 生成：输入初始 token 序列，逐 token 采样续写。
 * 返回完整 token 序列（含 prompt）。
 * opts: { temperature, topK, topP }
 */
export function sample(params, idx, maxNewTokens, cfg, opts = {}) {
  const seq = idx.slice()
  for (let i = 0; i < maxNewTokens; i++) {
    const ctx = seq.length > cfg.block_size ? seq.slice(-cfg.block_size) : seq
    const { logits } = forward(params, ctx, null, cfg)
    const probs = sampleProbs(logits[logits.length - 1], opts)
    seq.push(sampleFrom(probs))
  }
  return seq
}