// ============================================================
// src/attn.js
// 手算LM —— 带 Attention 记录的生成（Attention 直播的数据源）
// 每生成一个 token，记录它对前文所有位置的注意力分布（多头多层平均）。
// ============================================================

import { forward } from './model.js'
import { softmaxRows } from './matrix.js'

/**
 * 生成并记录每步 attention。
 * 返回 { seq, attnSteps }：
 *   seq: 完整 token 序列（含 prompt）
 *   attnSteps[i]: 第 i 个生成 token 的注意力分布 [contextLen]（多头多层平均，和为 1）
 */
export function sampleWithAttn(params, idx, maxNewTokens, cfg, opts = {}) {
  const temperature = opts.temperature ?? 1.0
  const topK = opts.topK ?? null
  const seq = idx.slice()
  const attnSteps = []

  for (let i = 0; i < maxNewTokens; i++) {
    const ctx = seq.length > cfg.block_size ? seq.slice(-cfg.block_size) : seq
    const { logits, cache } = forward(params, ctx, null, cfg)
    const T = ctx.length

    // 提取最后位置（当前正在生成）对所有前文的注意力，多头多层平均
    const avg = new Array(T).fill(0)
    let count = 0
    for (let l = 0; l < cfg.n_layer; l++) {
      const attProbs = cache.blocks[l].attn.cache.attProbs // [nHead][T][T]
      for (let h = 0; h < cfg.n_head; h++) {
        for (let s = 0; s < T; s++) avg[s] += attProbs[h][T - 1][s]
        count++
      }
    }
    for (let s = 0; s < T; s++) avg[s] /= count
    attnSteps.push(avg)

    // 采样下一个 token（同 sample）
    const last = logits[logits.length - 1]
    const scaled = last.map((v) => v / temperature)
    if (topK && topK > 0 && topK < scaled.length) {
      const sorted = scaled.slice().sort((a, b) => b - a)
      const thr = sorted[topK - 1]
      for (let j = 0; j < scaled.length; j++) if (scaled[j] < thr) scaled[j] = -Infinity
    }
    const probs = softmaxRows([scaled])[0]
    seq.push(sampleFrom(probs))
  }
  return { seq, attnSteps }
}

function sampleFrom(probs) {
  let r = Math.random()
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i]
    if (r <= 0) return i
  }
  return probs.length - 1
}
