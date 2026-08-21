// ============================================================
// src/attn.js
// 手算LM —— 带 Attention 记录的生成（Attention 直播的数据源）
// 每生成一个 token，记录它对前文所有位置的注意力分布（多头多层平均）。
// ============================================================

import { forward } from './model.js'
import { sampleProbs, sampleFrom } from './sample.js'

/**
 * 生成并记录每步 attention。
 * 返回 { seq, attnSteps }：
 *   seq: 完整 token 序列（含 prompt）
 *   attnSteps[i]: 第 i 个生成 token 的注意力分布 [contextLen]（多头多层平均，和为 1）
 */
export function sampleWithAttn(params, idx, maxNewTokens, cfg, opts = {}) {
  const seq = idx.slice()
  const attnSteps = []
  const probsSteps = []

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

// 采样下一个 token（复用统一采样核心，支持 topK/topP/重复惩罚）
    const probs = sampleProbs(logits[logits.length - 1], { ...opts, generated: seq })
    seq.push(sampleFrom(probs))
  }
  return { seq, attnSteps, probsSteps }
}
