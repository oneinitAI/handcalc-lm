// ============================================================
// src/sample.js
// 手算LM —— 从模型采样生成文本
// ============================================================

import { forward } from './model.js'
import { softmaxRows } from './matrix.js'

/**
 * 生成：输入初始 token 序列，逐 token 采样续写。
 * 返回完整 token 序列（含 prompt）。
 * opts: { temperature, topK }
 */
export function sample(params, idx, maxNewTokens, cfg, opts = {}) {
  const temperature = opts.temperature ?? 1.0
  const topK = opts.topK ?? null
  const seq = idx.slice()

  for (let i = 0; i < maxNewTokens; i++) {
    const ctx = seq.length > cfg.block_size ? seq.slice(-cfg.block_size) : seq
    const { logits } = forward(params, ctx, null, cfg)
    const last = logits[logits.length - 1]

    // temperature 缩放：logits / T
    const scaled = last.map((v) => v / temperature)

    // topK 裁剪
    if (topK && topK > 0 && topK < scaled.length) {
      const sorted = scaled.slice().sort((a, b) => b - a)
      const thr = sorted[topK - 1]
      for (let j = 0; j < scaled.length; j++) if (scaled[j] < thr) scaled[j] = -Infinity
    }

    const probs = softmaxRows([scaled])[0]
    seq.push(sampleFrom(probs))
  }
  return seq
}

/** 按概率采样一个索引 */
function sampleFrom(probs) {
  let r = Math.random()
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i]
    if (r <= 0) return i
  }
  return probs.length - 1
}
