// ============================================================
// src/train.js
// 手算LM —— 训练循环 + 优化器（SGD / AdamW）
// ============================================================

import { zeros } from './matrix.js'
import { forward, zeroGrad } from './model.js'
import { backward } from './backward.js'
import { crossEntropy } from './matrix.js'

/**
 * 创建优化器状态。
 * opts: { type: 'adam'|'sgd', lr, beta1, beta2, eps, momentum }
 */
export function createOptimizer(params, opts = {}) {
  const type = opts.type || 'adam'
  const o = {
    type,
    lr: opts.lr ?? 0.01,
    beta1: opts.beta1 ?? 0.9,
    beta2: opts.beta2 ?? 0.999,
    eps: opts.eps ?? 1e-8,
    momentum: opts.momentum ?? 0.9,
    state: {},
  }
  for (const key in params) {
    const rows = params[key].value.length
    const cols = params[key].value[0].length
    o.state[key] = type === 'adam'
      ? { m: zeros(rows, cols), v: zeros(rows, cols), t: 0 }
      : { v: zeros(rows, cols) }
  }
  return o
}

/** 用当前梯度更新所有参数（优化器一步） */
export function optStep(params, opt) {
  opt._t = (opt._t || 0) + 1
  const t = opt._t // 全局步数
  for (const key in params) {
    const { value, grad } = params[key]
    const rows = value.length
    const cols = value[0].length
    const st = opt.state[key]
    if (opt.type === 'adam') {
      const { m, v } = st
      const b1 = opt.beta1, b2 = opt.beta2, eps = opt.eps, lr = opt.lr
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const g = grad[r][c]
          m[r][c] = b1 * m[r][c] + (1 - b1) * g
          v[r][c] = b2 * v[r][c] + (1 - b2) * g * g
          const mHat = m[r][c] / (1 - Math.pow(b1, t))
          const vHat = v[r][c] / (1 - Math.pow(b2, t))
          value[r][c] -= lr * mHat / (Math.sqrt(vHat) + eps)
        }
      }
    } else {
      // SGD（带动量）
      const mu = opt.momentum
      const lr = opt.lr
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          st.v[r][c] = mu * st.v[r][c] - lr * grad[r][c]
          value[r][c] += st.v[r][c]
        }
      }
    }
  }
}

/**
 * 训练一步：forward → loss → backward → 参数更新。
 * opts.mask（可选）：mask[i]=false 的位置不训练（SFT 只学回答部分）。
 * 返回 { loss }。
 */
export function trainStep(params, idx, targets, cfg, opt, opts = {}) {
  const { logits, cache } = forward(params, idx, targets, cfg)
  const ce = crossEntropy(logits, targets, opts.mask || null)
  zeroGrad(params)
  backward(params, cache, ce.dlogits, cfg)
  optStep(params, opt)
  return { loss: ce.loss }
}
