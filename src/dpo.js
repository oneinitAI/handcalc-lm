// ============================================================
// src/dpo.js
// AI 学习本 —— Direct Preference Optimization（DPO）
// 参考：Rafailov et al. 2023《Direct Preference Optimization》
//   L_DPO = -log σ(β·log(πθ(yw|x)/πref(yw|x)) - β·log(πθ(yl|x)/πref(yl|x)))
// 只累计"回答部分"的对数概率（prompt 部分 mask 掉，与标准实现一致）。
// ============================================================

import { forward, zeroGrad } from './model.js'
import { backward } from './backward.js'
import { optStep } from './train.js'
import { softmaxRows } from './matrix.js'

/** 冻结参考模型：深拷贝所有权重 value（不含 grad） */
export function makeRefModel(params) {
  const ref = {}
  for (const key in params) ref[key] = { value: JSON.parse(JSON.stringify(params[key].value)) }
  return ref
}

/** 回答部分的对数概率：logπ(y|x) = Σ_{t∈回答} log p(seq[t+1] | seq[≤t]) */
function seqLogProb(logits, seq, xLen) {
  const T = logits.length
  let lp = 0
  for (let t = xLen - 1; t < T - 1; t++) {
    const row = logits[t]
    let mx = -Infinity
    for (const v of row) if (v > mx) mx = v
    let sum = 0
    for (const v of row) sum += Math.exp(v - mx)
    lp += Math.log(Math.exp(row[seq[t + 1]] - mx) / sum)
  }
  return lp
}

/** 构造回答部分的 dlogits：coeff·(onehot(target) - probs)，prompt 部分为 0 */
function makeDlogits(logits, seq, xLen, coeff) {
  const T = logits.length
  const vocab = logits[0].length
  const probs = softmaxRows(logits)
  const d = new Array(T)
  for (let t = 0; t < T; t++) {
    d[t] = new Array(vocab).fill(0)
    if (t >= xLen - 1 && t < T - 1) {
      const target = seq[t + 1]
      for (let v = 0; v < vocab; v++) d[t][v] = coeff * ((v === target ? 1 : 0) - probs[t][v])
    }
  }
  return d
}

/** 计算某个序列"回答部分"的对数概率（供验证/UI 用） */
export function answerLogProb(params, seq, xLen, cfg) {
  const { logits } = forward(params, seq, null, cfg)
  return seqLogProb(logits, seq, xLen)
}

/**
 * DPO 训练一步。x/prompt 为 token 数组，yw/yl 为两个回答的 token 数组。
 * πθ = params（可更新），πref = refParams（冻结）。
 * 返回 { loss }。
 */
export function dpoTrainStep(params, refParams, cfg, opt, x, yw, yl, beta = 0.1) {
  const seqW = [...x, ...yw]
  const seqL = [...x, ...yl]
  const xLen = x.length

  // 4 次前向：πθ/πref × yw/yl
  const fW = forward(params, seqW, null, cfg)
  const fL = forward(params, seqL, null, cfg)
  const rW = forward(refParams, seqW, null, cfg)
  const rL = forward(refParams, seqL, null, cfg)

  const lpThetaW = seqLogProb(fW.logits, seqW, xLen)
  const lpThetaL = seqLogProb(fL.logits, seqL, xLen)
  const lpRefW = seqLogProb(rW.logits, seqW, xLen)
  const lpRefL = seqLogProb(rL.logits, seqL, xLen)

  // 隐式奖励
  const rewW = beta * (lpThetaW - lpRefW)
  const rewL = beta * (lpThetaL - lpRefL)
  const z = rewL - rewW

  // 稳定 softplus：loss = log(1+e^z)
  const loss = z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z))

  // 梯度系数：dL/dlpθ(yw) = -β·σ(z)，dL/dlpθ(yl) = +β·σ(z)
  const sig = 1 / (1 + Math.exp(-z))
  const coeffW = -beta * sig
  const coeffL = beta * sig

  // 构造回答部分梯度并通过现有 backward 传播
  const dlogitsW = makeDlogits(fW.logits, seqW, xLen, coeffW)
  const dlogitsL = makeDlogits(fL.logits, seqL, xLen, coeffL)
  zeroGrad(params)
  backward(params, fW.cache, dlogitsW, cfg)
  backward(params, fL.cache, dlogitsL, cfg)
  optStep(params, opt)

  return { loss }
}
