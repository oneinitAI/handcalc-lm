// ============================================================
// src/linear.js
// AI 学习本 —— 线性层 + MLP（含 GELU）forward + backward
// ============================================================

import { zeros } from './matrix.js'
import { gelu, dgelu } from './layers.js'

/** 线性层前向：y[t][c] = Σ_j x[t][j]·W[j][c] + b[c] */
export function linearForward(x, W, b) {
  const T = x.length
  const cols = W[0].length
  const y = new Array(T)
  for (let t = 0; t < T; t++) {
    y[t] = new Array(cols)
    for (let c = 0; c < cols; c++) {
      let acc = b ? b[0][c] : 0
      for (let j = 0; j < W.length; j++) acc += x[t][j] * W[j][c]
      y[t][c] = acc
    }
  }
  return y
}

/** 线性层反向：返回 { dx, dW, db } */
export function linearBackward(dy, x, W, b) {
  const T = x.length
  const dW = zeros(W.length, W[0].length)
  const db = b ? zeros(1, b[0].length) : null
  const dx = new Array(T)
  for (let t = 0; t < T; t++) {
    dx[t] = new Array(W.length).fill(0)
    for (let c = 0; c < W[0].length; c++) {
      if (db) db[0][c] += dy[t][c]
      for (let j = 0; j < W.length; j++) {
        dW[j][c] += x[t][j] * dy[t][c]
        dx[t][j] += dy[t][c] * W[j][c]
      }
    }
  }
  return { dx, dW, db }
}

/** MLP 前向：y = c_proj(GELU(c_fc(x))) */
export function mlpForward(x, cfg, p, prefix) {
  const h1 = linearForward(x, p[prefix + 'c_fc.w'].value, cfg.bias ? p[prefix + 'c_fc.b'].value : null)
  const T = h1.length
  const act = new Array(T)
  for (let t = 0; t < T; t++) {
    act[t] = new Array(h1[t].length)
    for (let j = 0; j < h1[t].length; j++) act[t][j] = gelu(h1[t][j])
  }
  const y = linearForward(act, p[prefix + 'c_proj.w'].value, cfg.bias ? p[prefix + 'c_proj.b'].value : null)
  return { y, cache: { h1, act, x } }
}

/** MLP 反向：返回 c_fc 梯度(dW/db) + c_proj 梯度(dW2/db2) + 输入梯度 dx */
export function mlpBackward(dy, cache, cfg, p, prefix) {
  const { h1, act, x } = cache
  const back1 = linearBackward(dy, act, p[prefix + 'c_proj.w'].value, cfg.bias ? p[prefix + 'c_proj.b'].value : null)
  const T = h1.length
  const dact = new Array(T)
  for (let t = 0; t < T; t++) {
    dact[t] = new Array(h1[t].length)
    for (let j = 0; j < h1[t].length; j++) dact[t][j] = back1.dx[t][j] * dgelu(h1[t][j])
  }
  const back0 = linearBackward(dact, x, p[prefix + 'c_fc.w'].value, cfg.bias ? p[prefix + 'c_fc.b'].value : null)
  return { dx: back0.dx, dW: back0.dW, db: back0.db, dW2: back1.dW, db2: back1.db }
}
