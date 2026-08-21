// ============================================================
// src/layers.js
// 手算LM —— 基础层：LayerNorm、GELU（含 forward + backward）
// ============================================================

import { zeros } from './matrix.js'

/** erf 近似（Abramowitz-Stegun 7.1.26），教学精度足够 */
function erfApprox(x) {
  const sign = x < 0 ? -1 : 1
  const ax = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * ax)
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax)
  return sign * y
}

/** GELU 精确版：0.5*x*(1+erf(x/√2)) */
export function gelu(x) {
  return 0.5 * x * (1 + erfApprox(x / Math.SQRT2))
}

/** GELU 导数：0.5*(1+erf(x/√2)) + x/√(2π)·e^(-x²/2) */
export function dgelu(x) {
  return 0.5 * (1 + erfApprox(x / Math.SQRT2)) + x * Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)
}

// ---------- LayerNorm ----------

/**
 * LayerNorm 前向：输入 [T][d]，沿最后一维逐行归一化。
 * w/b 形状 [1][d]。返回 { y, cache }。
 */
export function layerNormForward(x, w, b) {
  const T = x.length
  const d = x[0].length
  const y = new Array(T)
  const mean = new Array(T)
  const rstd = new Array(T)
  for (let t = 0; t < T; t++) {
    let m = 0
    for (let j = 0; j < d; j++) m += x[t][j]
    m /= d
    let v = 0
    for (let j = 0; j < d; j++) v += (x[t][j] - m) ** 2
    v /= d
    const r = 1 / Math.sqrt(v + 1e-5)
    y[t] = new Array(d)
    for (let j = 0; j < d; j++) y[t][j] = (x[t][j] - m) * r * w[0][j] + b[0][j]
    mean[t] = m
    rstd[t] = r
  }
  return { y, cache: { x, w, mean, rstd, d } }
}

/** LayerNorm 反向：返回 { dx, dw, db } */
export function layerNormBackward(dy, cache) {
  const { x, w, mean, rstd, d } = cache
  const T = x.length
  const dx = new Array(T)
  const dw = zeros(1, d)
  const db = zeros(1, d)
  for (let t = 0; t < T; t++) {
    // dxhat[j] = dy[t][j] * w[0][j]
    const dxhat = new Array(d)
    for (let j = 0; j < d; j++) {
      dxhat[j] = dy[t][j] * w[0][j]
      dw[0][j] += dy[t][j] * (x[t][j] - mean[t]) * rstd[t]
      db[0][j] += dy[t][j]
    }
    // 归一化反向（标准 LN backward，沿最后一维）：
    // dx = rstd * (dxhat - mean(dxhat) - (x-mean)·mean(dxhat·(x-mean))·rstd²)
    let m1 = 0, m2 = 0
    for (let j = 0; j < d; j++) {
      m1 += dxhat[j]
      m2 += dxhat[j] * (x[t][j] - mean[t])
    }
    m1 /= d
    m2 /= d
    dx[t] = new Array(d)
    for (let j = 0; j < d; j++) {
      dx[t][j] = rstd[t] * (dxhat[j] - m1 - (x[t][j] - mean[t]) * m2 * rstd[t] * rstd[t])
    }
  }
  return { dx, dw, db }
}
