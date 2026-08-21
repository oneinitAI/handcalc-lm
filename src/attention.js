// ============================================================
// src/attention.js
// AI 学习本 —— 因果自注意力（通用多头，batch=1）forward + backward
// ============================================================

import { zeros } from './matrix.js'

/**
 * 因果自注意力前向。x: [T][nEmb]。
 * 返回 { y, cache }（cache 供 backward 使用）。
 */
export function attentionForward(x, cAttnW, cAttnB, cProjW, cProjB, cfg) {
  const T = x.length
  const nEmb = cfg.n_embd
  const nHead = cfg.n_head
  const headDim = nEmb / nHead
  const scale = 1 / Math.sqrt(headDim)

  // 1) QKV 投影：qkv[t][i] = Σ_j x[t][j]·W[j][i] + b[i]
  const qkv = new Array(T)
  for (let t = 0; t < T; t++) {
    qkv[t] = new Array(3 * nEmb)
    for (let i = 0; i < 3 * nEmb; i++) {
      let acc = cAttnB ? cAttnB[0][i] : 0
      for (let j = 0; j < nEmb; j++) acc += x[t][j] * cAttnW[j][i]
      qkv[t][i] = acc
    }
  }

  // 2) 拆 q/k/v 并按 head 重排：[nHead][T][headDim]
  const q = [], k = [], v = []
  for (let h = 0; h < nHead; h++) {
    q.push(new Array(T))
    k.push(new Array(T))
    v.push(new Array(T))
    for (let t = 0; t < T; t++) {
      q[h][t] = new Array(headDim)
      k[h][t] = new Array(headDim)
      v[h][t] = new Array(headDim)
      for (let j = 0; j < headDim; j++) {
        q[h][t][j] = qkv[t][h * headDim + j]
        k[h][t][j] = qkv[t][nEmb + h * headDim + j]
        v[h][t][j] = qkv[t][2 * nEmb + h * headDim + j]
      }
    }
  }

  // 3) 缩放点积注意力 + 因果掩码 + softmax + 加权
  const attProbs = [], attOut = []
  for (let h = 0; h < nHead; h++) {
    attProbs[h] = new Array(T)
    attOut[h] = new Array(T)
    for (let t = 0; t < T; t++) {
      // 行 logits：attL[t][s] = (q[t]·k[s])·scale，s>t 置 -Inf
      const row = new Array(T)
      let mx = -Infinity
      for (let s = 0; s <= t; s++) {
        let acc = 0
        for (let j = 0; j < headDim; j++) acc += q[h][t][j] * k[h][s][j]
        row[s] = acc * scale
        if (row[s] > mx) mx = row[s]
      }
      // 行 softmax
      let sum = 0
      for (let s = 0; s <= t; s++) { row[s] = Math.exp(row[s] - mx); sum += row[s] }
      const prob = new Array(T)
      for (let s = 0; s <= t; s++) prob[s] = row[s] / sum
      for (let s = t + 1; s < T; s++) prob[s] = 0
      attProbs[h][t] = prob
      // out[t] = Σ_s prob[s]·v[s]
      attOut[h][t] = new Array(headDim)
      for (let j = 0; j < headDim; j++) {
        let acc = 0
        for (let s = 0; s <= t; s++) acc += prob[s] * v[h][s][j]
        attOut[h][t][j] = acc
      }
    }
  }

  // 4) 合并 head → merged[t][nEmb]
  const merged = new Array(T)
  for (let t = 0; t < T; t++) {
    merged[t] = new Array(nEmb)
    for (let h = 0; h < nHead; h++) {
      for (let j = 0; j < headDim; j++) merged[t][h * headDim + j] = attOut[h][t][j]
    }
  }

  // 5) 输出投影：y[t][c] = Σ_j merged[t][j]·W[j][c] + b[c]
  const y = new Array(T)
  for (let t = 0; t < T; t++) {
    y[t] = new Array(nEmb)
    for (let c = 0; c < nEmb; c++) {
      let acc = cProjB ? cProjB[0][c] : 0
      for (let j = 0; j < nEmb; j++) acc += merged[t][j] * cProjW[j][c]
      y[t][c] = acc
    }
  }

  return { y, cache: { x, q, k, v, attProbs, merged, nHead, headDim, scale, T, nEmb, cProjW, cProjB, cAttnW } }
}

/**
 * 因果自注意力反向。dy: [T][nEmb]。
 * 返回 { dx, dcAttnW, dcAttnB, dcProjW, dcProjB }
 */
export function attentionBackward(dy, cache) {
  const { x, q, k, v, attProbs, merged, nHead, headDim, scale, T, nEmb, cProjW, cProjB, cAttnW } = cache
  const dx = new Array(T)
  for (let t = 0; t < T; t++) dx[t] = new Array(nEmb).fill(0)
  const dcAttnW = zeros(nEmb, 3 * nEmb)
  const dcAttnB = zeros(1, 3 * nEmb)
  const dcProjW = zeros(nEmb, nEmb)
  const dcProjB = zeros(1, nEmb)

  // 1) 输出投影反向
  const dmerged = new Array(T)
  for (let t = 0; t < T; t++) {
    dmerged[t] = new Array(nEmb)
    for (let j = 0; j < nEmb; j++) {
      let acc = 0
      for (let c = 0; c < nEmb; c++) acc += dy[t][c] * cProjW[j][c]
      dmerged[t][j] = acc
    }
    for (let c = 0; c < nEmb; c++) {
      dcProjB[0][c] += dy[t][c]
      for (let j = 0; j < nEmb; j++) dcProjW[j][c] += merged[t][j] * dy[t][c]
    }
  }

  // 2) 拆回 head 维度
  const dOut = new Array(nHead)
  for (let h = 0; h < nHead; h++) {
    dOut[h] = new Array(T)
    for (let t = 0; t < T; t++) {
      dOut[h][t] = new Array(headDim)
      for (let j = 0; j < headDim; j++) dOut[h][t][j] = dmerged[t][h * headDim + j]
    }
  }

  // 3) 逐 head 反向 attention（矩阵形式推导见注释）
  const dqkv = new Array(T)
  for (let t = 0; t < T; t++) dqkv[t] = new Array(3 * nEmb).fill(0)

  for (let h = 0; h < nHead; h++) {
    // 3a) dv[s][j] = Σ_t attProbs[t][s]·dOut[t][j]   （t 是 query 行，s 是 key 列）
    //     dAttProb[t][s] = Σ_j dOut[t][j]·v[s][j]
    const dv = [], dAttProb = []
    for (let s = 0; s < T; s++) { dv.push(new Array(headDim).fill(0)); dAttProb.push(new Array(T).fill(0)) }
    for (let t = 0; t < T; t++) {
      for (let s = 0; s <= t; s++) {
        for (let j = 0; j < headDim; j++) {
          dAttProb[t][s] += dOut[h][t][j] * v[h][s][j]
          dv[s][j] += attProbs[h][t][s] * dOut[h][t][j]
        }
      }
    }
    // 3b) softmax 反向：dLogit[t][s] = p[t][s]·(dAttProb[t][s] - Σ_s' p[t][s']·dAttProb[t][s'])
    const dLogit = new Array(T)
    for (let t = 0; t < T; t++) {
      dLogit[t] = new Array(T).fill(0)
      let dot = 0
      for (let s = 0; s <= t; s++) dot += attProbs[h][t][s] * dAttProb[t][s]
      for (let s = 0; s <= t; s++) dLogit[t][s] = attProbs[h][t][s] * (dAttProb[t][s] - dot)
    }
    // 3c) 缩放反向 + 点积反向
    //     logit = (q·k)·scale → d(q·k) = dLogit·scale
    const dq = [], dk = []
    for (let t = 0; t < T; t++) dq.push(new Array(headDim).fill(0))
    for (let s = 0; s < T; s++) dk.push(new Array(headDim).fill(0))
    for (let t = 0; t < T; t++) {
      for (let s = 0; s <= t; s++) {
        const dl = dLogit[t][s] * scale
        for (let j = 0; j < headDim; j++) {
          dq[t][j] += dl * k[h][s][j]
          dk[s][j] += dl * q[h][t][j]
        }
      }
    }
    // 3d) 写回 dqkv（累加）
    for (let t = 0; t < T; t++) {
      for (let j = 0; j < headDim; j++) {
        dqkv[t][h * headDim + j] += dq[t][j]
        dqkv[t][nEmb + h * headDim + j] += dk[t][j]
        dqkv[t][2 * nEmb + h * headDim + j] += dv[t][j]
      }
    }
  }

  // 4) QKV 投影反向
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < 3 * nEmb; i++) {
      dcAttnB[0][i] += dqkv[t][i]
      for (let j = 0; j < nEmb; j++) {
        dcAttnW[j][i] += x[t][j] * dqkv[t][i]
        dx[t][j] += dqkv[t][i] * cAttnW[j][i]
      }
    }
  }

  return { dx, dcAttnW, dcAttnB, dcProjW, dcProjB }
}
