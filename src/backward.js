// ============================================================
// src/backward.js
// 手算LM —— 反向传播主流程（残差连接正确累加）
// ============================================================

import { zeros, add } from './matrix.js'
import { layerNormBackward } from './layers.js'
import { mlpBackward } from './linear.js'
import { attentionBackward } from './attention.js'

/** 把一层梯度累加进对应参数 */
function accGrad(param, g) {
  for (let r = 0; r < g.length; r++) {
    for (let c = 0; c < g[r].length; c++) param.grad[r][c] += g[r][c]
  }
}

/**
 * 反向传播。需先 forward 拿 cache；结果累加到 params[*].grad。
 * dlogits: [T][vocab]（crossEntropy 的 dlogits）。
 */
export function backward(params, cache, dlogits, cfg) {
  const { T, nEmb, tokEmb, posEmb, blocks, lnf, idx, vocab } = cache
  const p = params

  // ---- lm_head 反向（与 wte 共享权重）----
  // logits[t][v] = Σ_j lnf.y[t][j]·wte[v][j]
  const dlnfX = new Array(T)
  const dwte = zeros(vocab, nEmb)
  for (let t = 0; t < T; t++) {
    dlnfX[t] = new Array(nEmb).fill(0)
    for (let v = 0; v < vocab; v++) {
      const d = dlogits[t][v]
      for (let j = 0; j < nEmb; j++) {
        dlnfX[t][j] += d * p.wte.value[v][j]
        dwte[v][j] += lnf.y[t][j] * d
      }
    }
  }

  // ---- final LayerNorm ----
  const lnfBack = layerNormBackward(dlnfX, lnf.cache)
  accGrad(p['ln_f.w'], lnfBack.dw)
  if (cfg.bias) accGrad(p['ln_f.b'], lnfBack.db)
  let dx = lnfBack.dx // d(xAfterMlp)（最后一层输出）

  // ---- 逆序 blocks（残差连接累加）----
  for (let l = cfg.n_layer - 1; l >= 0; l--) {
    const b = blocks[l]
    const prefix = `blocks.${l}.`

    // MLP 路径：dln2 = d(ln2.y)
    const mlpBack = mlpBackward(dx, b.mlp.cache, cfg, p, prefix + 'mlp.')
    accGrad(p[prefix + 'mlp.c_fc.w'], mlpBack.dW)
    if (cfg.bias) accGrad(p[prefix + 'mlp.c_fc.b'], mlpBack.db)
    accGrad(p[prefix + 'mlp.c_proj.w'], mlpBack.dW2)
    if (cfg.bias) accGrad(p[prefix + 'mlp.c_proj.b'], mlpBack.db2)
    const ln2Back = layerNormBackward(mlpBack.dx, b.ln2.cache)
    accGrad(p[prefix + 'ln2.w'], ln2Back.dw)
    if (cfg.bias) accGrad(p[prefix + 'ln2.b'], ln2Back.db)

    // xAfterAttn 的完整梯度 = 残差(dx) + MLP路径(ln2Back.dx)
    const dxAfterAttn = add(dx, ln2Back.dx)

    // attention 路径
    const attnBack = attentionBackward(dxAfterAttn, b.attn.cache)
    accGrad(p[prefix + 'attn.c_attn.w'], attnBack.dcAttnW)
    if (cfg.bias) accGrad(p[prefix + 'attn.c_attn.b'], attnBack.dcAttnB)
    accGrad(p[prefix + 'attn.c_proj.w'], attnBack.dcProjW)
    if (cfg.bias) accGrad(p[prefix + 'attn.c_proj.b'], attnBack.dcProjB)

    // LN1 路径
    const ln1Back = layerNormBackward(attnBack.dx, b.ln1.cache)
    accGrad(p[prefix + 'ln1.w'], ln1Back.dw)
    if (cfg.bias) accGrad(p[prefix + 'ln1.b'], ln1Back.db)

    // xIn 的完整梯度 = 残差(dxAfterAttn) + LN1路径(ln1Back.dx)
    dx = add(dxAfterAttn, ln1Back.dx)
  }

  // ---- embedding 反向 ----
  // token embedding 有两条梯度路径：
  //   (1) lm_head 共享权重 → dwte（所有 vocab 都要累加）
  //   (2) 输入查表路径 → dx（仅输入中出现过的 token）
  for (let v = 0; v < vocab; v++) {
    for (let j = 0; j < nEmb; j++) p.wte.grad[v][j] += dwte[v][j]
  }
  for (let t = 0; t < T; t++) {
    const id = idx[t]
    for (let j = 0; j < nEmb; j++) {
      p.wte.grad[id][j] += dx[t][j]
      p.wpe.grad[t][j] += dx[t][j]
    }
  }
}
