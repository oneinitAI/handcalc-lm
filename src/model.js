// ============================================================
// src/model.js
// 手算LM —— 微型 Transformer 主流程（纯 JS，手写）
// 结构参考 nanoGPT/model.py；教学简化 batch=1。
// 参数存储：params = { name: { value, grad } }，value/grad 均为二维数组。
// ============================================================

import { zeros, ones, randn, scale, crossEntropy } from './matrix.js'
import { layerNormForward } from './layers.js'
import { mlpForward } from './linear.js'
import { attentionForward } from './attention.js'

/** 手算LM 极简配置（~300 参数级） */
export const defaultCfg = {
  vocab_size: 32,
  block_size: 8,
  n_layer: 1,
  n_head: 1,
  n_embd: 4,
  bias: true,
  dropout: 0,
}

/** 参数总数（lm_head 与 wte 共享，不重复计数） */
export function paramCount(params) {
  let n = 0
  for (const key in params) {
    const v = params[key].value
    n += v.length * v[0].length
  }
  return n
}

/** 创建模型参数。lm_head 与 wte 共享（weight tying）。 */
export function createModel(cfg = defaultCfg, seed = 1337) {
  const params = {}
  let s = seed
  const std = 0.1 // 极小模型用稍大 std，让梯度信号可流动

  const add = (name, rows, cols, fill) => {
    if (fill === 'ones') {
      params[name] = { value: ones(rows, cols), grad: zeros(rows, cols) }
    } else if (fill === 'zeros') {
      params[name] = { value: zeros(rows, cols), grad: zeros(rows, cols) }
    } else {
      const r = randn(rows, cols, s)
      s = r.nextSeed
      params[name] = { value: scale(r.value, std), grad: zeros(rows, cols) }
    }
  }

  add('wte', cfg.vocab_size, cfg.n_embd) // token embedding（= lm_head 共享）
  add('wpe', cfg.block_size, cfg.n_embd) // position embedding

  for (let l = 0; l < cfg.n_layer; l++) {
    const p = `blocks.${l}.`
    add(p + 'ln1.w', 1, cfg.n_embd, 'ones')
    if (cfg.bias) add(p + 'ln1.b', 1, cfg.n_embd, 'zeros')
    add(p + 'attn.c_attn.w', cfg.n_embd, 3 * cfg.n_embd)
    if (cfg.bias) add(p + 'attn.c_attn.b', 1, 3 * cfg.n_embd, 'zeros')
    add(p + 'attn.c_proj.w', cfg.n_embd, cfg.n_embd)
    if (cfg.bias) add(p + 'attn.c_proj.b', 1, cfg.n_embd, 'zeros')
    add(p + 'ln2.w', 1, cfg.n_embd, 'ones')
    if (cfg.bias) add(p + 'ln2.b', 1, cfg.n_embd, 'zeros')
    add(p + 'mlp.c_fc.w', cfg.n_embd, 4 * cfg.n_embd)
    if (cfg.bias) add(p + 'mlp.c_fc.b', 1, 4 * cfg.n_embd, 'zeros')
    add(p + 'mlp.c_proj.w', 4 * cfg.n_embd, cfg.n_embd)
    if (cfg.bias) add(p + 'mlp.c_proj.b', 1, cfg.n_embd, 'zeros')
  }

  add('ln_f.w', 1, cfg.n_embd, 'ones')
  if (cfg.bias) add('ln_f.b', 1, cfg.n_embd, 'zeros')

  return { params, seed: s, cfg }
}

/** 清零所有梯度 */
export function zeroGrad(params) {
  for (const key in params) {
    const g = params[key].grad
    for (const row of g) row.fill(0)
  }
}

/**
 * 前向传播。idx: [T]（token ID 数组）；targets: [T]（可为 null）。
 * 返回 { logits, loss, probs, cache }。
 */
export function forward(params, idx, targets, cfg = defaultCfg) {
  const T = idx.length
  const nEmb = cfg.n_embd
  const vocab = cfg.vocab_size

  // 1) token + position embedding
  const tokEmb = new Array(T)
  const posEmb = new Array(T)
  for (let t = 0; t < T; t++) {
    tokEmb[t] = params.wte.value[idx[t]].slice()
    posEmb[t] = params.wpe.value[t].slice()
  }
  let x = new Array(T)
  for (let t = 0; t < T; t++) {
    x[t] = new Array(nEmb)
    for (let j = 0; j < nEmb; j++) x[t][j] = tokEmb[t][j] + posEmb[t][j]
  }

  // 2) blocks
  const blocks = []
  for (let l = 0; l < cfg.n_layer; l++) {
    const prefix = `blocks.${l}.`
    const p = params
    const xIn = x

    const ln1 = layerNormForward(xIn, p[prefix + 'ln1.w'].value, cfg.bias ? p[prefix + 'ln1.b'].value : zeros(1, nEmb))
    const attn = attentionForward(ln1.y,
      p[prefix + 'attn.c_attn.w'].value, cfg.bias ? p[prefix + 'attn.c_attn.b'].value : null,
      p[prefix + 'attn.c_proj.w'].value, cfg.bias ? p[prefix + 'attn.c_proj.b'].value : null, cfg)
    const xAfterAttn = addTensors(xIn, attn.y, nEmb)

    const ln2 = layerNormForward(xAfterAttn, p[prefix + 'ln2.w'].value, cfg.bias ? p[prefix + 'ln2.b'].value : zeros(1, nEmb))
    const mlp = mlpForward(ln2.y, cfg, p, prefix)
    const xAfterMlp = addTensors(xAfterAttn, mlp.y, nEmb)

    blocks.push({ xIn, ln1, attn, xAfterAttn, ln2, mlp, xAfterMlp })
    x = xAfterMlp
  }

  // 3) final LayerNorm
  const lnf = layerNormForward(x, params['ln_f.w'].value, cfg.bias ? params['ln_f.b'].value : zeros(1, nEmb))

  // 4) lm_head（与 wte 共享）：logits[t][v] = Σ_j x[t][j]·wte[v][j]
  const logits = new Array(T)
  for (let t = 0; t < T; t++) {
    logits[t] = new Array(vocab)
    for (let v = 0; v < vocab; v++) {
      let acc = 0
      for (let j = 0; j < nEmb; j++) acc += lnf.y[t][j] * params.wte.value[v][j]
      logits[t][v] = acc
    }
  }

  let loss = null, probs = null
  if (targets) {
    const ce = crossEntropy(logits, targets)
    loss = ce.loss
    probs = ce.probs
  }

  return { logits, loss, probs, cache: { idx, T, nEmb, tokEmb, posEmb, blocks, lnf, vocab } }
}

/** 逐元素相加（两个 [T][d]） */
function addTensors(a, b, d) {
  const T = a.length
  const out = new Array(T)
  for (let t = 0; t < T; t++) {
    out[t] = new Array(d)
    for (let j = 0; j < d; j++) out[t][j] = a[t][j] + b[t][j]
  }
  return out
}
