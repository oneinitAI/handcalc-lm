// ============================================================
// scripts/train-demo.js
// AI 学习本 —— 训练演示：小语料训练，验证 loss 下降 + 生成效果
// 运行：node scripts/train-demo.js
// ============================================================

import { createModel, paramCount } from '../src/model.js'
import { trainStep, createOptimizer } from '../src/train.js'
import { sample } from '../src/sample.js'

// ---- 语料（现代文，短小可背）----
const text = '月光洒在荷塘上，水面泛起银色的涟漪，风轻轻吹过，荷叶随风摇曳，远处传来几声蛙鸣。'

// ---- 按字切分，建字符表 ----
const chars = [...new Set(text.split(''))].sort()
const stoi = Object.fromEntries(chars.map((c, i) => [c, i]))
const itos = chars
const vocab = chars.length

const cfg = { vocab_size: vocab, block_size: 8, n_layer: 1, n_head: 1, n_embd: 8, bias: true }
const { params } = createModel(cfg, 42)
console.log(`语料 ${chars.length} 个字符，模型 ${paramCount(params)} 参数\n`)

// ---- 训练数据：滑动窗口 (X, Y) ----
const ids = text.split('').map((c) => stoi[c])
const seqLen = cfg.block_size
const X = [], Y = []
for (let i = 0; i < ids.length - seqLen; i++) {
  X.push(ids.slice(i, i + seqLen))
  Y.push(ids.slice(i + 1, i + seqLen + 1))
}

// ---- 训练 ----
const opt = createOptimizer(params, { type: 'adam', lr: 0.01 })
const steps = 2000
for (let s = 0; s < steps; s++) {
  const i = Math.floor(Math.random() * X.length)
  const { loss } = trainStep(params, X[i], Y[i], cfg, opt)
  if (s % 400 === 0) console.log(`step ${s}  loss=${loss.toFixed(4)}`)
}
const final = trainStep(params, X[0], Y[0], cfg, opt)
console.log(`step ${steps}  loss=${final.loss.toFixed(4)}\n`)

// ---- 生成验证 ----
const prompt = '月光洒在'
const promptIds = prompt.split('').map((c) => stoi[c])
const gen = sample(params, promptIds, 20, cfg, { temperature: 0.8 })
console.log('生成：' + gen.map((i) => itos[i]).join(''))
