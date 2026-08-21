// ============================================================
// scripts/m4-demo.js
// 手算LM —— M4 验证：显微镜模式 + Attention 直播数据源
// 运行：node scripts/m4-demo.js
// ============================================================

import { createModel } from '../src/model.js'
import { trainStep, createOptimizer } from '../src/train.js'
import { CORPUS, buildVocab } from '../src/corpus.js'
import { runMicroscope } from '../src/microscope.js'
import { sampleWithAttn } from '../src/attn.js'

const text = CORPUS[0].text
const { stoi, itos, vocab } = buildVocab(text)
const cfg = { vocab_size: vocab, block_size: 12, n_layer: 1, n_head: 1, n_embd: 8, bias: true }
const { params } = createModel(cfg, 42)
const opt = createOptimizer(params, { type: 'adam', lr: 0.01 })

// 训练一点，让模型有点语感
const ids = text.split('')
for (let s = 0; s < 1200; s++) {
  const i = Math.floor(Math.random() * Math.max(1, ids.length - cfg.block_size - 1))
  const x = ids.slice(i, i + cfg.block_size).map((c) => stoi[c])
  const y = ids.slice(i + 1, i + cfg.block_size + 1).map((c) => stoi[c])
  trainStep(params, x, y, cfg, opt)
}

// ---- 显微镜 ----
const idx = '月光'.split('').map((c) => stoi[c])
const ms = runMicroscope(params, idx, cfg)
console.log('--- 显微镜 ---')
console.log('步骤:', ms.steps.map((s) => s.title).join(' | '))
console.log('预测下一个字:', itos[ms.predicted])
const topStep = ms.steps[5]
console.log('最高概率:', (Math.max(...topStep.probs) * 100).toFixed(1) + '%')

// ---- Attention 直播 ----
const { seq, attnSteps } = sampleWithAttn(params, idx, 8, cfg, { temperature: 0.8 })
console.log('\n--- Attention 直播 ---')
console.log('生成:', seq.map((i) => itos[i]).join(''))
console.log('attention 步数:', attnSteps.length)
console.log('第 1 步分布长度:', attnSteps[0].length, ' 和≈', attnSteps[0].reduce((a, b) => a + b, 0).toFixed(3))
console.log('末步分布长度:', attnSteps[attnSteps.length - 1].length)
