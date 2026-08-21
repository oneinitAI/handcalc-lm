// ============================================================
// scripts/dpo-demo.js
// AI 学习本 —— DPO 验证：让模型偏向用户偏好（偏好概率 五五开→偏向）
// 运行：node scripts/dpo-demo.js
// ============================================================

import { createModel } from '../src/model.js'
import { trainStep, createOptimizer } from '../src/train.js'
import { CORPUS, buildVocab } from '../src/corpus.js'
import { DEFAULT_QA, formatPairs, qaPrompt, extendVocab } from '../src/sft.js'
import { dpoTrainStep, makeRefModel, answerLogProb } from '../src/dpo.js'

// ---- 预训练 + SFT（复用 sft-demo 流程）----
const text = CORPUS[0].text
const { stoi, itos, vocab } = buildVocab(text)
const cfg = { vocab_size: vocab, block_size: 12, n_layer: 1, n_head: 1, n_embd: 8, bias: true }
const { params } = createModel(cfg, 42)
const opt = createOptimizer(params, { type: 'adam', lr: 0.01 })

function trainOn(seq, steps) {
  const L = seq.length
  for (let s = 0; s < steps; s++) {
    const i = Math.floor(Math.random() * Math.max(1, L - cfg.block_size - 1))
    const x = seq.slice(i, i + cfg.block_size).map((c) => stoi[c])
    const y = seq.slice(i + 1, i + cfg.block_size + 1).map((c) => stoi[c])
    trainStep(params, x, y, cfg, opt)
  }
}
trainOn(text.split(''), 1500)
const formatted = formatPairs(DEFAULT_QA)
extendVocab({ params, cfg, stoi, itos }, formatted, opt)
trainOn(formatted.split(''), 1500)
console.log('预训练 + SFT 完成，模型已会问答格式')

// ---- 构造偏好对：问题"你好"，偏好"很高兴见到你"，非偏好"再见期待下次再见"（都接近训练数据，更公平）----
const x = qaPrompt(stoi, '你好')
const yw = '很高兴见到你'.split('').map((c) => stoi[c])
const yl = '再见期待下次再见'.split('').map((c) => stoi[c])

const refParams = makeRefModel(params) // 冻结参考模型（SFT 后）
const beta = 0.5

function prefProb() {
  const pw = answerLogProb(params, [...x, ...yw], x.length, cfg)
  const pl = answerLogProb(params, [...x, ...yl], x.length, cfg)
  return 1 / (1 + Math.exp(-beta * (pw - pl)))
}

console.log(`DPO 前：P(偏好回答) = ${prefProb().toFixed(4)}`)

let lastLoss = 0
for (let s = 0; s < 500; s++) {
  lastLoss = dpoTrainStep(params, refParams, cfg, opt, x, yw, yl, beta).loss
  if ((s + 1) % 100 === 0) {
    console.log(`step ${s + 1}  loss=${lastLoss.toFixed(4)}  P=${prefProb().toFixed(4)}`)
  }
}

console.log(`DPO 500 步后 loss=${lastLoss.toFixed(4)}`)
console.log(`DPO 后：P(偏好回答) = ${prefProb().toFixed(4)}`)
console.log(prefProb() > 0.9
  ? '\n✅ DPO 生效：模型已明显偏向你偏好的回答'
  : '\n⚠️ 偏好概率未达到 0.9（可调 β/步数再试）')
