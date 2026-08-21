// ============================================================
// scripts/finetune-check.js
// AI 学习本 —— 验证 SFT 防遗忘修复：微调后既能续写语料、又能回答
// 运行：node scripts/finetune-check.js
// ============================================================

import { createModel } from '../src/model.js'
import { trainStep, createOptimizer } from '../src/train.js'
import { sample } from '../src/sample.js'
import { CORPUS, buildVocab, tokensToText } from '../src/corpus.js'
import { DEFAULT_QA, formatPairs, qaPrompt, extendVocab } from '../src/sft.js'

const text = CORPUS[0].text
const { stoi, itos, vocab } = buildVocab(text)
const cfg = { vocab_size: vocab, block_size: 16, n_layer: 1, n_head: 1, n_embd: 8, bias: true }
const { params } = createModel(cfg, 42)

const corpusIds = text.split('')
let opt = createOptimizer(params, { type: 'adam', lr: 0.01 })

function gen(prompt) {
  const p = prompt.split('').map((c) => stoi[c] ?? 0)
  const seq = sample(params, p, 20, cfg, { temperature: 0.8 })
  return tokensToText(itos, seq)
}
function ask(q) {
  const x = qaPrompt(stoi, q)
  const seq = sample(params, x, 24, cfg, { temperature: 0.8 })
  return tokensToText(itos, seq)
}

// ---- 预训练 ----
for (let s = 0; s < 1500; s++) {
  const i = Math.floor(Math.random() * Math.max(1, corpusIds.length - cfg.block_size - 1))
  const x = corpusIds.slice(i, i + cfg.block_size).map((c) => stoi[c])
  const y = corpusIds.slice(i + 1, i + cfg.block_size + 1).map((c) => stoi[c])
  trainStep(params, x, y, cfg, opt)
}
console.log('预训练后  续写"月光" →', gen('月光'))
console.log('预训练后  问"你好" →', ask('你好'), '（应不会回答）')

// ---- SFT：50/50 混合语料+问答（模拟浏览器修复后的行为），学习率调低 ----
const sftSeq = formatPairs(DEFAULT_QA).split('')
extendVocab({ params, cfg, stoi, itos }, formatPairs(DEFAULT_QA), opt)
opt = createOptimizer(params, { type: 'adam', lr: 0.005 })
for (let s = 0; s < 2500; s++) {
  const useSft = Math.random() < 0.5
  const seq = useSft ? sftSeq : corpusIds
  const L = seq.length
  const i = Math.floor(Math.random() * Math.max(1, L - cfg.block_size - 1))
  const x = seq.slice(i, i + cfg.block_size).map((c) => stoi[c])
  const y = seq.slice(i + 1, i + cfg.block_size + 1).map((c) => stoi[c])
  trainStep(params, x, y, cfg, opt)
}
console.log('混合微调后 续写"月光" →', gen('月光'), '（应保留语料风格）')
console.log('混合微调后 问"你好" →', ask('你好'), '（应会回答）')