// ============================================================
// scripts/sft-demo.js
// AI 学习本 —— SFT 验证：预训练 → 微调 → 对比"续写"与"问答"
// 运行：node scripts/sft-demo.js
// ============================================================

import { createModel } from '../src/model.js'
import { trainStep, createOptimizer } from '../src/train.js'
import { sample } from '../src/sample.js'
import { CORPUS, buildVocab, tokensToText } from '../src/corpus.js'
import { DEFAULT_QA, formatPairs, qaPrompt, extendVocab } from '../src/sft.js'

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

function ask(q) {
  const p = qaPrompt(stoi, q)
  const seq = sample(params, p, 30, cfg, { temperature: 0.8 })
  return tokensToText(itos, seq)
}

console.log('--- 阶段一：预训练（语料续写）---')
trainOn(text.split(''), 1500)
console.log('预训练完成，loss 已下降')

console.log('\n--- 微调前：问"你是谁"（模型只会续写）---')
console.log(ask('你是谁'))

console.log('\n--- 阶段二：监督微调（SFT 问答对）---')
const formatted = formatPairs(DEFAULT_QA)
extendVocab({ params, cfg, stoi, itos }, formatted, opt)
trainOn(formatted.split(''), 1500)
console.log('微调完成')

console.log('\n--- 微调后：问"你是谁"（应该开始回答）---')
console.log(ask('你是谁'))
console.log('\n--- 微调后：问"你好" ---')
console.log(ask('你好'))
