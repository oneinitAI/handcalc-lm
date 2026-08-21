// ============================================================
// src/sft.js
// 手算LM —— 监督微调（SFT）：让模型学会"回答问题"而非"接着写"
// 核心：换数据格式（带 <u>/<a> 角色标记），在预训练权重上继续真实训练。
// ============================================================

import { USER, ASSISTANT, END } from './corpus.js'

/** 内置示例问答对（通用，任何语料都能用） */
export const DEFAULT_QA = [
  { q: '你是谁', a: '我是手算LM，一个只有几百个参数的小模型。' },
  { q: '什么是语言模型', a: '语言模型就是一个会猜下一个字的机器。' },
  { q: '你会做什么', a: '我会学习你给我的文本，然后试着续写下去。' },
  { q: '你好', a: '你好！很高兴见到你。' },
  { q: '再见', a: '再见，期待下次再见。' },
]

/** 3 套内置问答（微调时可切换） */
export const QA_SETS = {
  general: {
    name: '通用对话',
    pairs: DEFAULT_QA,
  },
  about: {
    name: '关于手算LM',
    pairs: [
      { q: '你是谁', a: '我是手算LM，一个在浏览器里运行的微型语言模型。' },
      { q: '你有多少参数', a: '我只有几百到几万个参数，比大模型小了一亿倍。' },
      { q: '你会什么', a: '我会猜下一个字，也会试着回答你的问题。' },
      { q: '你有多聪明', a: '我很小，只会背训练数据里的句子，不会真正思考。' },
      { q: '你在哪里运行', a: '我住在你的浏览器里，没有服务器，全靠你的电脑计算。' },
    ],
  },
  fun: {
    name: '趣味问答',
    pairs: [
      { q: '你好', a: '你好呀！今天想聊点什么？' },
      { q: '你喜欢什么', a: '我喜欢学习新的文本，那是我的全部世界。' },
      { q: '给我讲个笑话', a: '机器学习学不会讲笑话，因为笑话需要懂人类。' },
      { q: '1加1等于几', a: '我猜是2，但我也可能猜成3，毕竟我只是个猜字的机器。' },
      { q: '晚安', a: '晚安！我会在梦里继续背你的语料。' },
    ],
  },
}

/**
 * 问答对 → SFT 训练序列（角色标记格式）。
 * 形如：<u>你是谁<a>我是手算LM...<e><u>...
 */
export function formatPairs(pairs) {
  return pairs.map(({ q, a }) => USER + q + ASSISTANT + a + END).join('')
}

/** 从格式化序列构建滑动窗口训练数据 [(X,Y)]，与预训练相同的 next-token 任务 */
export function buildSftData(formatted, blockSize) {
  const ids = formatted.split('')
  const X = []
  const Y = []
  for (let i = 0; i < ids.length - blockSize; i++) {
    X.push(ids.slice(i, i + blockSize))
    Y.push(ids.slice(i + 1, i + blockSize + 1))
  }
  return X.length ? { X, Y } : null
}

/** 构造问答 prompt 序列：<u>问题<a>（模型从此继续生成回答） */
export function qaPrompt(stoi, question) {
  return [USER, ...question.split(''), ASSISTANT].map((c) => stoi[c] ?? 0)
}

/**
 * 扩展 vocab 以覆盖问答对中的新字符（OOV 处理）。
 * 新字追加随机初始化的 embedding 行（旧 token 权重继承）——微调学到新字是合理行为。
 * 若传入 opt（优化器），同步扩展其状态（Adam m/v / SGD v）。
 * 返回新增字符数。model = { params, cfg, stoi, itos }。
 */
export function extendVocab(model, text, opt) {
  const { stoi, itos, cfg, params } = model
  let added = 0
  for (const c of new Set(text.split(''))) {
    if (stoi[c] === undefined) {
      const idx = itos.length
      stoi[c] = idx
      itos.push(c)
      const newRow = new Array(cfg.n_embd)
      for (let j = 0; j < cfg.n_embd; j++) newRow[j] = (Math.random() * 2 - 1) * 0.1
      params.wte.value.push(newRow)
      params.wte.grad.push(new Array(cfg.n_embd).fill(0))
      added++
    }
  }
  if (added) {
    cfg.vocab_size += added
    // 同步扩展优化器状态（wte 行数变了）
    if (opt && opt.state && opt.state.wte) {
      const st = opt.state.wte
      for (let i = 0; i < added; i++) {
        if (st.m) st.m.push(new Array(cfg.n_embd).fill(0))
        if (st.v) st.v.push(new Array(cfg.n_embd).fill(0))
      }
    }
  }
  return added
}
