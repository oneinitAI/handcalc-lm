// ============================================================
// src/main.js
// 手算LM —— M1：预训练实验台（数据区 + 训练控制 + 基础可视化 + 流式生成）
// ============================================================

import './style.css'
import { CORPUS, buildVocab, tokensToText } from './corpus.js'
import { createModel, paramCount } from './model.js'
import { trainStep, createOptimizer } from './train.js'
import { sample } from './sample.js'
import { createLossChart, createHeatmap, createAttnHeatmap } from './ui.js'
import { sampleWithAttn } from './attn.js'
import { initMicroscopeUI } from './microscope-ui.js'
import { NOTES } from './notes.js'
import { DEFAULT_QA, formatPairs, buildSftData, qaPrompt, extendVocab } from './sft.js'
import { dpoTrainStep, makeRefModel } from './dpo.js'

const SIZES = {
  tiny:   { n_layer: 1, n_head: 1, n_embd: 8,  block_size: 8 },
  small:  { n_layer: 2, n_head: 2, n_embd: 16, block_size: 12 },
  medium: { n_layer: 2, n_head: 4, n_embd: 32, block_size: 16 },
}

// ---------- 状态 ----------
const state = {
  model: null,
  opt: null,
  losses: [],
  training: false,
  rafId: null,
  mode: 'cont', // 'cont' 续写 | 'qa' 问答
  snap: null,   // 权重快照（微调前，用于对比）
  sftData: null,
  prefs: [],    // DPO 偏好对 [{x, yw, yl}]
  pair: null,   // 当前待点选的回答对 {a, b}（token 数组）
  refParams: null, // DPO 参考模型（冻结）
  dpoOpt: null,  // DPO 专用优化器
  stage: 'pre',  // 'pre' 预训练 | 'sft' 微调 | 'dpo' 对齐
  blind: null,   // 盲测数据 { a, b, correctIsA }
  genCount: 0,   // 生成次数（Karpathy 彩蛋）
  initLoss: null,     // 初始 loss（前 10 步平均），进度条基准
  stopNotified: false, // 自动停止提示标志
}

// ---------- DOM ----------
const app = document.getElementById('app')
app.innerHTML = `
  <main class="stage">
    <header class="masthead">
      <h1 class="title">手算<span class="hl">LM</span></h1>
      <p class="sub">给你权重，亲手算出它的下一句话</p>
    </header>

    <div class="stage-bar">
      <span id="stagePre" class="stage-dot on">壹 预训练</span>
      <span class="stage-arrow">→</span>
      <span id="stageSft" class="stage-dot">贰 微调</span>
      <span class="stage-arrow">→</span>
      <span id="stageDpo" class="stage-dot">叁 对齐</span>
      <button id="flipBtn" class="btn ghost flip-btn">翻面 · 进阶模式</button>
    </div>

    <section class="card" id="dataCard">
      <h2>壹 · 语料</h2>
      <div class="corpus-pick">
        ${CORPUS.map((c) => `<button class="chip" data-id="${c.id}">${c.title}</button>`).join('')}
        <span class="hint">或直接粘贴你的文本 ↓</span>
      </div>
      <textarea id="corpus" rows="4"></textarea>
      <p class="muted" id="corpusInfo"></p>
    </section>

    <section class="card" id="modelCard">
      <h2>贰 · 模型与训练</h2>
      <div class="row">
        <label>档位
          <select id="size">
            <option value="tiny">微</option>
            <option value="small">小</option>
            <option value="medium" selected>中</option>
          </select>
        </label>
        <label>学习率 <input id="lr" value="0.01" size="6"></label>
        <label>速度 <input id="speed" type="range" min="20" max="1000" value="200"></label>
        <button id="trainBtn" class="btn">开始训练</button>
        <button id="stepBtn" class="btn ghost">单步</button>
        <button id="resetBtn" class="btn ghost">重建</button>
        <button id="snapSaveBtn" class="btn ghost">存快照</button>
        <button id="snapLoadBtn" class="btn ghost">读快照</button>
      </div>
      <p class="muted" id="modelInfo"></p>
      <div class="train-progress">
        <div class="progress-track"><div id="progressFill" class="progress-fill"></div></div>
        <span id="progressText" class="progress-text">—</span>
      </div>
      <div class="viz-row">
        <div class="viz">
          <div class="viz-title">loss 曲线 <span class="tag" id="lossTag">—</span></div>
          <canvas id="lossChart" class="canvas"></canvas>
        </div>
        <div class="viz">
          <div class="viz-title">权重热力图 <span class="tag">wte（词向量表）</span></div>
          <canvas id="heatmap" class="canvas"></canvas>
        </div>
      </div>
    </section>

    <section class="card" id="sftCard">
      <h2>叁 · 微调（SFT）</h2>
      <p class="muted">喂给模型问答对，让它从"接着写"学会"回答问题"——在预训练权重上继续真实训练。</p>
      <div class="corpus-pick">
        <button id="loadQaBtn" class="chip">载入示例问答</button>
        <span class="hint">每行一条「问题 / 回答」，斜杠分隔</span>
      </div>
      <textarea id="qaList" rows="4"></textarea>
      <div class="row">
        <button id="sftBtn" class="btn">开始微调</button>
        <button id="snapBtn" class="btn ghost">记快照（微调前）</button>
        <button id="cmpBtn" class="btn ghost" disabled>切到：微调前</button>
      </div>
      <p class="muted" id="sftInfo"></p>
    </section>

    <section class="card" id="genCard">
      <h2>肆 · 生成</h2>
      <div class="corpus-pick">
        <button id="modeCont" class="chip on">续写模式</button>
        <button id="modeQa" class="chip">问答模式</button>
      </div>
      <div class="row">
        <input id="prompt" value="月光" size="14">
        <button id="genBtn" class="btn" disabled>生成</button>
        <label>温度 <input id="temp" value="0.8" size="4"></label>
        <label>长度 <input id="len" value="32" size="4"></label>
      </div>
      <details class="advanced">
        <summary>进阶采样（top-k / top-p）</summary>
        <div class="row">
          <label>top-k <input id="topk" value="0" size="4"></label>
          <span class="hint">0=关 · 只保留概率最高的前 k 个</span>
          <label>top-p <input id="topp" value="1" size="4"></label>
          <span class="hint">1=关 · 保留累计概率达 p 的候选</span>
        </div>
      </details>
      <div id="genOut" class="gen">（先训练，再让它续写或回答）</div>
      <div class="muted" id="perf"></div>
      <div class="viz">
        <div class="viz-title">Attention 直播 <span class="tag">模型在"看"哪些字</span></div>
        <canvas id="attnHeatmap" class="canvas"></canvas>
      </div>
    </section>

    <section class="card" id="dpoCard">
      <h2>伍 · 偏好对齐（DPO）</h2>
      <p class="muted">让模型生成两个回答，你告诉它哪个更好——它会学会偏向你的偏好。这就是 DPO（2023 年论文算法），也是 OpenAI 标注员做的真实工作。</p>
      <div class="row">
        <input id="dpoQ" value="你好" size="14">
        <button id="genPairBtn" class="btn">生成两个回答</button>
      </div>
      <div class="pair" id="pairBox" hidden>
        <div class="answer">
          <div id="ansA" class="ans-text">（回答 A）</div>
          <button id="pickABtn" class="btn ghost">这个更好</button>
        </div>
        <div class="answer">
          <div id="ansB" class="ans-text">（回答 B）</div>
          <button id="pickBBtn" class="btn ghost">这个更好</button>
        </div>
      </div>
      <p class="muted" id="prefInfo">已收集 0 对偏好</p>
      <div class="row">
        <label>β <input id="dpoBeta" value="0.5" size="4"></label>
        <label>步数 <input id="dpoSteps" value="300" size="5"></label>
        <button id="dpoBtn" class="btn" disabled>开始 DPO 训练</button>
        <button id="dpoResetBtn" class="btn ghost">清空偏好</button>
      </div>
      <p class="muted" id="dpoInfo"></p>
      <div id="blindBox" hidden></div>
    </section>

    <div id="microscopeRoot"></div>
  </main>
`

const $ = (id) => document.getElementById(id)
const lossChart = createLossChart($('lossChart'))
const heatmap = createHeatmap($('heatmap'))
$('heatmap').addEventListener('mousemove', (e) => { heatmap.onHover(e); heatmap.draw() })
$('heatmap').addEventListener('mouseleave', () => { heatmap.draw() })

const attnHeatmap = createAttnHeatmap($('attnHeatmap'), 16)
$('attnHeatmap').addEventListener('mousemove', (e) => { attnHeatmap.onHover(e); attnHeatmap.draw() })
$('attnHeatmap').addEventListener('mouseleave', () => attnHeatmap.draw())

// 缩放修复：窗口/布局变化时重绘所有 canvas（避免画布拉伸错位）
let resizeTimer = null
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer)
  resizeTimer = setTimeout(() => {
    lossChart.draw()
    heatmap.draw()
    attnHeatmap.draw()
  }, 100)
})

// ---------- 模型构建 ----------
function buildModel() {
  const text = $('corpus').value.trim()
  if (!text) { alert('请输入或选择语料'); return false }
  const { chars, stoi, itos, vocab } = buildVocab(text)
  const size = SIZES[$('size').value]
  const cfg = { vocab_size: vocab, bias: true, ...size }
  const { params } = createModel(cfg, 42)
  state.model = { params, cfg, stoi, itos, chars }
  state.opt = createOptimizer(params, { type: 'adam', lr: parseFloat($('lr').value) })
  state.ids = text.split('')
  state.losses = []
  state.initLoss = null
  state.stopNotified = false
  $('progressFill').style.width = '0%'
  $('progressText').textContent = '—'
  state.sftData = null
  state.snap = null
  state.showingBefore = false
  $('cmpBtn').disabled = true
  $('cmpBtn').textContent = '切到：微调前'
  lossChart.clear()
  heatmap.set(params.wte.value, 'wte')
  heatmap.draw()
  $('modelInfo').textContent = `${chars.length} token（含角色标记）· ${paramCount(params)} 参数 · ${cfg.n_layer}层${cfg.n_head}头${cfg.n_embd}维`
  $('genBtn').disabled = false
  return true
}

// ---------- 训练 ----------
function runSteps(n) {
  const { model, opt } = state
  const ids = state.ids // 预训练=语料字符，微调=带标记的问答序列
  const L = ids.length
  for (let k = 0; k < n; k++) {
    const i = Math.floor(Math.random() * Math.max(1, L - model.cfg.block_size - 1))
    const x = ids.slice(i, i + model.cfg.block_size).map((c) => model.stoi[c])
    const y = ids.slice(i + 1, i + model.cfg.block_size + 1).map((c) => model.stoi[c])
    const { loss } = trainStep(model.params, x, y, model.cfg, opt)
    state.losses.push(loss)
  }
  // 初始 loss 基准（前 10 步平均）
  if (!state.initLoss && state.losses.length >= 10) {
    let s = 0
    for (let i = 0; i < 10; i++) s += state.losses[i]
    state.initLoss = s / 10
  }
  lossChart.push(state.losses[state.losses.length - 1])
  if (state.losses.length % 50 === 0) heatmap.draw()
  updateProgress()
  checkStop()
}

/** 进度条：loss 从初始降到 25% 视为"学到位"（0~100%） */
function updateProgress() {
  const cur = state.losses[state.losses.length - 1]
  if (!state.initLoss || state.losses.length < 10) { $('progressText').textContent = '—'; return }
  const target = state.initLoss * 0.25
  const p = Math.max(0, Math.min(1, (state.initLoss - cur) / (state.initLoss - target)))
  $('progressFill').style.width = (p * 100).toFixed(0) + '%'
  $('progressText').textContent = p >= 1 ? '已学到位' : `已学 ${(p * 100).toFixed(0)}%`
}

/** 自动停止检测：loss 到 25% 或停滞 300 步 */
function checkStop() {
  if (!state.training || state.stopNotified || !state.initLoss) return
  const L = state.losses.length
  const cur = state.losses[L - 1]
  if (cur <= state.initLoss * 0.25) {
    state.stopNotified = true
    stopTraining()
    $('modelInfo').textContent = '✅ 模型已基本学会语料规律（loss 降到初始 25%），可以验收了。也可以继续训练微调。'
    return
  }
  if (L > 400 && L % 100 === 0) {
    const before = state.losses[L - 300]
    if (before - cur < state.initLoss * 0.01) {
      state.stopNotified = true
      stopTraining()
      $('modelInfo').textContent = '⏸ loss 已停滞（300 步无明显下降）——可能学到位了，或学习率不合适。可继续或调整参数。'
    }
  }
}

function tick() {
  if (!state.training) return
  const speed = parseInt($('speed').value) || 200
  runSteps(speed)
  lossChart.draw()
  state.rafId = requestAnimationFrame(tick)
}

function stopTraining() {
  state.training = false
  if (state.rafId) cancelAnimationFrame(state.rafId)
  $('trainBtn').textContent = '继续训练'
  lossChart.draw()
  heatmap.draw()
}

function startTraining() {
  if (!state.model) buildModel()
  if (!state.model) return
  state.training = true
  $('trainBtn').textContent = '暂停'
  tick()
}

$('trainBtn').addEventListener('click', () => {
  if (state.training) stopTraining()
  else startTraining()
})
$('stepBtn').addEventListener('click', () => {
  if (!state.model) buildModel()
  if (!state.model) return
  runSteps(1)
  lossChart.draw()
  heatmap.draw()
})
$('resetBtn').addEventListener('click', () => {
  stopTraining()
  buildModel()
  lossChart.draw()
})

// ---------- 模型快照（localStorage 保存/加载）----------
function saveSnapshot() {
  if (!state.model) { alert('先构建模型'); return }
  const data = {
    params: JSON.parse(JSON.stringify(state.model.params.value)),
    cfg: { ...state.model.cfg },
    itos: state.model.itos.slice(),
    stoi: state.model.stoi,
    ts: Date.now(),
  }
  try {
    const snaps = JSON.parse(localStorage.getItem('handcalc:snaps') || '[]')
    snaps.push(data)
    const kept = snaps.slice(-3) // 保留最近 3 个
    localStorage.setItem('handcalc:snaps', JSON.stringify(kept))
    $('modelInfo').textContent = `✅ 快照已存（共 ${kept.length} 个，浏览器本地）`
  } catch (e) {
    $('modelInfo').textContent = '⚠️ 模型太大，快照存不下（localStorage 容量限制）'
  }
}

function loadSnapshot() {
  let snaps = []
  try { snaps = JSON.parse(localStorage.getItem('handcalc:snaps') || '[]') } catch (e) { snaps = [] }
  if (!snaps.length) { alert('还没有快照。先训练，再点「存快照」'); return }
  const data = snaps[snaps.length - 1]
  if (!state.model) buildModel()
  const m = state.model
  for (const key in m.params.value) m.params.value[key] = data.params[key].map((row) => row.slice())
  m.cfg.vocab_size = data.cfg.vocab_size
  m.itos = data.itos.slice()
  m.stoi = { ...data.stoi }
  state.opt = createOptimizer(m.params, { type: 'adam', lr: parseFloat($('lr').value) }) // 重建优化器
  $('modelInfo').textContent = `📂 已读快照：${m.itos.length} token · ${paramCount(m.params)} 参数`
  heatmap.set(m.params.wte.value, 'wte')
  heatmap.draw()
  $('genBtn').disabled = false
}
$('snapSaveBtn').addEventListener('click', saveSnapshot)
$('snapLoadBtn').addEventListener('click', loadSnapshot)

// ---------- SFT 微调 ----------
function deepCopy(v) { return JSON.parse(JSON.stringify(v)) }

function parseQA() {
  const lines = $('qaList').value.split('\n').map((l) => l.trim()).filter(Boolean)
  const pairs = []
  for (const line of lines) {
    const i = line.indexOf('/')
    if (i > 0) pairs.push({ q: line.slice(0, i).trim(), a: line.slice(i + 1).trim() })
  }
  return pairs
}

function buildSft() {
  const pairs = parseQA()
  if (!pairs.length) { alert('请输入问答对（每行「问题 / 回答」，斜杠分隔）'); return false }
  const formatted = formatPairs(pairs)
  const added = extendVocab(state.model, formatted, state.opt) // 问答对新字符扩展 vocab + 优化器
  state.ids = formatted.split('')
  state.sftData = { pairs }
  $('modelInfo').textContent = `${state.model.itos.length} token（含角色标记）· 微调新增 ${added} 字符`
  $('sftInfo').textContent = `${pairs.length} 条问答对 · 训练序列 ${state.ids.length} token`
  return true
}

function snapshotWeights() {
  state.snap = deepCopy(state.model.params.value)
  state.showingBefore = false
  $('cmpBtn').disabled = false
  $('cmpBtn').textContent = '切到：微调前'
  $('sftInfo').textContent = '已记录微调前权重快照'
}

function toggleCompare() {
  if (!state.snap) return
  const cur = state.model.params.value
  const tmp = deepCopy(cur)
  for (const key in cur) cur[key] = deepCopy(state.snap[key])
  state.snap = tmp
  state.showingBefore = !state.showingBefore
  $('cmpBtn').textContent = state.showingBefore ? '切到：微调后' : '切到：微调前'
  heatmap.draw()
}

function setMode(m) {
  state.mode = m
  $('modeCont').classList.toggle('on', m === 'cont')
  $('modeQa').classList.toggle('on', m === 'qa')
  $('prompt').value = m === 'qa' ? '你是谁' : '月光'
  $('genOut').textContent = m === 'qa' ? '（问答模式：输入问题，模型试着回答）' : '（先训练，再让它续写）'
}

/** 阶段感知：随训练阶段解锁功能 */
function updateStage() {
  const s = state.stage
  $('stagePre').classList.toggle('on', true)
  $('stageSft').classList.toggle('on', s === 'sft' || s === 'dpo')
  $('stageDpo').classList.toggle('on', s === 'dpo')
  // 问答模式需完成 SFT 后才解锁
  $('modeQa').disabled = s === 'pre'
  if (s === 'pre' && state.mode === 'qa') setMode('cont')
  // 盲测需完成 DPO 后才解锁
  if (s === 'dpo') initBlind()
}

$('loadQaBtn').addEventListener('click', () => {
  $('qaList').value = DEFAULT_QA.map((p) => `${p.q} / ${p.a}`).join('\n')
})
$('sftBtn').addEventListener('click', () => {
  if (!state.model) buildModel()
  if (!state.model) return
  if (!buildSft()) return
  state.stage = 'sft'
  updateStage()
  state.losses = []
  state.initLoss = null
  state.stopNotified = false
  $('progressFill').style.width = '0%'
  $('progressText').textContent = '—'
  lossChart.clear()
  startTraining()
})
$('snapBtn').addEventListener('click', () => { if (state.model) snapshotWeights() })
$('cmpBtn').addEventListener('click', toggleCompare)
$('modeCont').addEventListener('click', () => setMode('cont'))
$('modeQa').addEventListener('click', () => setMode('qa'))

// ---------- DPO 偏好对齐 ----------
function genAnswer(q, temp) {
  const p = qaPrompt(state.model.stoi, q)
  const seq = sample(state.model.params, p, 24, state.model.cfg, { temperature: temp })
  let end = seq.length
  for (let i = p.length; i < seq.length; i++) {
    if (state.model.itos[seq[i]] === '\u0003') { end = i; break } // <e> 回答结束
  }
  return seq.slice(p.length, end)
}

function showPair(a, b) {
  state.pair = { a, b }
  $('pairBox').hidden = false
  $('ansA').textContent = tokensToText(state.model.itos, a) || '（空回答）'
  $('ansB').textContent = tokensToText(state.model.itos, b) || '（空回答）'
}

function pick(prefIsA) {
  const { a, b } = state.pair
  const x = qaPrompt(state.model.stoi, $('dpoQ').value.trim() || '你好')
  state.prefs.push(prefIsA ? { x, yw: a, yl: b } : { x, yw: b, yl: a })
  $('prefInfo').textContent = `已收集 ${state.prefs.length} 对偏好`
  $('dpoBtn').disabled = false
  $('dpoInfo').textContent = `已记偏好：你选了「${tokensToText(state.model.itos, prefIsA ? a : b)}」`
}

$('genPairBtn').addEventListener('click', () => {
  if (!state.model) { alert('请先构建模型并训练'); return }
  const q = $('dpoQ').value.trim() || '你好'
  showPair(genAnswer(q, 0.8), genAnswer(q, 1.2))
})
$('pickABtn').addEventListener('click', () => pick(true))
$('pickBBtn').addEventListener('click', () => pick(false))

$('dpoBtn').addEventListener('click', () => {
  if (!state.prefs.length) return
  if (!state.refParams) state.refParams = makeRefModel(state.model.params) // 冻结当前为参考模型
  if (!state.dpoOpt) state.dpoOpt = createOptimizer(state.model.params, { type: 'adam', lr: 0.0005 })
  const beta = parseFloat($('dpoBeta').value) || 0.5
  const steps = parseInt($('dpoSteps').value) || 300
  const total = steps * state.prefs.length
  let done = 0
  let lastLoss = 0
  const loop = () => {
    if (done >= total) {
      state.stage = 'dpo'
      updateStage()
      $('dpoInfo').textContent = `DPO 完成（${steps} 步 × ${state.prefs.length} 对）· 模型已偏向你的偏好`
      heatmap.draw()
      // 彩蛋：手算者印（三阶段全部完成）
      if (!document.querySelector('.stamp')) {
        const stamp = document.createElement('div')
        stamp.className = 'stamp'
        stamp.textContent = '手算者 · 完成'
        document.querySelector('.masthead').after(stamp)
      }
      return
    }
    const k = 20
    for (let i = 0; i < k && done < total; i++) {
      const pair = state.prefs[Math.floor(Math.random() * state.prefs.length)]
      lastLoss = dpoTrainStep(state.model.params, state.refParams, state.model.cfg, state.dpoOpt, pair.x, pair.yw, pair.yl, beta).loss
      done++
    }
    $('dpoInfo').textContent = `DPO 训练中… ${done}/${total}  loss=${lastLoss.toFixed(4)}`
    requestAnimationFrame(loop)
  }
  loop()
})

$('dpoResetBtn').addEventListener('click', () => {
  state.prefs = []
  state.pair = null
  $('prefInfo').textContent = '已收集 0 对偏好'
  $('pairBox').hidden = true
  $('dpoBtn').disabled = true
  $('dpoInfo').textContent = ''
})

// ---------- 盲测（DPO 验收：猜猜哪个是对齐后的模型）----------
function answerFrom(params, x, temp) {
  const seq = sample(params, x, 24, state.model.cfg, { temperature: temp })
  let end = seq.length
  for (let i = x.length; i < seq.length; i++) {
    if (state.model.itos[seq[i]] === '\u0003') { end = i; break }
  }
  return tokensToText(state.model.itos, seq.slice(x.length, end)) || '（空）'
}

function initBlind() {
  const box = $('blindBox')
  box.hidden = false
  if (box.dataset.ready) return
  box.dataset.ready = '1'
  box.innerHTML = `
    <div class="blind-title">🔍 盲测：猜猜哪个是「对齐后」的模型？</div>
    <div class="row">
      <input id="blindQ" value="你好" size="12">
      <button id="blindGen" class="btn ghost">生成两版回答</button>
    </div>
    <div id="blindPair" class="pair"></div>
    <div id="blindResult" class="muted"></div>
  `
  $('blindGen').addEventListener('click', blindGenerate)
}

function blindGenerate() {
  if (!state.refParams) { $('blindResult').textContent = '（还没有参考模型，先做 SFT 和 DPO）'; return }
  const q = $('blindQ').value.trim() || '你好'
  const x = qaPrompt(state.model.stoi, q)
  const cur = answerFrom(state.model.params, x, 0.8) // DPO 后
  const ref = answerFrom(state.refParams, x, 0.8)   // DPO 前
  const swap = Math.random() < 0.5
  state.blind = { a: swap ? ref : cur, b: swap ? cur : ref, correctIsA: !swap }
  const pair = $('blindPair')
  pair.innerHTML = `
    <div class="answer"><div class="ans-text">${state.blind.a}</div><button id="guessA" class="btn ghost">我猜这个是对齐后</button></div>
    <div class="answer"><div class="ans-text">${state.blind.b}</div><button id="guessB" class="btn ghost">我猜这个是对齐后</button></div>
  `
  $('guessA').addEventListener('click', () => blindGuess(true))
  $('guessB').addEventListener('click', () => blindGuess(false))
  $('blindResult').textContent = '点选你的猜测…'
}

function blindGuess(guessIsA) {
  const correct = state.blind.correctIsA === guessIsA
  const real = state.blind.correctIsA ? state.blind.a : state.blind.b
  $('blindResult').textContent = (correct ? '✅ 猜对了！' : '❌ 猜错了。') +
    ` 对齐后的回答是「${real}」。${correct ? '你已经能分辨模型的"性格"了。' : '再感受一下两者的差别。'}`
}

// ---------- 语料选择 ----------
function setCorpus(text, title) {
  $('corpus').value = text
  $('corpusInfo').textContent = `「${title}」 ${text.length} 字`
  stopTraining()
  buildModel()
  lossChart.draw()
}
document.querySelectorAll('.chip').forEach((btn) => {
  btn.addEventListener('click', () => {
    const c = CORPUS.find((x) => x.id === btn.dataset.id)
    setCorpus(c.text, c.title)
  })
})
$('corpus').addEventListener('input', () => {
  $('corpusInfo').textContent = `自定义语料 ${$('corpus').value.length} 字`
  stopTraining()
  buildModel()
  lossChart.draw()
})

// ---------- 流式生成 ----------
let genTimer = null
$('genBtn').addEventListener('click', () => {
  if (!state.model) return
  state.genCount++
  // 彩蛋：致敬 Karpathy（连续生成 5 次）
  if (state.genCount === 5 && !document.querySelector('.egg-karpathy')) {
    const kg = document.createElement('div')
    kg.className = 'egg egg-karpathy'
    kg.textContent = "Let's build GPT! — 致敬 Andrej Karpathy"
    $('genOut').after(kg)
  }
  if (genTimer) { clearInterval(genTimer); genTimer = null }
  const temp = parseFloat($('temp').value) || 1
  const len = parseInt($('len').value) || 32
  const out = $('genOut')

  if (state.mode === 'qa') {
    // 问答模式：<u>问题<a> → 模型生成回答
    const q = $('prompt').value.trim() || '你是谁'
    const p = qaPrompt(state.model.stoi, q)
    const sampOpts = { temperature: temp, topK: parseInt($('topk').value) || 0, topP: parseFloat($('topp').value) || 1 }
    const t0 = performance.now()
    const seq = sample(state.model.params, p, len, state.model.cfg, sampOpts)
    const msPerTok = (performance.now() - t0) / Math.max(1, seq.length - p.length)
    $('perf').textContent = `${msPerTok.toFixed(1)} ms/token · ${(1000 / msPerTok).toFixed(0)} tokens/s`
    const text = tokensToText(state.model.itos, seq)
    const aIdx = text.indexOf('<a>')
    let answer = aIdx >= 0 ? text.slice(aIdx + 3) : text
    const eIdx = answer.indexOf('<e>')
    if (eIdx >= 0) answer = answer.slice(0, eIdx)
    out.textContent = '答：'
    let i = 0
    genTimer = setInterval(() => {
      if (i < answer.length) { out.textContent += answer[i]; i++ }
      else { clearInterval(genTimer); genTimer = null }
    }, 40)
  } else {
    // 续写模式（带 Attention 直播：文本流与热力图同节奏）
    const prompt = $('prompt').value
    const p = prompt.split('').map((c) => state.model.stoi[c] ?? 0)
    const sampOpts = { temperature: temp, topK: parseInt($('topk').value) || 0, topP: parseFloat($('topp').value) || 1 }
    const t0 = performance.now()
    const { seq, attnSteps } = sampleWithAttn(state.model.params, p, len, state.model.cfg, sampOpts)
    const msPerTok = (performance.now() - t0) / Math.max(1, attnSteps.length)
    $('perf').textContent = `${msPerTok.toFixed(1)} ms/token · ${(1000 / msPerTok).toFixed(0)} tokens/s · ${paramCount(state.model.params)} 参数`
    const win = seq.slice(0, Math.min(seq.length, state.model.cfg.block_size))
    attnHeatmap.setContext(win.map((i) => state.model.itos[i]))
    attnHeatmap.clear()
    out.textContent = prompt
    let i = prompt.length
    let stepIdx = 0
    genTimer = setInterval(() => {
      if (i < seq.length) {
        const tok = state.model.itos[seq[i]]
        out.textContent += tok
        if (stepIdx < attnSteps.length) {
          attnHeatmap.pushStep(tok, attnSteps[stepIdx])
          attnHeatmap.draw()
          stepIdx++
        }
        i++
      } else {
        clearInterval(genTimer)
        genTimer = null
      }
    }, 80)
  }
})

// ---------- 双层讲解（纸张翻面：正面直觉 / 背面公式）----------
function initNotes() {
  let flipped = false
  for (const id in NOTES) {
    const card = $(id)
    if (!card) continue
    const h2 = card.querySelector('h2')
    if (!h2) continue
    const n = NOTES[id]
    const div = document.createElement('div')
    div.className = 'note'
    div.innerHTML = `<span class="note-novice">${n.novice}</span><span class="note-expert" hidden>${n.expert}</span>`
    h2.after(div)
  }
  $('flipBtn').addEventListener('click', () => {
    flipped = !flipped
    document.querySelectorAll('.note-expert').forEach((el) => (el.hidden = !flipped))
    document.querySelectorAll('.note-novice').forEach((el) => (el.hidden = flipped))
    $('flipBtn').textContent = flipped ? '翻回 · 直觉模式' : '翻面 · 进阶模式'
  })
}

// ---------- 启动 ----------
initNotes()
initMicroscopeUI($('microscopeRoot'), () => state.model)

// 彩蛋：深夜引言（00:00-04:00 打开页面）
const _h = new Date().getHours()
if (_h >= 0 && _h < 4) {
  const _sub = document.querySelector('.sub')
  if (_sub) _sub.textContent = '夜深了，数字的世界依然清醒。'
}

// 彩蛋：无穷小 μ（档位拉到最小）
$('size').addEventListener('change', () => {
  const title = document.querySelector('.title')
  const mu = title.querySelector('.mu')
  if ($('size').value === 'tiny' && !mu) title.insertAdjacentHTML('beforeend', '<span class="mu"> μ</span>')
  if ($('size').value !== 'tiny' && mu) mu.remove()
})

setCorpus(CORPUS[0].text, CORPUS[0].title)
