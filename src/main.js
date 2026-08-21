// ============================================================
// src/main.js
// 手算LM —— M1：预训练实验台（数据区 + 训练控制 + 基础可视化 + 流式生成）
// ============================================================

import './style.css'
import { CORPUS, buildVocab } from './corpus.js'
import { createModel, paramCount } from './model.js'
import { trainStep, createOptimizer } from './train.js'
import { sample } from './sample.js'
import { createLossChart, createHeatmap } from './ui.js'

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
}

// ---------- DOM ----------
const app = document.getElementById('app')
app.innerHTML = `
  <main class="stage">
    <header class="masthead">
      <h1 class="title">手算<span class="hl">LM</span></h1>
      <p class="sub">给你权重，亲手算出它的下一句话</p>
    </header>

    <section class="card" id="dataCard">
      <h2>壹 · 语料</h2>
      <div class="corpus-pick">
        ${CORPUS.map((c) => `<button class="chip" data-id="${c.id}">${c.title}</button>`).join('')}
        <span class="hint">或直接粘贴你的文本 ↓</span>
      </div>
      <textarea id="corpus" rows="4"></textarea>
      <p class="muted" id="corpusInfo"></p>
    </section>

    <section class="card">
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
      </div>
      <p class="muted" id="modelInfo"></p>
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

    <section class="card">
      <h2>叁 · 生成</h2>
      <div class="row">
        <input id="prompt" value="月光" size="14">
        <button id="genBtn" class="btn" disabled>生成</button>
        <label>温度 <input id="temp" value="0.8" size="4"></label>
        <label>长度 <input id="len" value="32" size="4"></label>
      </div>
      <div id="genOut" class="gen">（先训练，再让它续写）</div>
    </section>
  </main>
`

const $ = (id) => document.getElementById(id)
const lossChart = createLossChart($('lossChart'))
const heatmap = createHeatmap($('heatmap'))
$('heatmap').addEventListener('mousemove', (e) => { heatmap.onHover(e); heatmap.draw() })
$('heatmap').addEventListener('mouseleave', () => { heatmap.draw() })

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
  state.losses = []
  lossChart.clear()
  heatmap.set(params.wte.value, 'wte')
  heatmap.draw()
  $('modelInfo').textContent = `${chars.length} 字符 · ${paramCount(params)} 参数 · ${cfg.n_layer}层${cfg.n_head}头${cfg.n_embd}维`
  $('genBtn').disabled = false
  return true
}

// ---------- 训练 ----------
function runSteps(n) {
  const { model, opt } = state
  const ids = $('corpus').value.split('')
  const L = ids.length
  for (let k = 0; k < n; k++) {
    const i = Math.floor(Math.random() * Math.max(1, L - model.cfg.block_size - 1))
    const x = ids.slice(i, i + model.cfg.block_size).map((c) => model.stoi[c])
    const y = ids.slice(i + 1, i + model.cfg.block_size + 1).map((c) => model.stoi[c])
    const { loss } = trainStep(model.params, x, y, model.cfg, opt)
    state.losses.push(loss)
  }
  lossChart.push(state.losses[state.losses.length - 1])
  if (state.losses.length % 50 === 0) heatmap.draw()
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
  if (genTimer) { clearInterval(genTimer); genTimer = null }
  const prompt = $('prompt').value
  const temp = parseFloat($('temp').value) || 1
  const len = parseInt($('len').value) || 32
  const p = prompt.split('').map((c) => state.model.stoi[c] ?? 0)
  const seq = sample(state.model.params, p, len, state.model.cfg, { temperature: temp })
  const out = $('genOut')
  out.textContent = prompt
  let i = prompt.length
  genTimer = setInterval(() => {
    if (i < seq.length) {
      out.textContent += state.model.itos[seq[i]]
      i++
    } else {
      clearInterval(genTimer)
      genTimer = null
    }
  }, 60)
})

// ---------- 启动 ----------
setCorpus(CORPUS[0].text, CORPUS[0].title)
