// ============================================================
// src/main.js
// 手算LM —— 浏览器入口（M0 功能验证版，艺术品 UI 在后续里程碑）
// ============================================================

import './style.css'
import { createModel, paramCount } from './model.js'
import { trainStep, createOptimizer } from './train.js'
import { sample } from './sample.js'

const SIZES = {
  tiny:   { name: '微',   cfg: { vocab_size: 0, block_size: 8,  n_layer: 1, n_head: 1, n_embd: 8,  bias: true } },
  small:  { name: '小',   cfg: { vocab_size: 0, block_size: 12, n_layer: 2, n_head: 2, n_embd: 16, bias: true } },
  medium: { name: '中',   cfg: { vocab_size: 0, block_size: 16, n_layer: 2, n_head: 4, n_embd: 32, bias: true } },
}

let model = null
let opt = null
let meta = null

const app = document.getElementById('app')
app.innerHTML = `
  <main class="stage">
    <h1 class="title">手算<span class="hl">LM</span></h1>
    <p class="sub">给你权重，亲手算出它的下一句话</p>

    <section class="card">
      <h2>① 语料</h2>
      <textarea id="corpus" rows="3">月光洒在荷塘上，水面泛起银色的涟漪，风轻轻吹过，荷叶随风摇曳，远处传来几声蛙鸣。</textarea>
    </section>

    <section class="card">
      <h2>② 模型与训练</h2>
      <div class="row">
        <label>档位
          <select id="size">
            <option value="tiny">微</option>
            <option value="small">小</option>
            <option value="medium" selected>中</option>
          </select>
        </label>
        <label>学习率 <input id="lr" value="0.01" size="6"></label>
        <label>步数 <input id="steps" value="1500" size="6"></label>
        <button id="trainBtn" class="btn">开始训练</button>
      </div>
      <p id="modelInfo" class="muted"></p>
      <div id="lossBox" class="loss"></div>
    </section>

    <section class="card">
      <h2>③ 生成</h2>
      <div class="row">
        <input id="prompt" value="月光" size="12">
        <button id="genBtn" class="btn" disabled>生成</button>
        <label>温度 <input id="temp" value="0.8" size="4"></label>
      </div>
      <div id="genOut" class="gen"></div>
    </section>
  </main>
`

const $ = (id) => document.getElementById(id)

function buildModel() {
  const text = $('corpus').value
  if (!text) { alert('请输入语料'); return null }
  const chars = [...new Set(text.split(''))].sort()
  const stoi = Object.fromEntries(chars.map((c, i) => [c, i]))
  const itos = chars
  const size = SIZES[$('size').value]
  const cfg = { ...size.cfg, vocab_size: chars.length }
  const { params } = createModel(cfg, 42)
  model = { params, cfg, stoi, itos }
  opt = createOptimizer(params, { type: 'adam', lr: parseFloat($('lr').value) })
  meta = null
  $('modelInfo').textContent = `模型已构建：${chars.length} 字符，${paramCount(params)} 参数，${cfg.n_layer} 层 ${cfg.n_head} 头 ${cfg.n_embd} 维`
  $('genBtn').disabled = false
}

function trainChunk(remaining, losses, onDone) {
  if (remaining <= 0) { onDone(losses); return }
  const n = Math.min(200, remaining)
  const ids = model.cfg.block_size
  const textIds = $('corpus').value.split('').map((c) => model.stoi[c])
  for (let k = 0; k < n; k++) {
    const i = Math.floor(Math.random() * Math.max(1, textIds.length - ids - 1))
    const x = textIds.slice(i, i + ids)
    const y = textIds.slice(i + 1, i + ids + 1)
    const { loss } = trainStep(model.params, x, y, model.cfg, opt)
    losses.push(loss)
  }
  renderLoss(losses)
  requestAnimationFrame(() => trainChunk(remaining - n, losses, onDone))
}

function renderLoss(losses) {
  const n = losses.length
  const last = losses[n - 1]
  $('lossBox').textContent = n ? `step ${n}  loss=${last.toFixed(4)}` : ''
}

$('trainBtn').addEventListener('click', () => {
  if (!model) buildModel()
  if (!model) return
  $('trainBtn').disabled = true
  const total = parseInt($('steps').value) || 1000
  trainChunk(total, [], () => { $('trainBtn').disabled = false })
})

$('genBtn').addEventListener('click', () => {
  if (!model) return
  const prompt = $('prompt').value
  const p = prompt.split('').map((c) => model.stoi[c] ?? 0)
  const temp = parseFloat($('temp').value) || 1
  const gen = sample(model.params, p, 24, model.cfg, { temperature: temp })
  $('genOut').textContent = gen.map((i) => model.itos[i]).join('')
})

buildModel()
