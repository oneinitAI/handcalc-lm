// ============================================================
// src/microscope-ui.js
// 手算LM —— 显微镜模式 UI：自动演示（逐步前向）+ 自己算（查表+校验+计算器）
// ============================================================

import { runMicroscope } from './microscope.js'
import { TOKEN_NAME } from './corpus.js'

export function initMicroscopeUI(root, getModel) {
  root.innerHTML = `
    <section class="card">
      <h2>陆 · 显微镜</h2>
      <p class="muted">给你权重集和纸笔，你可以把 LLM 的下一句话算出来。这是当前模型对这句话的<b>真实计算过程</b>——没有魔法，只有乘法。</p>
      <div class="row">
        <input id="msInput" value="月光" size="10">
        <button id="msRun" class="btn">放大观察</button>
        <button id="msTabDemo" class="chip on">看演示</button>
        <button id="msTabCalc" class="chip">自己算</button>
      </div>
      <div id="msDemo"></div>
      <div id="msCalc" hidden></div>
    </section>
  `

  const msInput = root.querySelector('#msInput')
  const msRun = root.querySelector('#msRun')
  const msDemo = root.querySelector('#msDemo')
  const msCalc = root.querySelector('#msCalc')
  const msTabDemo = root.querySelector('#msTabDemo')
  const msTabCalc = root.querySelector('#msTabCalc')

  let lastResult = null

  function tokenName(model, id) {
    return TOKEN_NAME[model.itos[id]] ?? model.itos[id]
  }

  function renderStep(step, model) {
    let html = `<div class="ms-step"><div class="ms-title">${step.title}</div><div class="ms-desc">${step.desc}</div>`
    if (step.rows) {
      html += `<table class="ms-table"><tr><th></th>${step.rows[0] ? step.rows[0].vec.map((_, j) => `<th>${j}</th>`).join('') : ''}</tr>`
      step.rows.forEach((r) => {
        html += `<tr><td class="ms-tok">${tokenName(model, r.tokenId ?? -1)}${r.token}</td>${r.vec.map((v) => `<td>${v}</td>`).join('')}</tr>`
      })
      html += '</table>'
    }
    if (step.attnProb) {
      html += `<div class="ms-attn">注意力权重：${step.attnProb.map((v) => v.toFixed(2)).join(' · ')}</div>`
      html += `<div class="ms-desc">Q 向量：${step.attnQ.join(', ')}</div>`
    }
    if (step.logits) {
      html += `<div class="ms-desc">各候选字分数：</div><div class="ms-logits">${step.logits.map((v, i) => `<span>${tokenName(model, i)}:${v}</span>`).join(' ')}</div>`
    }
    if (step.probs) {
      const top = step.top
      html += `<div class="ms-probs">${step.probs.map((v, i) => `<span class="${i === top ? 'top' : ''}">${tokenName(model, i)} ${(v * 100).toFixed(1)}%</span>`).join(' ')}</div>`
    }
    html += '</div>'
    return html
  }

  function renderDemo(model) {
    if (!lastResult) { msDemo.innerHTML = '<p class="muted">点击"放大观察"</p>'; return }
    let html = lastResult.steps.map((s) => renderStep(s, model)).join('')
    const top = lastResult.predicted
    const topName = tokenName(model, top)
    html += `<div class="ms-final">模型预测的下一个字：<b>「${topName}」</b></div>`
    html += `<div class="ms-slogan">没有魔法，只有乘法。</div>`
    if (topName === '月') html += `<div class="ms-egg">🌙 床前明月光</div>`
    msDemo.innerHTML = html
  }

  msRun.addEventListener('click', () => {
    const model = getModel()
    if (!model) { alert('请先构建模型并训练'); return }
    const text = msInput.value.trim().slice(0, 3)
    const idx = text.split('').map((c) => model.stoi[c] ?? 0)
    lastResult = runMicroscope(model.params, idx, model.cfg)
    if (!msCalc.hidden) renderCalc(model)
    else renderDemo(model)
  })

  // ---------- 自己算（简化版：查表 + attention 加权校验 + 计算器）----------
  function renderCalc(model) {
    if (!lastResult) { msCalc.innerHTML = '<p class="muted">点击"放大观察"后，这里给你权重表让你亲手算</p>'; return }
    const step = lastResult.steps[2] // attention
    const wte = model.params.wte.value
    const html = `
      <div class="ms-calc-intro">你手里有纸笔和权重表。任务：算出「${step.attnProb.length} 个注意力权重 × 对应向量」的<b>第一个分量</b>。</div>
      <div class="ms-wte"><b>wte 词向量表（查表用）：</b>
        <table class="ms-table"><tr><th>字</th>${model.itos.map((_, j) => `<th>${j}</th>`).join('')}</tr>
        ${wte.map((row, i) => `<tr><td class="ms-tok">${tokenName(model, i)}</td>${row.map((v) => `<td>${v.toFixed(2)}</td>`).join('')}</tr>`).join('')}
      </table></div>
      <div class="ms-desc">注意力权重：${step.attnProb.join(' · ')}</div>
      <div class="row">
        <input id="msAns" placeholder="算出的第一个分量" size="12">
        <button id="msCheck" class="btn ghost">校验</button>
        <button id="msSkip" class="btn ghost">看答案</button>
      </div>
      <div id="msCalcResult" class="muted"></div>
      <div class="ms-calc">
        <div class="ms-calc-title">简易计算器</div>
        <div id="calcDisplay" class="ms-calc-display">0</div>
        <div class="ms-calc-keys">${['7','8','9','÷','4','5','6','×','1','2','3','−','0','.','=','+'].map((k) => `<button class="calc-key" data-k="${k}">${k}</button>`).join('')}</div>
      </div>
    `
    msCalc.innerHTML = html
    // 计算器逻辑
    let acc = 0, op = null, fresh = true
    const display = msCalc.querySelector('#calcDisplay')
    const press = (k) => {
      if (k === '=') {
        const b = parseFloat(display.textContent)
        acc = op === '+' ? acc + b : op === '−' ? acc - b : op === '×' ? acc * b : op === '÷' ? (b ? acc / b : NaN) : b
        display.textContent = (Math.round(acc * 1e6) / 1e6).toString()
        op = null; fresh = true; return
      }
      if (['+','−','×','÷'].includes(k)) {
        if (op) { press('=') }
        acc = parseFloat(display.textContent); op = k; fresh = true; return
      }
      if (fresh) { display.textContent = k === '.' ? '0.' : k; fresh = false }
      else display.textContent = display.textContent === '0' && k !== '.' ? k : display.textContent + k
    }
    msCalc.querySelectorAll('.calc-key').forEach((b) => b.addEventListener('click', () => press(b.dataset.k)))
    // 校验：答案 = Σ attnProb[s] × attnQ... 不对，任务是"注意力加权第一个分量"。
    // 简化答案：Σ_s attnProb[s] × wte[idx[s]][0]（第一个 token 向量第 0 维加权）
    const idx = lastResult.context
    const target = idx.reduce((sum, id, s) => sum + step.attnProb[s] * model.params.wte.value[id][0], 0)
    const check = msCalc.querySelector('#msCheck')
    const ans = msCalc.querySelector('#msAns')
    const res = msCalc.querySelector('#msCalcResult')
    check.addEventListener('click', () => {
      const v = parseFloat(ans.value)
      if (isNaN(v)) { res.textContent = '请输入数字'; return }
      const ok = Math.abs(v - target) < 0.05
      res.textContent = ok ? '✅ 算对了！你亲手算出了模型的内部计算。' : `还差一点，答案是 ${target.toFixed(3)}（可跳过）`
    })
    msCalc.querySelector('#msSkip').addEventListener('click', () => {
      res.textContent = `答案是 ${target.toFixed(3)}。这就是全部——没有魔法，只有乘法。`
    })
  }

  msTabDemo.addEventListener('click', () => {
    msTabDemo.classList.add('on'); msTabCalc.classList.remove('on')
    msDemo.hidden = false; msCalc.hidden = true
    renderDemo(getModel())
  })
  msTabCalc.addEventListener('click', () => {
    msTabCalc.classList.add('on'); msTabDemo.classList.remove('on')
    msDemo.hidden = true; msCalc.hidden = false
    renderCalc(getModel())
  })
}
