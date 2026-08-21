// ============================================================
// src/microscope-ui.js
// AI 学习本 —— 显微镜模式 UI：自动演示（逐步前向）+ 自己算（查表+校验+计算器）
// ============================================================

import { runMicroscope } from './microscope.js'
import { TOKEN_NAME } from './corpus.js'

export function initMicroscopeUI(root, getModel) {
  root.innerHTML = `
    <section class="card">
      <h2>陆 · 显微镜</h2>
      <p class="muted">给你权重表（模型里存的所有数字）和纸笔，你可以把模型的下一句话算出来。这里说的"模型"，就是你在「文本模型」tab 训练的那个"猜下一个字"的小家伙（也叫手算 LM）。这是它对待这句话的<b>真实计算过程</b>——没有魔法，只有乘法。</p>
      <div class="howto">① 输入几个字（只看前 3 个）→ 点「放大观察」<br>② 看 6 步真实计算：查表 → 位置 → 注意力 → 前馈 → 打分 → 选字<br>③ 切「自己算」：用权重表 + 内置计算器亲手算出答案<br>④ 页边楷体批注是新手提示，点右上「翻面·进阶模式」看公式</div>
      <div class="row">
        <input id="msInput" value="月光" size="10" title="输入几个字（只看前 3 个），显微镜会展示模型如何预测下一个字">
        <button id="msRun" class="btn" title="用当前模型真实权重，逐步展示预测下一个字的完整计算过程">放大观察</button>
        <button id="msTabDemo" class="chip on" title="看演示：系统逐步播放每一步计算（查表→注意力→变换→选字）">看演示</button>
        <button id="msTabCalc" class="chip" title="自己算：用权重表 + 内置计算器亲手算出答案，全部算对会触发彩蛋">自己算</button>
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
      const topN = step.logits.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 12)
      html += `<div class='ms-desc'>各候选字分数（前 12，共 ${step.logits.length} 个）：</div><div class='ms-logits'>` +
        topN.map(([v, i]) => `<span>${tokenName(model, i)}:${v}</span>`).join(' ') + `</div>`
    }
    if (step.probs) {
      const topN = step.probs.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 12)
      html += `<div class='ms-probs'>` +
        topN.map(([v, i]) => `<span class='${i === step.top ? 'top' : ''}'>${tokenName(model, i)} ${(v * 100).toFixed(1)}%</span>`).join(' ') +
        `<span class='hint'>（前 12，共 ${step.probs.length} 个）</span></div>`
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
      <div class="ms-calc-intro">你手里有纸笔和权重表。<b>这一步在算"注意力怎么把大家混起来"</b>：每个词的新理解 = 它看的每个旧词 × 注意它的比例，全部加起来。你只要算加起来的<b>第一个数</b>（向量最上面那一格，也就是"第一个分量"）就行。</div>
      <div class="ms-wte"><b>wte 词向量表（查表用）：</b>
        <table class="ms-table"><tr><th>字</th>${model.itos.map((_, j) => `<th>${j}</th>`).join('')}</tr>
        ${wte.map((row, i) => `<tr><td class="ms-tok">${tokenName(model, i)}</td>${row.map((v) => `<td>${v.toFixed(2)}</td>`).join('')}</tr>`).join('')}
      </table></div>
      <div class="ms-desc">注意力权重：${step.attnProb.join(' · ')}</div>
      <div class="row">
        <input id="msAns" placeholder="算出的第一个分量" size="12" title="把你的计算结果填在这里，然后点校验">
        <button id="msCheck" class="btn ghost" title="检查你的答案对不对（答对会解锁成就）">校验</button>
        <button id="msSkip" class="btn ghost" title="算不出来就点这里直接看答案">看答案</button>
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
      if (ok) document.dispatchEvent(new CustomEvent('handcalc:calcwin'))
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
