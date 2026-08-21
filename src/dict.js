// ============================================================
// src/dict.js —— 动画词典：每个术语用迷你动画+一句话解释
// ============================================================

function R(x, y) { // 便捷 fillText
  return { x, y }
}

// ---------- 各术语动画（ctx, w, h, t）t=秒 ----------

function drawToken(ctx, w, h, t) {
  const s = '月光如流水'
  const n = s.length
  const cw = w / n
  const i = Math.floor(t / 0.6) % n
  ctx.font = '20px serif'
  ctx.textAlign = 'center'
  for (let k = 0; k < n; k++) {
    ctx.fillStyle = k === i ? '#b3442c' : '#6b6357'
    ctx.fillText(s[k], cw * k + cw / 2, h / 2 + 7)
    if (k === i) {
      ctx.strokeStyle = '#b3442c'
      ctx.lineWidth = 2
      ctx.strokeRect(cw * k + 4, h / 2 - 18, cw - 8, 34)
      ctx.fillStyle = '#b3442c'
      ctx.font = '10px serif'
      ctx.fillText(`token${i}`, cw * k + cw / 2, h / 2 + 26)
    }
  }
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText('文本被切成一个个"token"（这里=每个字）', w / 2, h - 6)
}

function drawAttention(ctx, w, h, t) {
  const nodes = ['月', '光', '荷', '塘', '叶']
  const n = nodes.length
  const xs = nodes.map((_, i) => 30 + i * ((w - 60) / (n - 1)))
  const cur = Math.floor(t / 0.8) % n
  ctx.font = '16px serif'
  ctx.textAlign = 'center'
  nodes.forEach((c, i) => {
    ctx.fillStyle = i === cur ? '#b3442c' : '#6b6357'
    ctx.fillText(c, xs[i], h / 2 + 6)
  })
  // 当前节点看其他节点：连线粗细随权重
  for (let i = 0; i < n; i++) {
    if (i === cur) continue
    const weight = 0.2 + 0.8 * Math.abs(Math.sin(t * 2 + i))
    ctx.strokeStyle = `rgba(31,122,109,${0.3 + weight * 0.5})`
    ctx.lineWidth = 1 + weight * 4
    ctx.beginPath()
    ctx.moveTo(xs[i], h / 2 - 8)
    ctx.lineTo(xs[cur], h / 2 - 8)
    ctx.stroke()
  }
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText(`「${nodes[cur]}」在看其他字（线越粗=注意力越多）`, w / 2, h - 6)
}

function drawSoftmax(ctx, w, h, t) {
  const n = 10
  const slot = w / n
  const phase = (Math.sin(t * 1.2) + 1) / 2
  const vals = []
  for (let i = 0; i < n; i++) vals.push(0.3 + 0.7 * Math.abs(Math.sin(i * 1.7 + t * 2)))
  const sum = vals.reduce((a, b) => a + b, 0)
  const bh = (h - 40)
  for (let i = 0; i < n; i++) {
    const rawH = vals[i] * (h / 2)
    const probH = (vals[i] / sum) * bh
    const hh = rawH * (1 - phase) + probH * phase
    ctx.fillStyle = i === 0 ? '#b3442c' : 'rgba(31,122,109,0.7)'
    ctx.fillRect(i * slot + 2, h - 10 - hh, slot - 4, hh)
  }
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.textAlign = 'center'
  ctx.fillText(phase > 0.5 ? '→ 变成概率（总和=100%）' : '原始分数', w / 2, h - 3)
}

function drawTemperature(ctx, w, h, t) {
  const flat = (Math.sin(t * 1) + 1) / 2 // 0=锐,1=平
  ctx.strokeStyle = '#1f7a6d'
  ctx.lineWidth = 2.5
  ctx.beginPath()
  for (let px = 0; px <= w; px += 3) {
    const x = (px / w) * 6 - 3
    const sigma = 0.5 + flat * 1.6
    const y = h / 2 - Math.exp(-(x * x) / (2 * sigma * sigma)) * (h / 2 - 8) * (1 / (sigma * 0.7 + 0.5))
    px === 0 ? ctx.moveTo(px, y) : ctx.lineTo(px, y)
  }
  ctx.stroke()
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.textAlign = 'center'
  ctx.fillText(flat > 0.5 ? '温度高 → 分布变平（爱冒险）' : '温度低 → 分布变尖（只选最稳）', w / 2, h - 5)
}

function drawTopK(ctx, w, h, t) {
  const n = 8
  const slot = w / n
  const sortPhase = Math.sin(t * 1) > 0 ? 1 : 0
  const vals = Array.from({ length: n }, (_, i) => 0.3 + 0.7 * Math.abs(Math.sin(i * 2.3 + t)))
  const order = vals.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])
  for (let i = 0; i < n; i++) {
    const srcIdx = sortPhase ? order[i][1] : i
    const hh = vals[srcIdx] * (h - 30)
    const keep = i < 3
    ctx.fillStyle = keep ? '#b3442c' : 'rgba(43,38,32,0.15)'
    ctx.fillRect(i * slot + 2, h - 10 - hh, slot - 4, hh)
  }
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.textAlign = 'center'
  ctx.fillText('top-k=3：只保留概率最高的 3 个（红），其余淘汰', w / 2, h - 3)
}

function drawContext(ctx, w, h, t) {
  const tokens = ['月', '光', '如', '流', '水', '一', '般', '青', '雾', '荷', '塘', '叶', '子']
  const n = tokens.length
  const tw = Math.min(24, w / n)
  const win = 5
  const start = Math.floor(t / 0.7) % (n - win)
  ctx.font = '14px serif'
  ctx.textAlign = 'center'
  for (let i = 0; i < n; i++) {
    const inWin = i >= start && i < start + win
    ctx.fillStyle = inWin ? '#2b2620' : 'rgba(43,38,32,0.2)'
    ctx.fillText(tokens[i], i * tw + tw / 2, h / 2 + 5)
  }
  ctx.strokeStyle = '#b3442c'
  ctx.lineWidth = 2
  ctx.strokeRect(start * tw - 2, h / 2 - 20, win * tw + 4, 36)
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText(`上下文窗口 = 只看最近的 ${win} 个 token（框内），更早的淡出`, w / 2, h - 4)
}

function drawCacheHit(ctx, w, h, t) {
  const phase = Math.floor(t / 1.4) % 2 // 0=第一次计算,1=命中
  ctx.font = '12px serif'
  ctx.textAlign = 'left'
  const y1 = h / 3, y2 = (h * 2) / 3
  ctx.fillStyle = '#6b6357'
  ctx.fillText('第一次：逐 token 算 K/V（慢）', 10, y1 - 4)
  ctx.fillStyle = '#1f7a6d'
  ctx.fillText('第二次：缓存命中（秒取）', 10, y2 - 4)
  // 第一行：逐字点亮
  const n1 = 8
  const progress1 = phase === 0 ? (t % 1.4) / 1.4 : 1
  for (let i = 0; i < n1; i++) {
    ctx.fillStyle = i < progress1 * n1 ? '#b3442c' : 'rgba(43,38,32,0.1)'
    ctx.fillRect(10 + i * ((w - 30) / n1), y1, (w - 30) / n1 - 4, 18)
  }
  // 第二行：一道闪电秒过
  if (phase === 1) {
    const flash = (t % 1.4) / 1.4
    ctx.fillStyle = '#1f7a6d'
    ctx.fillRect(10, y2, (w - 30) * flash, 18)
    if (flash > 0.1 && flash < 0.6) {
      ctx.fillStyle = '#1f7a6d'
      ctx.font = 'bold 16px serif'
      ctx.fillText('⚡', 10 + (w - 30) * flash, y2 + 15)
    }
  } else {
    ctx.fillStyle = 'rgba(43,38,32,0.1)'
    ctx.fillRect(10, y2, w - 30, 18)
  }
}

function drawEmbedding(ctx, w, h, t) {
  const pop = Math.min(1, t * 0.8 % 1.6)
  ctx.font = '22px serif'
  ctx.textAlign = 'center'
  ctx.fillStyle = '#2b2620'
  ctx.fillText('月', w / 4, h / 2 + 8)
  ctx.strokeStyle = '#b3442c'
  ctx.lineWidth = 2
  ctx.strokeRect(w / 4 - 14, h / 2 - 18, 28, 34)
  const vecX = (w / 2) + 20
  const cellH = 18
  const n = 6
  for (let i = 0; i < n; i++) {
    const v = (Math.sin(i * 3.1 + pop * 10) * 0.5 + 0.5)
    ctx.fillStyle = `rgba(31,122,109,${0.2 + v * 0.6})`
    const y = h / 2 - (n / 2) * cellH + i * cellH
    ctx.fillRect(vecX, y, 30, cellH - 2)
    ctx.fillStyle = '#6b6357'
    ctx.font = '9px monospace'
    ctx.fillText(v.toFixed(2), vecX + 15, y + 13)
  }
  ctx.font = '11px serif'
  ctx.fillStyle = '#6b6357'
  ctx.fillText('字 → 一组数字（向量）', w / 2, h - 5)
}

function drawMulti(ctx, w, h, t) {
  const tags = [['文', '#b3442c'], ['图', '#1f7a6d'], ['音', '#8a5a44']]
  const flow = (t % 2) / 2
  ctx.font = '14px serif'
  ctx.textAlign = 'center'
  tags.forEach(([tag, color], i) => {
    const x = 30 + i * 55
    const y = 20 + flow * (h - 60)
    ctx.fillStyle = color
    ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2); ctx.fill()
    ctx.fillStyle = '#fffdf6'
    ctx.fillText(tag, x, y + 5)
  })
  ctx.strokeStyle = '#2b2620'
  ctx.lineWidth = 2.5
  ctx.strokeRect(w - 90, h / 2 - 25, 70, 50)
  ctx.fillStyle = '#2b2620'
  ctx.font = '12px serif'
  ctx.fillText('模型', w - 55, h / 2 + 4)
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText('文本/图像/声音 → 同一个 Transformer', w / 2, h - 4)
}

// ---------- 词典数据 ----------
export const DICT_ITEMS = [
  { term: 'Token 词元', one: '模型眼里的"一个单元"。文本被切成的块。', draw: drawToken },
  { term: '注意力 Attention', one: '生成时"看"前文哪些字、各看多少。', draw: drawAttention },
  { term: 'Softmax', one: '把任意分数变成概率（总和=100%）。', draw: drawSoftmax },
  { term: '温度 Temperature', one: '生成时多"冒险"：低=稳，高=乱。', draw: drawTemperature },
  { term: 'Top-k', one: '只从概率最高的前 k 个候选里选。', draw: drawTopK },
  { term: '上下文窗口', one: '模型一次能"看到"多少个前文。', draw: drawContext },
  { term: '缓存命中 Cache Hit', one: 'K/V 算过一遍就缓存，下次直接取（提速）。', draw: drawCacheHit },
  { term: 'Embedding', one: '把字变成一组数字（向量），让模型能算。', draw: drawEmbedding },
  { term: '多模态', one: '一个 Transformer 处理文字+图像+声音。', draw: drawMulti },
]

// ---------- 词典渲染引擎：驱动所有词条画布 ----------
export function initDict(root) {
  const anims = []
  root.innerHTML = `<div class="dict-grid">${DICT_ITEMS.map((it, i) => `
    <div class="dict-item card">
      <div class="dict-term">${it.term}</div>
      <div class="dict-one">${it.one}</div>
      <canvas class="dict-canvas" data-i="${i}"></canvas>
    </div>`).join('')}</div>`
  root.querySelectorAll('.dict-canvas').forEach((cv) => {
    anims.push({ cv, draw: DICT_ITEMS[+cv.dataset.i].draw })
  })
  let last = performance.now()
  let t = 0
  function loop() {
    const now = performance.now()
    t += (now - last) / 1000
    last = now
    for (const { cv, draw } of anims) {
      const dpr = window.devicePixelRatio || 1
      const w = cv.clientWidth || 260
      const h = cv.clientHeight || 120
      if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr }
      const ctx = cv.getContext('2d')
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      try { draw(ctx, w, h, t) } catch (e) { /* 忽略 */ }
    }
    requestAnimationFrame(loop)
  }
  requestAnimationFrame(loop)
}