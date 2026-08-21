// ============================================================
// src/dict-anims.js —— 词典动画（每个术语一个迷你动画）
// 绘制函数签名：(ctx, w, h, t) t=秒
// ============================================================

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
  ctx.fillText('文本被切成一个个 token', w / 2, h - 6)
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
  ctx.fillText(`「${nodes[cur]}」在看其他字（线越粗=越关注）`, w / 2, h - 6)
}

function drawSoftmax(ctx, w, h, t) {
  const n = 10
  const slot = w / n
  const phase = (Math.sin(t * 1.2) + 1) / 2
  const vals = []
  for (let i = 0; i < n; i++) vals.push(0.3 + 0.7 * Math.abs(Math.sin(i * 1.7 + t * 2)))
  const sum = vals.reduce((a, b) => a + b, 0)
  const bh = h - 40
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
  const flat = (Math.sin(t) + 1) / 2
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
  ctx.fillText(flat > 0.5 ? '温度高 → 分布平（爱冒险）' : '温度低 → 分布尖（只选最稳）', w / 2, h - 5)
}

function drawTopK(ctx, w, h, t) {
  const n = 8
  const slot = w / n
  const sorted = Math.sin(t) > 0
  const vals = Array.from({ length: n }, (_, i) => 0.3 + 0.7 * Math.abs(Math.sin(i * 2.3 + t)))
  const order = vals.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])
  for (let i = 0; i < n; i++) {
    const srcIdx = sorted ? order[i][1] : i
    const hh = vals[srcIdx] * (h - 30)
    ctx.fillStyle = i < 3 ? '#b3442c' : 'rgba(43,38,32,0.15)'
    ctx.fillRect(i * slot + 2, h - 10 - hh, slot - 4, hh)
  }
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.textAlign = 'center'
  ctx.fillText('top-k=3：只留概率最高的 3 个，其余淘汰', w / 2, h - 3)
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
  ctx.fillText('上下文窗口 = 只看最近几个 token（框内）', w / 2, h - 4)
}

function drawCacheHit(ctx, w, h, t) {
  const phase = Math.floor(t / 1.4) % 2
  const y1 = h / 3, y2 = (h * 2) / 3
  ctx.font = '12px serif'
  ctx.textAlign = 'left'
  ctx.fillStyle = '#6b6357'
  ctx.fillText('第一次：逐 token 算 K/V（慢）', 10, y1 - 4)
  ctx.fillStyle = '#1f7a6d'
  ctx.fillText('第二次：缓存命中（秒取）', 10, y2 - 4)
  const n1 = 8
  const progress1 = phase === 0 ? (t % 1.4) / 1.4 : 1
  for (let i = 0; i < n1; i++) {
    ctx.fillStyle = i < progress1 * n1 ? '#b3442c' : 'rgba(43,38,32,0.1)'
    ctx.fillRect(10 + i * ((w - 30) / n1), y1, (w - 30) / n1 - 4, 18)
  }
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
  const pop = Math.min(1, (t * 0.8) % 1.6)
  ctx.font = '22px serif'
  ctx.textAlign = 'center'
  ctx.fillStyle = '#2b2620'
  ctx.fillText('月', w / 4, h / 2 + 8)
  ctx.strokeStyle = '#b3442c'
  ctx.lineWidth = 2
  ctx.strokeRect(w / 4 - 14, h / 2 - 18, 28, 34)
  const vecX = w / 2 + 20
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

// ---------- 工具链术语动画 ----------

function drawVibe(ctx, w, h, t) {
  const step = Math.floor(t / 1.2) % 3
  // 用户输入气泡
  ctx.fillStyle = '#b3442c'
  ctx.font = '13px serif'
  ctx.textAlign = 'left'
  ctx.fillText('用户：做一个计算器', 12, 30)
  // AI 代码逐行出现
  const lines = ['const calc = (a, op, b) => {', '  if (op === "+") return a + b;', '  return a * b;', '};']
  const visible = step * 2
  ctx.fillStyle = '#2b2620'
  ctx.font = '12px monospace'
  for (let i = 0; i < Math.min(visible, lines.length); i++) {
    ctx.fillText(lines[i], 12, 60 + i * 18)
  }
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText('vibe coding：用自然语言指挥 AI 写代码', w / 2, h - 5)
  ctx.textAlign = 'center'
}

function drawTool(ctx, w, h, t) {
  const phase = (t % 2.4) / 2.4
  const cx = w / 2
  // 模型框
  ctx.fillStyle = '#fffdf6'
  ctx.strokeStyle = '#2b2620'
  ctx.lineWidth = 1.5
  ctx.strokeRect(10, h / 2 - 20, 70, 40)
  ctx.fillStyle = '#2b2620'
  ctx.font = '12px serif'
  ctx.textAlign = 'center'
  ctx.fillText('模型', 45, h / 2 + 4)
  // 工具框
  ctx.strokeRect(w - 80, h / 2 - 20, 70, 40)
  ctx.fillText('工具', w - 45, h / 2 + 4)
  // 调用箭头（去）和结果（回）
  const goX = 80 + phase * (w - 160)
  ctx.strokeStyle = '#b3442c'
  ctx.lineWidth = 2
  ctx.beginPath(); ctx.moveTo(80, h / 2); ctx.lineTo(goX, h / 2); ctx.stroke()
  ctx.fillStyle = '#b3442c'
  ctx.fillText('调用', goX, h / 2 - 8)
  const backX = w - 80 - phase * (w - 160)
  ctx.strokeStyle = '#1f7a6d'
  ctx.beginPath(); ctx.moveTo(w - 80, h / 2 + 10); ctx.lineTo(backX, h / 2 + 10); ctx.stroke()
  ctx.fillStyle = '#1f7a6d'
  ctx.fillText('结果', backX, h / 2 + 22)
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText('Tool：模型调用外部功能，结果回来继续生成', w / 2, h - 5)
}

function drawSkill(ctx, w, h, t) {
  const flow = (t % 2) / 2
  // skill 文档
  ctx.fillStyle = 'rgba(179,68,44,0.1)'
  ctx.strokeStyle = '#b3442c'
  ctx.lineWidth = 1.5
  ctx.strokeRect(10, h / 2 - 40, 80, 80)
  ctx.fillStyle = '#b3442c'
  ctx.font = '12px serif'
  ctx.textAlign = 'center'
  ctx.fillText('SKILL', 50, h / 2 - 20)
  ctx.fillText('指令+流程', 50, h / 2 + 0)
  ctx.fillText('+规范', 50, h / 2 + 20)
  // AI
  ctx.strokeStyle = '#2b2620'
  ctx.strokeRect(w - 90, h / 2 - 25, 70, 50)
  ctx.fillStyle = '#2b2620'
  ctx.fillText('AI', w - 55, h / 2 + 4)
  // 注入箭头
  const x = 90 + flow * (w - 180)
  ctx.strokeStyle = '#1f7a6d'
  ctx.lineWidth = 3
  ctx.beginPath(); ctx.moveTo(90, h / 2); ctx.lineTo(x, h / 2); ctx.stroke()
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText('skill 注入上下文 → AI 按规范干活', w / 2, h - 5)
}

function drawMCP(ctx, w, h, t) {
  const servers = [['GitHub', '#b3442c'], ['数据库', '#1f7a6d'], ['浏览器', '#8a5a44']]
  ctx.font = '11px serif'
  ctx.textAlign = 'center'
  servers.forEach(([name, color], i) => {
    const sx = 15 + i * ((w - 60) / 2)
    const sy = 20
    ctx.fillStyle = color
    ctx.fillRect(sx, sy, 52, 24)
    ctx.fillStyle = '#fffdf6'
    ctx.fillText(name, sx + 26, sy + 16)
    // 连线到中心
    const cx = w / 2
    ctx.strokeStyle = 'rgba(43,38,32,0.4)'
    ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(sx + 26, sy + 24); ctx.lineTo(cx, h - 30); ctx.stroke()
  })
  ctx.fillStyle = '#2b2620'
  ctx.font = '12px serif'
  ctx.fillText('MCP 客户端', w / 2, h - 15)
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText('AI 通过 MCP 协议连接任意外部工具/数据', w / 2, h - 35)
}

function drawAgent(ctx, w, h, t) {
  const steps = ['规划', '行动', '观察']
  const cur = Math.floor(t / 0.7) % 3
  const cx = w / 2
  ctx.font = '14px serif'
  ctx.textAlign = 'center'
  for (let i = 0; i < 3; i++) {
    const x = cx + (i - 1) * 70
    const y = h / 2
    ctx.fillStyle = i === cur ? '#b3442c' : 'rgba(31,122,109,0.2)'
    ctx.beginPath(); ctx.arc(x, y, 22, 0, Math.PI * 2); ctx.fill()
    ctx.fillStyle = i === cur ? '#fffdf6' : '#2b2620'
    ctx.fillText(steps[i], x, y + 5)
  }
  // 循环箭头
  ctx.strokeStyle = '#2b2620'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.arc(cx + 70, h / 2 + 55, 45, Math.PI * 1.1, Math.PI * 1.9, false)
  ctx.stroke()
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText('Agent：规划 → 行动 → 观察 → 再规划（循环）', w / 2, h - 4)
}

function drawCnn(ctx, w, h, t) {
  const n = 10
  const cw = w / n
  const start = Math.floor(t / 0.7) % (n - 2)
  const y = h / 2
  ctx.font = '12px serif'
  ctx.textAlign = 'center'
  for (let k = 0; k < n; k++) {
    ctx.fillStyle = '#6b6357'
    ctx.fillRect(k * cw + cw * 0.18, y - 13, cw * 0.64, 26)
    ctx.fillStyle = '#fdfcf9'
    ctx.fillText(String(k + 1), k * cw + cw / 2, y + 5)
  }
  ctx.strokeStyle = '#b3442c'
  ctx.lineWidth = 2.5
  ctx.strokeRect(start * cw, y - 20, cw * 3, 40)
  ctx.fillStyle = '#b3442c'
  ctx.font = '12px serif'
  ctx.fillText('窗口只看这 3 个邻居', start * cw + cw * 1.5, y - 28)
  ctx.fillStyle = '#6b6357'
  ctx.font = '11px serif'
  ctx.fillText('卷积核滑动，一次只看窗口内（局部视野）', w / 2, h - 8)
}

export const DRAW = { drawToken, drawAttention, drawSoftmax, drawTemperature, drawTopK, drawContext, drawCacheHit, drawEmbedding, drawMulti, drawVibe, drawTool, drawSkill, drawMCP, drawAgent, drawCnn }