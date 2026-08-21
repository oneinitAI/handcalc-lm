// ============================================================
// src/ui.js
// 手算LM —— 可视化组件：loss 曲线 + 权重热力图（canvas）
// 可读性铁律：所有颜色都映射到具体数值，鼠标悬停显示精确值。
// ============================================================

/** 处理 canvas 高分屏缩放 */
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr
    canvas.height = h * dpr
  }
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  return { ctx, w, h }
}

/** 暖色调映射：低→白暖，高→朱砂深红（数学手稿 × 朱砂印） */
function warmColor(t) {
  const r = 179, g = 68, b = 44
  // t∈[0,1]：0 → 纸色 #faf7f0，1 → 朱砂 #b3442c
  const paper = [250, 247, 240]
  const mix = (a, c) => Math.round(a + (c - a) * Math.sqrt(t))
  return `rgb(${mix(paper[0], r)},${mix(paper[1], g)},${mix(paper[2], b)})`
}

// ---------- loss 曲线 ----------

export function createLossChart(canvas) {
  let losses = []
  const maxPoints = 5000
  const pad = 10

  const api = {
    push(v) { losses.push(v); if (losses.length > maxPoints) losses = losses.slice(-maxPoints) },
    clear() { losses = [] },
    get length() { return losses.length },
    draw() {
      const { ctx, w, h } = fitCanvas(canvas)
      // 网格
      ctx.strokeStyle = 'rgba(43,38,32,0.08)'
      ctx.lineWidth = 1
      for (let gx = 0; gx <= 5; gx++) {
        ctx.beginPath(); ctx.moveTo(pad + (gx / 5) * (w - 2 * pad), 4); ctx.lineTo(pad + (gx / 5) * (w - 2 * pad), h - 12); ctx.stroke()
      }
      for (let gy = 0; gy <= 4; gy++) {
        ctx.beginPath(); ctx.moveTo(4, pad + (gy / 4) * (h - pad - 12)); ctx.lineTo(w - pad, pad + (gy / 4) * (h - pad - 12)); ctx.stroke()
      }
      if (losses.length === 0) {
        ctx.fillStyle = '#6b6357'; ctx.font = '12px serif'; ctx.textAlign = 'center'
        ctx.fillText('训练开始后，这里会出现 loss 曲线', w / 2, h / 2)
        return
      }
      // 动态范围
      let min = Infinity, max = -Infinity
      for (const v of losses) { if (v < min) min = v; if (v > max) max = v }
      const range = max - min || 1
      const yTop = pad, yBot = h - 12
      // 曲线（墨迹渐变：旧段淡墨 → 新段朱砂，模拟墨迹书写）
      const grad = ctx.createLinearGradient(pad, 0, w - pad, 0)
      grad.addColorStop(0, 'rgba(43,38,32,0.3)')
      grad.addColorStop(0.65, 'rgba(179,68,44,0.65)')
      grad.addColorStop(1, '#b3442c')
      ctx.beginPath()
      for (let i = 0; i < losses.length; i++) {
        const x = pad + (i / (losses.length - 1)) * (w - 2 * pad)
        const y = yBot - ((losses[i] - min) / range) * (yBot - yTop)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.strokeStyle = grad
      ctx.lineWidth = 1.7
      ctx.stroke()
      // 当前数据点：青磷光光晕（科技反差）
      const lx = pad + ((losses.length - 1) / (losses.length - 1)) * (w - 2 * pad)
      const ly = yBot - ((losses[losses.length - 1] - min) / range) * (yBot - yTop)
      ctx.fillStyle = 'rgba(31,122,109,0.22)'
      ctx.beginPath(); ctx.arc(lx, ly, 5.5, 0, Math.PI * 2); ctx.fill()
      ctx.fillStyle = '#1f7a6d'
      ctx.beginPath(); ctx.arc(lx, ly, 2.2, 0, Math.PI * 2); ctx.fill()
      // 当前值标签
      ctx.fillStyle = '#b3442c'
      ctx.font = '13px serif'
      ctx.textAlign = 'right'
      const last = losses[losses.length - 1]
      ctx.fillText(`${last.toFixed(4)}`, w - 2, 14)
      ctx.textAlign = 'left'
      ctx.fillStyle = '#6b6357'
      ctx.fillText(`min ${min.toFixed(3)}`, pad, 14)
      ctx.font = '11px serif'
      ctx.fillText(`step ${losses.length}`, pad, h - 2)
    },
  }
  return api
}

// ---------- 权重热力图 ----------

export function createHeatmap(canvas) {
  let matrix = null
  let name = ''
  let hover = null // [r, c]

  const api = {
    set(m, n) { matrix = m; name = n },
    draw() {
      const { ctx, w, h } = fitCanvas(canvas)
      if (!matrix) {
        ctx.fillStyle = '#6b6357'; ctx.font = '12px serif'; ctx.textAlign = 'center'
        ctx.fillText(name || '训练后这里会显示权重', w / 2, h / 2)
        return
      }
      const rows = matrix.length
      const cols = matrix[0].length
      let min = Infinity, max = -Infinity
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const v = matrix[r][c]
        if (v < min) min = v
        if (v > max) max = v
      }
      const range = max - min || 1
      const cellW = w / cols
      const cellH = h / rows
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const t = (matrix[r][c] - min) / range
          ctx.fillStyle = warmColor(t)
          ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5)
        }
      }
      // 网格线
      ctx.strokeStyle = 'rgba(43,38,32,0.15)'
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let c = 0; c <= cols; c++) { ctx.moveTo(c * cellW, 0); ctx.lineTo(c * cellW, h) }
      for (let r = 0; r <= rows; r++) { ctx.moveTo(0, r * cellH); ctx.lineTo(w, r * cellH) }
      ctx.stroke()
      // 悬停显示精确值（可读性铁律）
      if (hover) {
        const [r, c] = hover
        ctx.fillStyle = 'rgba(43,38,32,0.85)'
        ctx.fillRect(6, 6, 150, 20)
        ctx.fillStyle = '#faf7f0'
        ctx.font = '12px monospace'
        ctx.textAlign = 'left'
        ctx.fillText(`${name}[${r}][${c}] = ${matrix[r][c].toFixed(4)}`, 10, 20)
      }
    },
    onHover(e) {
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      if (!matrix) { hover = null; return }
      const rows = matrix.length
      const cols = matrix[0].length
      const cellW = rect.width / cols
      const cellH = rect.height / rows
      const c = Math.floor(x / cellW)
      const r = Math.floor(y / cellH)
      hover = (r >= 0 && r < rows && c >= 0 && c < cols) ? [r, c] : null
    },
  }
  return api
}

// ---------- Attention 直播热力图 ----------

/**
 * Attention 直播：生成时逐 token 累积。
 * 行 = 生成的 token（文本），列 = 上下文窗口 token（固定列数，右对齐补 0）。
 * 默认累积 + 当前行（最后一行）高亮；悬停显示精确值。
 */
export function createAttnHeatmap(canvas, fixedCols = 16) {
  let colLabels = [] // 列标签（长度 = fixedCols 的窗口）
  let steps = []     // { rowText, dist }：生成 token 文本 + 注意力分布（长度 <= fixedCols）
  let hover = null

  const api = {
    setContext(tokens, cols) { colLabels = tokens.slice(); if (cols) fixedCols = cols },
    clear() { steps = [] },
    pushStep(rowText, dist) { steps.push({ rowText, dist }) },
    get length() { return steps.length },
    draw() {
      const { ctx, w, h } = fitCanvas(canvas)
      const leftPad = 22
      const botPad = 16
      const n = steps.length
      if (!n || !colLabels.length) {
        ctx.fillStyle = '#6b6357'; ctx.font = '12px serif'; ctx.textAlign = 'center'
        ctx.fillText('生成时，这里会实时显示模型在"看"哪些字', w / 2, h / 2)
        return
      }
      const cellW = (w - leftPad) / fixedCols
      const cellH = (h - botPad) / n
      // 动态范围（只统计实际分布值）
      let min = Infinity, max = -Infinity
      for (const st of steps) for (const v of st.dist) { if (v < min) min = v; if (v > max) max = v }
      const range = max - min || 1
      for (let r = 0; r < n; r++) {
        const dist = steps[r].dist
        for (let c = 0; c < fixedCols; c++) {
          const v = dist[c] ?? 0
          const t = (v - min) / range
          ctx.fillStyle = warmColor(t)
          ctx.fillRect(leftPad + c * cellW, r * cellH, cellW + 0.5, cellH + 0.5)
        }
      }
      // 网格 + 当前行高亮
      ctx.strokeStyle = 'rgba(43,38,32,0.18)'
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let c = 0; c <= fixedCols; c++) { ctx.moveTo(leftPad + c * cellW, 0); ctx.lineTo(leftPad + c * cellW, h - botPad) }
      for (let r = 0; r <= n; r++) { ctx.moveTo(leftPad, r * cellH); ctx.lineTo(w, r * cellH) }
      ctx.stroke()
      ctx.strokeStyle = '#b3442c'
      ctx.lineWidth = 2
      ctx.strokeRect(leftPad, (n - 1) * cellH, w - leftPad, cellH)
      // 行标签（生成 token 文本）
      ctx.fillStyle = '#2b2620'
      ctx.font = '10px serif'
      ctx.textAlign = 'right'
      for (let r = 0; r < n; r++) {
        ctx.fillText(steps[r].rowText, leftPad - 3, r * cellH + cellH / 2 + 3)
      }
      // 列标签（上下文 token，稀疏显示避免重叠）
      ctx.textAlign = 'center'
      ctx.fillStyle = '#6b6357'
      const skip = Math.max(1, Math.floor(cellW / 12))
      for (let c = 0; c < fixedCols; c += skip) {
        ctx.fillText(colLabels[c] ?? '', leftPad + c * cellW + cellW / 2, h - 4)
      }
      // 悬停显示精确值（可读性铁律）
      if (hover && hover.r < n && hover.c < fixedCols) {
        const dist = steps[hover.r].dist
        const v = dist[hover.c] ?? 0
        ctx.fillStyle = 'rgba(43,38,32,0.85)'
        ctx.fillRect(6, 6, 190, 20)
        ctx.fillStyle = '#faf7f0'
        ctx.font = '12px monospace'
        ctx.textAlign = 'left'
        ctx.fillText(`「${colLabels[hover.c] ?? ''}」 权重 = ${v.toFixed(3)}`, 10, 20)
      }
    },
    onHover(e) {
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      if (!steps.length || !colLabels.length) { hover = null; return }
      const leftPad = 22
      const botPad = 16
      const cellW = (rect.width - leftPad) / fixedCols
      const cellH = (rect.height - botPad) / steps.length
      hover = { r: Math.floor(y / cellH), c: Math.floor((x - leftPad) / cellW) }
    },
  }
  return api
}
