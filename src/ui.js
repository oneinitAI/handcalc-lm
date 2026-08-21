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
      // 曲线（墨色到朱砂：用朱砂）
      ctx.beginPath()
      for (let i = 0; i < losses.length; i++) {
        const x = pad + (i / (losses.length - 1)) * (w - 2 * pad)
        const y = yBot - ((losses[i] - min) / range) * (yBot - yTop)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.strokeStyle = '#b3442c'
      ctx.lineWidth = 1.6
      ctx.stroke()
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
