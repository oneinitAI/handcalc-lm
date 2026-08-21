// ============================================================
// src/pixel.js
// 手算LM —— 图像模型（完整版）：16×16 灰度像素序列
// 灰度 0(白)~15(黑)，用字符 '0'-'9','a'-'f' 表示
// 图案由代码生成；支持用户在自己的画布上绘制训练数据
// ============================================================

const SIDE = 16
const MAX = 15

// ---- 字符 ↔ 灰度值 ----
export function charToVal(c) {
  if (c >= '0' && c <= '9') return +c
  return c.charCodeAt(0) - 87 // a=10 ... f=15
}
export function valToChar(v) {
  return v < 10 ? String(v) : String.fromCharCode(87 + v)
}

// ---- 代码生成图案（避免手写 16 行字符串）----
function makeGrid(fn) {
  const g = []
  for (let r = 0; r < SIDE; r++) {
    let row = ''
    for (let c = 0; c < SIDE; c++) row += fn(r, c)
    g.push(row)
  }
  return g
}

export const PIXEL_PATTERNS = [
  {
    id: 'circle',
    name: '圆',
    grid: makeGrid((r, c) => {
      const d = (r - 7.5) ** 2 + (c - 7.5) ** 2
      return d <= 42 ? 'f' : '0'
    }),
  },
  {
    id: 'ring',
    name: '圆环',
    grid: makeGrid((r, c) => {
      const d = (r - 7.5) ** 2 + (c - 7.5) ** 2
      return d >= 20 && d <= 45 ? 'f' : '0'
    }),
  },
  {
    id: 'heart',
    name: '心形',
    grid: makeGrid((r, c) => {
      const d1 = (r - 5) ** 2 + (c - 5.5) ** 2
      const d2 = (r - 5) ** 2 + (c - 10.5) ** 2
      const tri = r >= 6 && c >= 4 && c <= 12 && r <= (c - 4) + 6 && r <= (12 - c) + 6
      return d1 <= 16 || d2 <= 16 || (tri && r >= 6) ? 'f' : '0'
    }),
  },
  {
    id: 'letterH',
    name: '字母 H',
    grid: makeGrid((r, c) => {
      const bar = r >= 6 && r <= 9
      const left = c >= 2 && c <= 4
      const right = c >= 11 && c <= 13
      return left || right || (bar && c >= 5 && c <= 10) ? 'f' : '0'
    }),
  },
  {
    id: 'diag',
    name: '对角线',
    grid: makeGrid((r, c) => (Math.abs(c - r) <= 1 ? 'f' : '0')),
  },
  {
    id: 'grad',
    name: '渐变',
    grid: makeGrid((r) => {
      let row = ''
      for (let c = 0; c < SIDE; c++) row += valToChar(Math.round((c / (SIDE - 1)) * MAX))
      return row
    }),
  },
]

// ---- 序列转换 ----
export function gridToSeq(grid) {
  return grid.join('').split('').map(charToVal)
}
export function seqToGrid(seq, side = SIDE) {
  const g = []
  for (let r = 0; r < side; r++) {
    let row = ''
    for (let c = 0; c < side; c++) row += valToChar(seq[r * side + c] ?? 0)
    g.push(row)
  }
  return g
}

// ---- 渲染（灰度：白纸 → 墨色）----
export function renderGrid(canvas, grid) {
  const side = grid.length
  const dpr = window.devicePixelRatio || 1
  const size = canvas.clientWidth || 180
  canvas.width = size * dpr
  canvas.height = size * dpr
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, size, size)
  const cell = size / side
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      const t = charToVal(grid[r][c]) / MAX
      const bg = [250, 247, 240]
      const ink = [43, 38, 32]
      const v = ink.map((x, i) => Math.round(bg[i] + (x - bg[i]) * t))
      ctx.fillStyle = `rgb(${v[0]},${v[1]},${v[2]})`
      ctx.fillRect(c * cell, r * cell, cell - 0.5, cell - 0.5)
    }
  }
  ctx.strokeStyle = 'rgba(43,38,32,0.1)'
  ctx.lineWidth = 0.5
  ctx.beginPath()
  for (let i = 0; i <= side; i++) {
    ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size)
    ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell)
  }
  ctx.stroke()
}

// ---- 用户绘制：把一块 canvas 变成绘制板（笔刷灰度 MAX），输出 grid ----
export function attachDrawing(canvas, onGrid) {
  const size = canvas.clientWidth || 180
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = '#faf7f0'
  ctx.fillRect(0, 0, size, size)
  let drawing = false
  const cell = size / SIDE
  const grid = Array.from({ length: SIDE }, () => Array(SIDE).fill(0))

  function paint(x, y, v = MAX) {
    const c = Math.floor(x / cell)
    const r = Math.floor(y / cell)
    if (r < 0 || r >= SIDE || c < 0 || c >= SIDE) return
    grid[r][c] = v
    const t = v / MAX
    const bg = [250, 247, 240]
    const ink = [43, 38, 32]
    const col = ink.map((z, i) => Math.round(bg[i] + (z - bg[i]) * t))
    ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`
    ctx.fillRect(c * cell + 0.5, r * cell + 0.5, cell - 1, cell - 1)
  }

  canvas.addEventListener('mousedown', (e) => {
    drawing = true
    const rect = canvas.getBoundingClientRect()
    paint(e.clientX - rect.left, e.clientY - rect.top)
    onGrid(gridToSeq(grid))
  })
  canvas.addEventListener('mousemove', (e) => {
    if (!drawing) return
    const rect = canvas.getBoundingClientRect()
    paint(e.clientX - rect.left, e.clientY - rect.top)
    onGrid(gridToSeq(grid))
  })
  canvas.addEventListener('mouseup', () => { drawing = false })
  canvas.addEventListener('mouseleave', () => { drawing = false })

  return {
    clear() {
      for (let r = 0; r < SIDE; r++) grid[r].fill(0)
      ctx.fillStyle = '#faf7f0'
      ctx.fillRect(0, 0, size, size)
      onGrid(gridToSeq(grid))
    },
    getSeq: () => gridToSeq(grid),
  }
}