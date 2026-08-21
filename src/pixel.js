// ============================================================
// src/pixel.js
// 手算LM —— 图像演示：像素即序列
// 图案展平成 0/1 像素序列，用与文本完全相同的 Transformer 训练"猜下一个像素"
// ============================================================

export const PIXEL_PATTERNS = [
  {
    id: 'moon',
    name: '月亮',
    grid: [
      '........',
      '....####',
      '..######',
      '..###...',
      '..######',
      '....####',
      '........',
      '........',
    ],
  },
  {
    id: 'heart',
    name: '心形',
    grid: [
      '.##..##.',
      '########',
      '########',
      '########',
      '.######.',
      '..####..',
      '...##...',
      '........',
    ],
  },
  {
    id: 'letterH',
    name: '字母 H',
    grid: [
      '##....##',
      '##....##',
      '##....##',
      '########',
      '########',
      '##....##',
      '##....##',
      '##....##',
    ],
  },
  {
    id: 'wave',
    name: '波浪',
    grid: [
      '...##...',
      '..####..',
      '.##..##.',
      '##....##',
      '.##..##.',
      '..####..',
      '...##...',
      '........',
    ],
  },
]

/** 图案 → 0/1 像素序列（逐行展开） */
export function gridToSeq(grid) {
  return grid.join('').split('').map((c) => (c === '#' ? 1 : 0))
}

/** 0/1 序列 → 图案 grid */
export function seqToGrid(seq, side = 8) {
  const grid = []
  for (let r = 0; r < side; r++) {
    let row = ''
    for (let c = 0; c < side; c++) row += seq[r * side + c] ? '#' : '.'
    grid.push(row)
  }
  return grid
}

/** 在 canvas 上渲染像素网格（放大显示） */
export function renderGrid(canvas, grid) {
  const side = grid.length
  const dpr = window.devicePixelRatio || 1
  const size = canvas.clientWidth || 160
  canvas.width = size * dpr
  canvas.height = size * dpr
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, size, size)
  const cell = size / side
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      ctx.fillStyle = grid[r][c] === '#' ? 'var(--ink, #2b2620)' : 'rgba(43,38,32,0.06)'
      ctx.fillRect(c * cell, r * cell, cell - 1, cell - 1)
    }
  }
  // 网格线
  ctx.strokeStyle = 'rgba(43,38,32,0.12)'
  ctx.lineWidth = 1
  ctx.beginPath()
  for (let i = 0; i <= side; i++) {
    ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, size)
    ctx.moveTo(0, i * cell); ctx.lineTo(size, i * cell)
  }
  ctx.stroke()
}