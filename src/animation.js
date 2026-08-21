// ============================================================
// src/animation.js
// 手算LM —— 教学动画：梯度下降（loss 碗 + 小球）+ 声波（频率→波形）
// ============================================================

/** 梯度下降动画：小球在 loss 碗（z=x²+y²）里滚向最低点。学习率=步长。 */
export function createGradientDescent(canvas) {
  let x = 1.5, y = 1.5
  let lr = 0.1
  let raf = null
  let running = false
  let path = [[x, y]]
  let over = false // 发散（训崩）标志

  function step() {
    const gx = 2 * x, gy = 2 * y
    x -= lr * gx
    y -= lr * gy
    path.push([x, y])
    if (path.length > 300) path.shift()
    if (Math.abs(x) > 50 || Math.abs(y) > 50) over = true // 训崩
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth || 320
    const h = canvas.clientHeight || 200
    canvas.width = w * dpr; canvas.height = h * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const scale = Math.min(w, h) / 5
    const cx = w / 2, cy = h / 2
    const toX = (v) => cx + v * scale
    const toY = (v) => cy - v * scale

    // 等高线（同心圆 = loss 等值线）
    for (let r = 0.25; r <= 2; r += 0.25) {
      ctx.beginPath()
      ctx.arc(cx, cy, r * scale, 0, Math.PI * 2)
      ctx.strokeStyle = `rgba(43,38,32,${0.06 + r * 0.045})`
      ctx.lineWidth = 1
      ctx.stroke()
    }
    // 底部中心标记
    ctx.fillStyle = 'rgba(43,38,32,0.25)'
    ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.fill()

    // 路径（青磷）
    if (path.length > 1) {
      ctx.beginPath()
      ctx.moveTo(toX(path[0][0]), toY(path[0][1]))
      for (let i = 1; i < path.length; i++) ctx.lineTo(toX(path[i][0]), toY(path[i][1]))
      ctx.strokeStyle = '#1f7a6d'
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
    // 小球（朱砂）
    const px = Math.max(-2.2, Math.min(2.2, x))
    const py = Math.max(-2.2, Math.min(2.2, y))
    ctx.beginPath()
    ctx.arc(toX(px), toY(py), 7, 0, Math.PI * 2)
    ctx.fillStyle = '#b3442c'
    ctx.fill()
    ctx.strokeStyle = '#fffdf6'
    ctx.lineWidth = 2
    ctx.stroke()

    ctx.fillStyle = '#6b6357'
    ctx.font = '11px serif'
    ctx.fillText(`loss = ${(x * x + y * y).toFixed(4)}`, 8, 16)
    ctx.fillText(`学习率 = ${lr}${over ? '（太大，训崩了！）' : ''}`, 8, 30)
    ctx.fillText('小球 = 参数 · 碗底 = 最优解', w - 168, h - 8)
  }

  function loop() {
    step()
    draw()
    if (running && !over) raf = requestAnimationFrame(loop)
    else if (over) { running = false }
  }

  return {
    setLr(v) { lr = v; over = false },
    reset() { x = 1.5; y = 1.5; path = [[x, y]]; over = false; running = false; if (raf) cancelAnimationFrame(raf); draw() },
    start() { if (!running) { running = true; loop() } },
    draw,
  }
}

/** 声波动画：正弦波，频率越高波越密。 */
export function createWave(canvas) {
  let freq = 440
  let phase = 0
  let raf = null
  function draw() {
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth || 320
    const h = canvas.clientHeight || 100
    canvas.width = w * dpr; canvas.height = h * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    ctx.strokeStyle = 'rgba(43,38,32,0.15)'
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke()
    const cycles = freq / 100
    ctx.beginPath()
    for (let px = 0; px <= w; px += 2) {
      const t = px / w
      const y = h / 2 - Math.sin(t * Math.PI * 2 * cycles + phase) * (h / 2 - 8)
      px === 0 ? ctx.moveTo(px, y) : ctx.lineTo(px, y)
    }
    ctx.strokeStyle = '#1f7a6d'
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.fillStyle = '#6b6357'
    ctx.font = '11px serif'
    ctx.fillText(`${freq} Hz · ${cycles.toFixed(1)} 个完整波`, 8, 14)
  }
  function loop() { phase += 0.06; draw(); if (raf) raf = requestAnimationFrame(loop) }
  return {
    setFreq(v) { freq = v; draw() },
    start() { if (!raf) raf = requestAnimationFrame(loop) },
    stop() { if (raf) { cancelAnimationFrame(raf); raf = null } },
    draw,
  }
}