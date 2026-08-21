// ============================================================
// src/particles.js —— 背景"数字生命"粒子（大胆层）
// 数字/汉字在深空底上缓缓飘落，青磷/朱砂双色
// ============================================================

export function initParticles(canvas) {
  const dpr = window.devicePixelRatio || 1
  let w = 0, h = 0, ctx = null
  const chars = '0123456789abcdefAI 学习本月光荷塘注意力模型token梯度loss数据曲线像素音高猜字魔法'.split('')
  const parts = []
  const N = 80

  function resize() {
    w = window.innerWidth
    h = window.innerHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  function make() {
    return {
      x: Math.random() * w,
      y: Math.random() * h,
      v: 0.15 + Math.random() * 0.7,
      s: 9 + Math.random() * 14,
      c: Math.random() < 0.5 ? '31,122,109' : '179,68,44', // 青磷 / 朱砂（演草纸淡色）
      ch: chars[Math.floor(Math.random() * chars.length)],
      a: 0.04 + Math.random() * 0.12,
      drift: (Math.random() - 0.5) * 0.3,
    }
  }

  function loop() {
    ctx.clearRect(0, 0, w, h)
    for (const p of parts) {
      p.y -= p.v
      p.x += p.drift
      if (p.y < -20 || p.x < -20 || p.x > w + 20) {
        Object.assign(p, make())
        p.y = h + 20
      }
      ctx.fillStyle = `rgba(${p.c},${p.a})`
      ctx.font = `${p.s}px serif`
      ctx.fillText(p.ch, p.x, p.y)
    }
    requestAnimationFrame(loop)
  }

  resize()
  for (let i = 0; i < N; i++) parts.push(make())
  window.addEventListener('resize', resize)
  requestAnimationFrame(loop)
}