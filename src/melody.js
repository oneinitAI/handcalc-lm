// ============================================================
// src/melody.js
// AI 学习本 —— 语音模型（旋律序列，完整版）
// 旋律 = 一串音高 token（简谱 1-7 + 0 休止）。训练"猜下一个音"，
// 学会后模型能续写旋律并演奏——和文字模型同一架构。
// ============================================================

export const MELODIES = [
  { id: 'twinkle', name: '小星星', seq: '115566544332215544332554433211556654433221' },
  { id: 'ode', name: '欢乐颂', seq: '334554321123322334554321123211' },
  { id: 'tiger', name: '两只老虎', seq: '123112311345345565431565431251251' },
  { id: 'london', name: '伦敦桥', seq: '565434345234565434251' },
]

/** 简谱字符串 → 音高 token 数组（0=休止，1-7=音阶） */
export function parseMelody(s) {
  return s.split('').map((c) => (c >= '1' && c <= '7' ? +c : 0))
}

/** 音高 → 频率（C 大调自然音阶，C4 基准） */
export function noteToFreq(n) {
  if (n <= 0) return 0
  const scale = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88] // C4 D4 E4 F4 G4 A4 B4
  return scale[(n - 1) % 7]
}

/** 单例 AudioContext */
let _ctx = null
function audioCtx() {
  if (!_ctx) _ctx = new (window.AudioContext || window.webkitAudioContext)()
  if (_ctx.state === 'suspended') _ctx.resume()
  return _ctx
}

/** 播放音高序列（Web Audio，每音一个振荡器） */
export function playMelody(seq, bpm = 220, onStep = null) {
  const ctx = audioCtx()
  const dur = 60 / bpm
  const t0 = ctx.currentTime + 0.05
  seq.forEach((n, i) => {
    const freq = noteToFreq(n)
    if (freq > 0) {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.type = 'triangle'
      osc.frequency.value = freq
      gain.gain.setValueAtTime(0.14, t0 + i * dur)
      gain.gain.exponentialRampToValueAtTime(0.001, t0 + (i + 1) * dur)
      osc.connect(gain).connect(ctx.destination)
      osc.start(t0 + i * dur)
      osc.stop(t0 + (i + 1) * dur)
    }
    if (onStep) onStep(i)
  })
}

/** 播放单个音调（频率滑杆动手用） */
export function playTone(freq, dur = 0.8) {
  const ctx = audioCtx()
  const osc = ctx.createOscillator()
  const gain = ctx.createGain()
  osc.type = 'sine'
  osc.frequency.value = freq
  gain.gain.setValueAtTime(0.15, ctx.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur)
  osc.connect(gain).connect(ctx.destination)
  osc.start()
  osc.stop(ctx.currentTime + dur)
}

/** 音高阶梯可视化：每音一柱，高度=音高，0=底线 */
export function renderMelody(canvas, seq) {
  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth || 300
  const h = canvas.clientHeight || 120
  canvas.width = w * dpr
  canvas.height = h * dpr
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  const n = seq.length
  if (!n) return
  const slot = w / n
  const base = h - 10
  for (let i = 0; i < n; i++) {
    const v = seq[i]
    if (v > 0) {
      const bh = (v / 7) * (h - 30)
      ctx.fillStyle = i % 2 ? 'var(--accent, #b3442c)' : '#d66a4a'
      ctx.fillRect(i * slot + 1, base - bh, slot - 2, bh)
    } else {
      ctx.fillStyle = 'rgba(43,38,32,0.12)'
      ctx.fillRect(i * slot + 1, base - 3, slot - 2, 3)
    }
  }
  ctx.strokeStyle = 'rgba(43,38,32,0.15)'
  ctx.beginPath()
  ctx.moveTo(0, base); ctx.lineTo(w, base)
  ctx.stroke()
}