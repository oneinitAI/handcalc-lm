// ============================================================
// src/mm.js —— 多模态：一个 Transformer 通吃 文本+像素+音高
// ============================================================

export function initMultiModal(deps) {
  const { $, createModel, createOptimizer, trainStep, sample, CORPUS, PIXEL_PATTERNS, MELODIES, gridToSeq, seqToGrid, renderGrid, parseMelody, playMelody, renderMelody, state } = deps

  const MM_TXT = CORPUS[0].text.slice(0, 400) // 荷塘月色前 400 字
  const MM_IMG = PIXEL_PATTERNS.slice(0, 3)   // 圆 / 心形 / H
  const MM_AUD = MELODIES.slice(0, 2)         // 小星星 / 欢乐颂
  const M_IMG = 0, M_AUD = 1, M_TXT = 2
  const imgBase = 3, audBase = 3 + 16, txtBase = 3 + 16 + 8

  function buildMultiModal() {
    const txtChars = [...new Set(MM_TXT.split(''))].sort()
    const itos = ['<图>', '<音>', '<文>', ...'0123456789abcdef'.split(''), ...'01234567'.split(''), ...txtChars]
    const stoi = Object.fromEntries(itos.map((c, i) => [c, i]))
    const vocab = itos.length
    const cfg = { vocab_size: vocab, block_size: 48, n_layer: 1, n_head: 1, n_embd: 16, bias: true }
    const { params } = createModel(cfg, 42)
    const seq = []
    MM_IMG.forEach((p) => { seq.push(M_IMG); seq.push(...gridToSeq(p.grid).map((v) => imgBase + v)) })
    MM_AUD.forEach((m) => { seq.push(M_AUD); seq.push(...parseMelody(m.seq).map((v) => audBase + v)) })
    seq.push(M_TXT)
    seq.push(...MM_TXT.split('').map((c) => stoi[c]))
    state.mm = { params, cfg, opt: createOptimizer(params, { type: 'adam', lr: 0.01 }), seq, itos, stoi, imgBase, audBase, txtBase }
    // 统一序列轨道（教学：三种模态的 token 数字）
    $('mmTxtTrack').textContent = '月光如流水 → ' + '月光如流水'.split('').map((c) => stoi[c]).join(' ')
    $('mmImgTrack').textContent = '圆 → ' + gridToSeq(MM_IMG[0].grid).slice(0, 16).join(' ')
    $('mmAudTrack').textContent = '小星星 → ' + parseMelody(MM_AUD[0].seq).slice(0, 16).join(' ')
    $('mmInfo').textContent = `混合模型就绪：${vocab} 个 token（文本+像素+音高）· 训练序列 ${seq.length}`
  }

  function trainMM() {
    const m = state.mm
    const L = m.seq.length
    let done = 0
    const total = 3000
    const loop = () => {
      const k = 100
      for (let i = 0; i < k && done < total; i++, done++) {
        const start = Math.floor(Math.random() * (L - m.cfg.block_size))
        trainStep(m.params, m.seq.slice(start, start + m.cfg.block_size), m.seq.slice(start + 1, start + m.cfg.block_size + 1), m.cfg, m.opt)
      }
      $('mmInfo').textContent = `混合训练中… ${done}/${total}`
      if (done < total) requestAnimationFrame(loop)
      else {
        $('mmTxtBtn').disabled = false; $('mmImgBtn').disabled = false; $('mmAudBtn').disabled = false
        $('mmInfo').textContent = '完成！同一个模型现在会续写文本、画图、作曲——点三个按钮试试'
      }
    }
    loop()
  }

  function genMMTxt() {
    const m = state.mm
    const prompt = [M_TXT, ...'月光'.split('').map((c) => m.stoi[c] ?? m.txtBase)]
    const gen = sample(m.params, prompt, 40, m.cfg, { temperature: 0.6 })
    const txt = gen.slice(prompt.length).filter((v) => v >= m.txtBase).map((v) => m.itos[v]).join('')
    $('mmTxtOut').textContent = '月光' + txt
    $('mmInfo').textContent = '文本模态：同一个模型续写了荷塘月色风格的文本'
  }

  function genMMImg() {
    const m = state.mm
    const gen = sample(m.params, [M_IMG], 260, m.cfg, { temperature: 0.05 })
    const pix = gen.slice(1).filter((v) => v >= m.imgBase && v < m.imgBase + 16).map((v) => v - m.imgBase).slice(0, 256)
    renderGrid($('mmImgOut'), seqToGrid(pix))
    $('mmInfo').textContent = '图像模态：同一个模型画出了一张图'
  }

  function genMMAUD() {
    const m = state.mm
    const gen = sample(m.params, [M_AUD], 40, m.cfg, { temperature: 0.4 })
    const mel = gen.slice(1).filter((v) => v >= m.audBase && v < m.audBase + 8).map((v) => v - m.audBase).slice(0, 48)
    renderMelody($('mmAudOut'), mel)
    playMelody(mel)
    $('mmInfo').textContent = '音频模态：同一个模型作曲并播放'
  }

  $('mmTrainBtn').addEventListener('click', () => { if (!state.mm) buildMultiModal(); trainMM() })
  $('mmTxtBtn').addEventListener('click', genMMTxt)
  $('mmImgBtn').addEventListener('click', genMMImg)
  $('mmAudBtn').addEventListener('click', genMMAUD)
  buildMultiModal()

  // 跨模态转换
  const mmXImg = $('mmXImg')
  PIXEL_PATTERNS.forEach((p) => mmXImg.insertAdjacentHTML('beforeend', `<option value="${p.id}">${p.name}</option>`))
  $('mmSingBtn').addEventListener('click', () => {
    const p = PIXEL_PATTERNS.find((x) => x.id === mmXImg.value)
    if (!p) return
    const pix = gridToSeq(p.grid)
    const mel = pix.filter((v) => v > 0).map((v) => 1 + Math.floor((v / 16) * 7))
    playMelody(mel.slice(0, 40), 200)
    $('mmXOut').textContent = `「${p.name}」在唱歌：${pix.filter(Boolean).length} 个亮像素 → 音高序列播放中（像素=音高）`
  })
  $('mmToTextBtn').addEventListener('click', () => {
    const p = PIXEL_PATTERNS.find((x) => x.id === mmXImg.value)
    if (!p) return
    const txt = gridToSeq(p.grid).map((v) => (v < 10 ? String(v) : String.fromCharCode(87 + v))).join('')
    $('mmXOut').textContent = `「${p.name}」的文本形态（16×16=256 个像素字符，图=一串字）：\n${txt}`
  })
}