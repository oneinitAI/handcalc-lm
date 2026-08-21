// ============================================================
// src/dict.js —— 动画词典引擎：列表视图 + 术语详情页（可跳转）
// ============================================================

import { DICT_ITEMS } from './dict-data.js'

let _root = null

/** 驱动单个画布的动画循环 */
function startAnim(canvas, draw) {
  let last = performance.now()
  let t = 0
  function loop() {
    const now = performance.now()
    t += (now - last) / 1000
    last = now
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth || 300
    const h = canvas.clientHeight || 140
    if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr }
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    try { draw(ctx, w, h, t) } catch (e) { /* 忽略 */ }
    requestAnimationFrame(loop)
  }
  requestAnimationFrame(loop)
}

/** 词典：卡片墙 ↔ 单个术语详情页（点击跳转） */
export function initDict(root) {
  _root = root
  showList()
}

function showList() {
  _root.innerHTML = `<div class="dict-grid">${DICT_ITEMS.map((it, i) => `
      <div class="dict-item card" data-i="${i}" title="点击查看详情页">
        <div class="dict-term">${it.term} <span class="tag">${it.cat}</span></div>
        <div class="dict-one">${it.one}</div>
        <canvas class="dict-canvas" data-i="${i}"></canvas>
        <div class="dict-hint">点击进入详情 →</div>
      </div>`).join('')}</div>`
  _root.querySelectorAll('.dict-canvas').forEach((cv) => {
    startAnim(cv, DICT_ITEMS[+cv.dataset.i].draw)
  })
  _root.querySelectorAll('.dict-item').forEach((card) => {
    card.addEventListener('click', () => showDetail(+card.dataset.i))
    card.style.cursor = 'pointer'
  })
}

function showDetail(i) {
  const it = DICT_ITEMS[i]
  const d = it.detail
  _root.innerHTML = `
      <button class="btn ghost" id="dictBack">← 返回词典</button>
      <div class="card">
        <div class="dict-term">${it.term} <span class="tag">${it.cat}</span></div>
        <div class="dict-one">${it.one}</div>
        <canvas class="dict-canvas" id="dictDetailCanvas"></canvas>
        <div class="teach">
          <h3>它是什么</h3><p>${d.what}</p>
          <h3>为什么重要</h3><p>${d.why}</p>
          <h3>工作流程</h3><p>${d.how}</p>
          <h3>例子</h3><p>${d.example}</p>
          <h3>在本项目中的体现</h3><p>${d.inProject}</p>
        </div>
      </div>`
  startAnim(_root.querySelector('#dictDetailCanvas'), it.draw)
  _root.querySelector('#dictBack').addEventListener('click', showList)
  _root.querySelector('#dictBack').scrollIntoView({ block: 'start' })
}

/**
 * 从任意板块跳到词典并打开某个术语（名称模糊匹配，或直接给索引）
 * 供 Transformer 板块等处的"去词典看动画"链接使用
 */
export function dictOpenDetail(term) {
  // 切到词典 tab
  document.querySelectorAll('.tab-btn').forEach((x) => x.classList.remove('on'))
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('on'))
  const btn = document.querySelector('.tab-btn[data-tab="dict"]')
  if (btn) btn.classList.add('on')
  const panel = document.getElementById('tab-dict')
  if (panel) panel.classList.add('on')
  // 移动端收起抽屉
  document.querySelectorAll('.tabs, .nav-toggle').forEach((el) => el.classList.remove('open'))
  // 找术语
  let i = typeof term === 'number' ? term : DICT_ITEMS.findIndex((d) => d.term.toLowerCase().includes(String(term).toLowerCase()))
  if (i < 0) i = 0
  showDetail(i)
}