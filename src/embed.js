// ============================================================
// src/embed.js —— 底层原理：词向量语义空间算法
// PCA 2D 投影 + 余弦相似度：展示"训练让语义相近的字向量相近"
// ============================================================

function dot(a, b) { return a.reduce((s, x, i) => s + x * b[i], 0) }

/** 余弦相似度（1=同向，0=无关，-1=反向） */
export function cosineSim(a, b) {
  let d = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
  return d / (Math.sqrt(na) * Math.sqrt(nb) || 1)
}

/** 最相似的词对（降序） */
export function topSimilarPairs(wte, k = 8, min = 0.1) {
  const pairs = []
  for (let i = 0; i < wte.length; i++) {
    for (let j = i + 1; j < wte.length; j++) {
      const s = cosineSim(wte[i], wte[j])
      if (s > min) pairs.push({ i, j, s })
    }
  }
  pairs.sort((a, b) => b.s - a.s)
  return pairs.slice(0, k)
}

function powerIterate(M, iters = 80) {
  let v = new Array(M.length).fill(1)
  for (let i = 0; i < iters; i++) {
    const mv = M.map((row) => row.reduce((s, x, j) => s + x * v[j], 0))
    const norm = Math.sqrt(mv.reduce((s, x) => s + x * x, 0)) || 1
    v = mv.map((x) => x / norm)
  }
  return v
}

function deflate(M, e) {
  const Me = M.map((row) => row.reduce((s, x, j) => s + x * e[j], 0))
  const lr = e.reduce((s, x, i) => s + x * Me[i], 0)
  return M.map((row, i) => row.map((x, j) => x - lr * e[i] * e[j]))
}

/** PCA 降到 2D：points [n][d] → [[x,y], ...] */
export function pca2d(points) {
  const n = points.length
  const d = points[0].length
  const mean = new Array(d).fill(0)
  for (const p of points) for (let i = 0; i < d; i++) mean[i] += p[i] / n
  const centered = points.map((p) => p.map((v, i) => v - mean[i]))
  const C = Array.from({ length: d }, () => new Array(d).fill(0))
  for (const p of centered) {
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += (p[i] * p[j]) / n
  }
  const e1 = powerIterate(C)
  const e2 = powerIterate(deflate(C, e1))
  return centered.map((p) => [dot(p, e1), dot(p, e2)])
}