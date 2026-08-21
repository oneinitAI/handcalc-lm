// ============================================================
// src/matrix.js
// 手算LM —— 纯 JS 矩阵运算库（零依赖）
// 教学优先：清晰可读 > 极致性能。所有矩阵用二维数组 number[][]。
// ============================================================

// ---------- 创建 ----------

/** 全零矩阵 [rows][cols] */
export function zeros(rows, cols) {
  const m = new Array(rows)
  for (let r = 0; r < rows; r++) m[r] = new Array(cols).fill(0)
  return m
}

/** 全 1 矩阵 */
export function ones(rows, cols) {
  const m = new Array(rows)
  for (let r = 0; r < rows; r++) m[r] = new Array(cols).fill(1)
  return m
}

/**
 * 高斯随机矩阵（Box-Muller），可种子化。
 * 返回 { value, nextSeed } 便于链式播种（教学里每个参数单独种子可复现）。
 */
export function randn(rows, cols, seed = 1337) {
  const m = new Array(rows)
  let s = seed
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
  for (let r = 0; r < rows; r++) {
    m[r] = new Array(cols)
    for (let c = 0; c < cols; c++) {
      // Box-Muller（两个均匀 → 一个高斯）
      let u1 = 0, u2 = 0
      while (u1 === 0) u1 = rnd()
      u2 = rnd()
      m[r][c] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
    }
  }
  return { value: m, nextSeed: s }
}

// ---------- 基本运算 ----------

/** 矩阵乘法 C[r][c] = Σ_k A[r][k]·B[k][c] */
export function matmul(A, B) {
  const rows = A.length
  const k = A[0].length
  const cols = B[0].length
  const C = new Array(rows)
  for (let r = 0; r < rows; r++) {
    const Ar = A[r]
    const Cr = new Array(cols)
    for (let c = 0; c < cols; c++) {
      let acc = 0
      for (let i = 0; i < k; i++) acc += Ar[i] * B[i][c]
      Cr[c] = acc
    }
    C[r] = Cr
  }
  return C
}

/** 矩阵 × 向量：A[rows][k] · v[k] → out[rows] */
export function matvec(A, v) {
  const rows = A.length
  const k = A[0].length
  const out = new Array(rows)
  for (let r = 0; r < rows; r++) {
    let acc = 0
    const Ar = A[r]
    for (let i = 0; i < k; i++) acc += Ar[i] * v[i]
    out[r] = acc
  }
  return out
}

/** 转置 */
export function transpose(M) {
  const rows = M.length
  const cols = M[0].length
  const T = new Array(cols)
  for (let c = 0; c < cols; c++) {
    T[c] = new Array(rows)
    for (let r = 0; r < rows; r++) T[c][r] = M[r][c]
  }
  return T
}

/** 逐元素加（同形状） */
export function add(A, B) {
  const rows = A.length
  const cols = A[0].length
  const C = new Array(rows)
  for (let r = 0; r < rows; r++) {
    C[r] = new Array(cols)
    for (let c = 0; c < cols; c++) C[r][c] = A[r][c] + B[r][c]
  }
  return C
}

/** 逐元素减 */
export function sub(A, B) {
  const rows = A.length
  const cols = A[0].length
  const C = new Array(rows)
  for (let r = 0; r < rows; r++) {
    C[r] = new Array(cols)
    for (let c = 0; c < cols; c++) C[r][c] = A[r][c] - B[r][c]
  }
  return C
}

/** 矩阵缩放 */
export function scale(M, s) {
  const rows = M.length
  const C = new Array(rows)
  for (let r = 0; r < rows; r++) {
    const Mr = M[r]
    const Cr = new Array(Mr.length)
    for (let c = 0; c < Mr.length; c++) Cr[c] = Mr[c] * s
    C[r] = Cr
  }
  return C
}

/** 逐元素乘（Hadamard） */
export function mul(A, B) {
  const rows = A.length
  const cols = A[0].length
  const C = new Array(rows)
  for (let r = 0; r < rows; r++) {
    C[r] = new Array(cols)
    for (let c = 0; c < cols; c++) C[r][c] = A[r][c] * B[r][c]
  }
  return C
}

/** 逐元素求平均（用于数值梯度检查） */
export function meanAbs(M) {
  let sum = 0, n = 0
  for (const row of M) for (const v of row) { sum += Math.abs(v); n++ }
  return sum / n
}

// ---------- 概率与损失 ----------

/**
 * 行 softmax：输入 [n][vocab]，输出同形状，每行和为 1（数值稳定版）。
 */
export function softmaxRows(M) {
  const n = M.length
  const out = new Array(n)
  for (let i = 0; i < n; i++) {
    const row = M[i]
    let mx = -Infinity
    for (const v of row) if (v > mx) mx = v
    let sum = 0
    const e = new Array(row.length)
    for (let j = 0; j < row.length; j++) {
      e[j] = Math.exp(row[j] - mx)
      sum += e[j]
    }
    out[i] = new Array(row.length)
    for (let j = 0; j < row.length; j++) out[i][j] = e[j] / sum
  }
  return out
}

/**
 * 交叉熵（平均）：logits [n][vocab]，targets [n]（整数索引）。
 * 返回 { loss, probs, dlogits } —— probs 供可视化，dlogits 供反向传播。
 * 教学点：dlogits = (probs - onehot(targets)) / n，这是"预测与真实之差"。
 */
export function crossEntropy(logits, targets) {
  const n = logits.length
  const vocab = logits[0].length
  const probs = softmaxRows(logits)
  let loss = 0
  const dlogits = new Array(n)
  for (let i = 0; i < n; i++) {
    const t = targets[i]
    loss += -Math.log(Math.max(probs[i][t], 1e-12))
    dlogits[i] = new Array(vocab)
    for (let j = 0; j < vocab; j++) {
      dlogits[i][j] = (probs[i][j] - (j === t ? 1 : 0)) / n
    }
  }
  return { loss: loss / n, probs, dlogits }
}

// ---------- 数值梯度检查辅助 ----------

/**
 * 数值梯度：对参数矩阵 M 的每个元素，扰动 h，测损失变化率。
 * f 是一个"无参闭包"：调用 f() 返回当前损失（模型已更新到该参数值）。
 * 返回与 M 同形状的数值梯度矩阵。
 */
export function numericalGradient(flatFn, M, h = 1e-6) {
  const rows = M.length
  const cols = M[0].length
  const grad = new Array(rows)
  for (let r = 0; r < rows; r++) {
    grad[r] = new Array(cols)
    for (let c = 0; c < cols; c++) {
      const orig = M[r][c]
      M[r][c] = orig + h
      const lp = flatFn()
      M[r][c] = orig - h
      const lm = flatFn()
      M[r][c] = orig
      grad[r][c] = (lp - lm) / (2 * h)
    }
  }
  return grad
}
