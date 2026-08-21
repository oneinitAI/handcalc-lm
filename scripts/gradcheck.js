// ============================================================
// scripts/gradcheck.js
// 手算LM —— 数值梯度 vs 解析梯度对比，验证 backward 正确性
// 运行：node scripts/gradcheck.js
// ============================================================

import { createModel, forward, zeroGrad, defaultCfg } from '../src/model.js'
import { backward } from '../src/backward.js'
import { crossEntropy, numericalGradient } from '../src/matrix.js'

const cfg = { ...defaultCfg }
const { params } = createModel(cfg, 42)

// 固定输入（小序列）
const T = 6
const idx = Array.from({ length: T }, (_, i) => (i * 7 + 3) % cfg.vocab_size)
const targets = idx.slice(1).concat([5])

// ---- 解析梯度 ----
const { logits, cache } = forward(params, idx, targets, cfg)
const ce = crossEntropy(logits, targets)
console.log(`loss = ${ce.loss.toFixed(6)}`)
zeroGrad(params)
backward(params, cache, ce.dlogits, cfg)

// ---- 数值梯度（每次扰动后重新 forward 得 loss）----
const lossFn = () => forward(params, idx, targets, cfg).loss

let worst = 0
let worstAbs = 0
let worstKey = ''
for (const key of Object.keys(params)) {
  const M = params[key].value
  const ng = numericalGradient(lossFn, M, 1e-5)
  const ag = params[key].grad
  const rows = M.length
  const cols = M[0].length
  let maxRel = 0
  let maxAbs = 0
  let worstPos = null
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const denom = Math.abs(ag[r][c]) + Math.abs(ng[r][c])
      const rel = denom === 0 ? 0 : Math.abs(ag[r][c] - ng[r][c]) / denom
      const absErr = Math.abs(ag[r][c] - ng[r][c])
      if (absErr > maxAbs) { maxAbs = absErr; worstPos = [r, c, ag[r][c], ng[r][c]] }
      if (rel > maxRel) maxRel = rel
    }
  }
  if (maxRel > worst) worst = maxRel
  if (maxAbs > worstAbs) { worstAbs = maxAbs; worstKey = key }
  const flag = maxAbs > 1e-4 ? ' ⚠️' : ''
  console.log(`${key.padEnd(24)} maxRelErr=${maxRel.toExponential(2)}  maxAbsErr=${maxAbs.toExponential(2)}${flag}` +
    (flag ? `   @[${worstPos[0]},${worstPos[1]}] ana=${worstPos[2].toFixed(4)} num=${worstPos[3].toFixed(4)}` : ''))
}

console.log(`\nWORST: ${worstKey} maxAbsErr=${worstAbs.toExponential(2)}`)
console.log(worstAbs < 1e-4 ? '✅ 梯度检查通过（backward 正确）' : '❌ 梯度检查失败')
