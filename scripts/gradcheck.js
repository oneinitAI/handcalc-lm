// ============================================================
// scripts/gradcheck.js
// 手算LM —— 数值梯度 vs 解析梯度对比（多配置验证 backward 正确性）
// 运行：node scripts/gradcheck.js
// ============================================================

import { createModel, forward, zeroGrad, paramCount } from '../src/model.js'
import { backward } from '../src/backward.js'
import { crossEntropy, numericalGradient } from '../src/matrix.js'

// ---- 覆盖参数滑杆可能的各种配置 ----
const configs = [
  { name: '基础 1层1头', cfg: { vocab_size: 32, block_size: 8, n_layer: 1, n_head: 1, n_embd: 4, bias: true } },
  { name: '多层多头',    cfg: { vocab_size: 32, block_size: 8, n_layer: 2, n_head: 2, n_embd: 8, bias: true } },
  { name: '更大配置',    cfg: { vocab_size: 48, block_size: 12, n_layer: 2, n_head: 4, n_embd: 16, bias: true } },
  { name: '无偏置',      cfg: { vocab_size: 24, block_size: 6, n_layer: 1, n_head: 1, n_embd: 4, bias: false } },
]

let allPass = true
for (const { name, cfg } of configs) {
  const { params } = createModel(cfg, 42)
  const T = cfg.block_size
  const idx = Array.from({ length: T }, (_, i) => (i * 7 + 3) % cfg.vocab_size)
  const targets = idx.slice(1).concat([(idx[0] + 5) % cfg.vocab_size])

  const { logits, cache } = forward(params, idx, targets, cfg)
  const ce = crossEntropy(logits, targets)
  zeroGrad(params)
  backward(params, cache, ce.dlogits, cfg)

  const lossFn = () => forward(params, idx, targets, cfg).loss
  let worstAbs = 0
  let worstKey = ''
  for (const key of Object.keys(params)) {
    const M = params[key].value
    const ng = numericalGradient(lossFn, M, 1e-5)
    const ag = params[key].grad
    const rows = M.length
    const cols = M[0].length
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const absErr = Math.abs(ag[r][c] - ng[r][c])
        if (absErr > worstAbs) { worstAbs = absErr; worstKey = key }
      }
    }
  }
  const pass = worstAbs < 1e-4
  allPass = allPass && pass
  console.log(`${pass ? '✅' : '❌'} ${name}  (${paramCount(params)} 参数)  loss=${ce.loss.toFixed(4)}  worstAbsErr=${worstAbs.toExponential(2)} @${worstKey}`)
}

console.log(`\n${allPass ? '✅ 全部配置梯度检查通过' : '❌ 存在失败配置'}`)
