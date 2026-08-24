// ============================================================
// tests/backend.spec.ts —— 后端脚本逻辑集成验证
// 在全局 beforeAll 中用 child_process.exec 执行项目自带的数值验证脚本：
//   - scripts/gradcheck.js   （数值梯度 vs 解析梯度，多配置）
//   - scripts/train-demo.js  （小语料真实训练 + 生成）
// 断言：退出码为 0，stdout/stderr 不含 FAIL/Error 关键字（忽略大小写）。
// ============================================================
import { exec } from 'node:child_process'
import { test, expect } from './helpers'

const ROOT = process.cwd()

type ExecResult = { code: number; stdout: string; stderr: string }
const SCRIPTS = ['gradcheck.js', 'train-demo.js'] as const

test.describe('后端脚本逻辑集成验证', () => {
  const results: Record<string, ExecResult> = {}

  test.beforeAll(async () => {
    await new Promise<void>((resolve) => {
      let pending = SCRIPTS.length
      for (const name of SCRIPTS) {
        exec(`node scripts/${name}`, { cwd: ROOT, encoding: 'utf8' }, (err, stdout, stderr) => {
          // node 脚本非零退出时 err.code 携带退出码；0 正常
          const code =
            err && typeof (err as unknown as { code?: unknown }).code === 'number'
              ? (err as unknown as { code: number }).code
              : err
                ? 1
                : 0
          results[name] = { code, stdout: stdout ?? '', stderr: stderr ?? '' }
          if (--pending === 0) resolve()
        })
      }
    })
  }, 180_000)

  for (const name of SCRIPTS) {
    test(`scripts/${name} 退出码为 0`, async () => {
      const r = results[name]
      expect(r, `脚本 ${name} 应已在 beforeAll 执行`).toBeTruthy()
      expect(r.code, `退出码应为 0\n--- stdout ---\n${r.stdout}\n--- stderr ---\n${r.stderr}`).toBe(0)
    })

    test(`scripts/${name} 输出不含 FAIL/Error 关键字（忽略大小写）`, async () => {
      const r = results[name]
      const combined = `${r.stdout}\n${r.stderr}`.toLowerCase()
      expect(combined, '不应包含 "fail"').not.toContain('fail')
      expect(combined, '不应包含 "error"').not.toContain('error')
      expect(r.stdout.trim().length, '应有实质 stdout 输出').toBeGreaterThan(0)
    })
  }

  test('gradcheck：全部配置梯度检查通过', async () => {
    const r = results['gradcheck.js']
    expect(r.stdout).toContain('梯度检查通过')
    expect(r.stdout).toContain('✅')
    expect(r.stdout).not.toContain('❌')
    // 每个配置行都带 worstAbsErr 数值且有限
    const errs = [...r.stdout.matchAll(/worstAbsErr=(-?\d+(?:\.\d+)?e[+-]?\d+)/g)].map((m) => Number(m[1]))
    expect(errs.length).toBeGreaterThanOrEqual(4) // 脚本内置 4 组配置
    for (const n of errs) expect(Number.isFinite(n)).toBe(true)
  })

  test('train-demo：loss 为有限数字并显著收敛，产出生成文本', async () => {
    const r = results['train-demo.js']
    const losses = [...r.stdout.matchAll(/loss=(-?\d+(?:\.\d+)?)/g)].map((m) => Number(m[1]))
    expect(losses.length, '应输出多步 loss').toBeGreaterThanOrEqual(3)
    for (const n of losses) {
      expect(Number.isFinite(n), `loss=${n} 应为有效有限数字`).toBe(true)
      expect(n, `loss=${n} 不应为 NaN/Infinity`).not.toBeNaN()
    }
    // 语料 24 个字符 → 初始 loss ≈ ln(24) ≈ 3.18；2000 步后应明显低于它
    const finalLoss = losses[losses.length - 1]
    expect(finalLoss, `最终 loss ${finalLoss} 应显著低于 ln(24)≈3.18`).toBeLessThan(Math.log(24))
    // 生成段存在且非空
    expect(r.stdout).toContain('语料')
    expect(r.stdout).toContain('生成：')
  })
})
