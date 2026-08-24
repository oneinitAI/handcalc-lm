// ============================================================
// tests/persistence.spec.ts —— 持久化：模型快照 / 成就 / 主题刷新后完整还原
// ============================================================
import {
  test, expect, trackPage, expectDiagClean,
  setValue, getValue, trainThenPause, btnText, clickWired,
} from './helpers'

type Snap = {
  params: Record<string, number[][]>
  cfg: Record<string, unknown>
  itos: string[]
  stoi: Record<string, number>
  ts: number
}

const readSnaps = (page: Page): Promise<Snap[]> =>
  page.evaluate(() => JSON.parse(localStorage.getItem('handcalc:snaps') || '[]'))

const readAch = (page: Page): Promise<string[]> =>
  page.evaluate(() => {
    try { return JSON.parse(localStorage.getItem('handcalc:ach') || '[]') as string[] } catch { return [] }
  })

/** 若训练循环仍在跑则暂停 */
async function pauseIfTraining(page: Page) {
  for (let i = 0; i < 6; i++) {
    if ((await btnText(page, '#trainBtn')) !== '暂停') return
    await clickWired(page, '#trainBtn')
    await page.waitForTimeout(300)
  }
}

test.describe('持久化 · localStorage 快照/成就/主题', () => {

  test('训练→存快照→刷成就→切主题→刷新页面，全部结构化还原', async ({ page }) => {
    test.setTimeout(150_000)
    const diag = trackPage(page)
    await page.goto('/')

    // 等初始模型就绪（修复加载竞态后自动构建）并训练一小段
    await expect(page.locator('#modelInfo')).toContainText(/token/, { timeout: 20_000 })
    await trainThenPause(page)

    // ① 存快照 → handcalc:snaps 结构完整（修复 params 序列化路径后可用）
    await clickWired(page, '#snapSaveBtn')
    await expect(page.locator('#modelInfo')).toContainText('快照已存', { timeout: 15_000 })

    const snapsBefore = await readSnaps(page)
    expect(snapsBefore.length, '保存后应至少有 1 个快照').toBeGreaterThanOrEqual(1)
    const snap = snapsBefore[snapsBefore.length - 1]
    expect(typeof snap.ts, '快照应带时间戳 ts').toBe('number')
    expect(snap.params && typeof snap.params === 'object', '快照应包含 params 权重字典').toBe(true)
    for (const key of Object.keys(snap.params)) {
      expect(Array.isArray(snap.params[key]), `params.${key} 应为矩阵`).toBe(true)
    }
    expect(snap.cfg, '快照应包含 cfg 配置').toBeTruthy()
    for (const key of ['vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd']) {
      expect(snap.cfg, `cfg.${key} 缺失`).toHaveProperty(key)
    }
    expect(Array.isArray(snap.itos), '快照应包含 itos 字符表').toBe(true)
    expect(typeof snap.stoi === 'object' && snap.stoi !== null, '快照应包含 stoi 反查表').toBe(true)

    // ② 连续训练至解锁首个成就（满 100 步 → 「初见」；默认速度一帧 200 步）
    await clickWired(page, '#trainBtn')
    await expect
      .poll(async () => readAch(page), { timeout: 90_000 })
      .toContain('first')
    await pauseIfTraining(page)
    const achBefore = await readAch(page)
    expect(achBefore, '成就数组应为字符串数组结构').toEqual(expect.arrayContaining(['first']))

    // ③ 切暗色主题 → handcalc:theme 持久化
    await clickWired(page, '#themeBtn')
    await expect.poll(() => page.evaluate(() => localStorage.getItem('handcalc:theme')), { timeout: 15_000 })
      .toBe('dark')

    // ④ 刷新页面：一切从 localStorage 还原
    const snapsCount = snapsBefore.length
    const snapJson = JSON.stringify(snap)
    await page.reload()
    await expect(page.locator('#modelCard')).toBeVisible({ timeout: 30_000 })

    const snapsAfter = await readSnaps(page)
    expect(snapsAfter.length, '刷新后快照数量不变').toBe(snapsCount)
    expect(JSON.stringify(snapsAfter[snapsAfter.length - 1])).toBe(snapJson)

    const achAfter = await readAch(page)
    expect(achAfter, '成就数据刷新后仍在且包含 first').toEqual(expect.arrayContaining(['first']))
    expect(Array.isArray(achAfter), '成就数据应为数组结构').toBe(true)

    // 主题与成就徽章还原
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark')
    await expect(page.locator('#achBadge')).toContainText(/成就 [1-9]\d*\/\d+/)

    // 读快照按钮可成功还原（UI 层持久化闭环）
    await clickWired(page, '#snapLoadBtn')
    await expect(page.locator('#modelInfo')).toContainText('已读快照', { timeout: 15_000 })

    expectDiagClean(diag, '持久化')
  })

  test('主题切换往返：light↔dark 均写入 handcalc:theme', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')

    await clickWired(page, '#themeBtn')
    await expect.poll(() => page.evaluate(() => localStorage.getItem('handcalc:theme')), { timeout: 15_000 })
      .toBe('dark')
    await expect.poll(() => page.evaluate(() => document.documentElement.getAttribute('data-theme')), { timeout: 15_000 })
      .toBe('dark')

    await clickWired(page, '#themeBtn')
    await expect.poll(() => page.evaluate(() => document.documentElement.getAttribute('data-theme')), { timeout: 15_000 })
      .toBe('light')

    expectDiagClean(diag, '主题切换')
  })
})
