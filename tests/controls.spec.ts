// ============================================================
// tests/controls.spec.ts —— 交互控件：语料选择 / 档位开关 / 数值反馈与重绘
// ============================================================
import {
  test, expect, trackPage, expectDiagClean, expectFiniteMetrics,
  setValue, getValue, isDisabled, clickWired,
} from './helpers'

test.describe('交互控件 · 语料/档位/速度', () => {

  test('语料 chip 切换、随机乱文开关、档位 μ 彩蛋、速度滑杆反馈', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    // 等初始模型就绪
    await expect(page.locator('#modelInfo')).toContainText(/token/, { timeout: 20_000 })

    // 语料 chip 切换 → textarea 内容与语料统计随之刷新（修复后不再抛 TypeError）
    const corpusBefore = await getValue(page, '#corpus')
    await page.locator('#dataCard .chip[data-id]').nth(1).click()
    await expect.poll(() => getValue(page, '#corpus'), { timeout: 15_000 }).not.toBe(corpusBefore)
    const infoRaw = await page.locator('#corpusInfo').textContent()
    expect((infoRaw ?? '').trim().length, 'corpusInfo 应显示语料统计').toBeGreaterThan(0)
    expectFiniteMetrics((infoRaw ?? '').replace(/[^\d]/g, ''), 'corpusInfo')

    // 词表可视化：折叠的 details，点开后字符表可见
    await expect(page.locator('#vocabView details summary')).toContainText(/字符表（\d+ 个）/, { timeout: 15_000 })
    await page.locator('#vocabView details summary').click()
    await expect(page.locator('#vocabView .vocab-chars')).toBeVisible({ timeout: 10_000 })
    await expect.poll(async () =>
      (await page.locator('#vocabView .vocab-chars span, #vocabView .vocab-chars div').count()), { timeout: 10_000 }
    ).toBeGreaterThan(0)

    // 实验开关：随机乱文 —— 语料再次变化
    const afterChip = await getValue(page, '#corpus')
    await clickWired(page, '#corpusRandom') // chip 原生按钮，clickWired 同样兼容
    await expect.poll(() => getValue(page, '#corpus'), { timeout: 15_000 }).not.toBe(afterChip)

    // 档位下拉（修复后保留原生 select）切到"超微" → 触发 μ 彩蛋标记；
    // 档位本身不自动重建模型，点「重建」后参数量应按新档位刷新
    await page.locator('#size').selectOption('ultratiny')
    await expect(page.locator('.title .mu')).toHaveText('μ', { timeout: 15_000 })
    await clickWired(page, '#resetBtn')
    await expect.poll(() => page.locator('#modelInfo').textContent(), { timeout: 15_000 })
      .toContain('token')
    expectFiniteMetrics(await page.locator('#modelInfo').textContent(), 'modelInfo')
    // 超微档参数量应显著小于默认中档
    const params = Number((await page.locator('#modelInfo').textContent())?.match(/([\d,]+)\s*参数/)?.[1]?.replace(/,/g, ''))
    expect(Number.isFinite(params)).toBe(true)
    expect(params).toBeLessThan(40000)

    // 优化器切换后单步训练不抛异常
    await page.locator('#optType').selectOption('sgd')
    await clickWired(page, '#stepBtn')
    await page.waitForTimeout(800)

    // 速度滑杆值反馈（wired-slider 的 .value 为数值类型）
    await setValue(page, '#speed', 777)
    await expect
      .poll(() => page.locator('#speed').evaluate((el) => Number((el as HTMLInputElement).value)), { timeout: 10_000 })
      .toBe(777)

    expectDiagClean(diag, '语料/档位控件')
  })

  test('生成面板：采样参数输入联动、示例开头 chip 填入、生成按钮随模型解锁', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')

    // 初始模型自动构建后生成按钮解锁（buildModel 内解除置灰）
    await expect(page.locator('#modelInfo')).toContainText(/token/, { timeout: 20_000 })
    await expect.poll(() =>
      page.locator('#genBtn').evaluate((el) => el.hasAttribute('disabled')), { timeout: 15_000 }
    ).toBe(false)

    // 采样参数输入可写入（wired-input 经 host .value）
    await setValue(page, '#temp', '0.1')
    await expect.poll(() => getValue(page, '#temp'), { timeout: 10_000 }).toBe('0.1')
    await setValue(page, '#len', '16')
    await setValue(page, '#topk', '5')
    await setValue(page, '#topp', '0.9')

    // 示例开头 chip 点击填入 prompt（原生按钮）
    await page.locator('#promptEx .chip[data-p="红岸基地"]').click()
    await expect
      .poll(() => getValue(page, '#prompt'), { timeout: 10_000 })
      .toBe('红岸基地')

    // 续写/问答模式切换开关
    await page.locator('#modeQa').click()
    await expect(page.locator('#modeQa')).toHaveClass(/on/)
    await page.locator('#modeCont').click()
    await expect(page.locator('#modeCont')).toHaveClass(/on/)

    expectDiagClean(diag, '生成面板控件')
  })

  test('梯度下降演示：开始→滚动→重置；语义空间 PCA 图绘制', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    await expect(page.locator('#modelInfo')).toContainText(/token/, { timeout: 20_000 })

    await clickWired(page, '#gdStart')
    await page.waitForTimeout(1200)   // 让小球动画跑几帧
    await clickWired(page, '#gdReset')
    await page.waitForTimeout(300)

    await clickWired(page, '#embedBtn')
    await page.waitForTimeout(800)
    await clickWired(page, '#embedResetBtn')
    await page.waitForTimeout(300)

    expectDiagClean(diag, '梯度下降/语义空间演示')
  })
})
