// ============================================================
// tests/ui-modules.spec.ts —— 图像 / 语音 / 多模态 / Transformer / 显微镜 / 词典 / 前沿
// wired-button 一律用 clickWired（真实点击优先，超时降级宿主派发）；
// .tab-btn/.chip/.calc-key/dict 卡片为原生按钮，直接 click()。
// ============================================================
import {
  test, expect, trackPage, expectDiagClean, expectFiniteMetrics,
  isDisabled, setValue, clickWired, openTab, litPixels,
} from './helpers'

test.describe('前端 UI · 图像模型（像素序列）', () => {

  test('训练→进度指标→生成出图→手绘板输入驱动训练', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    await openTab(page, 'image')

    await expect(page.locator('#pixStage')).toHaveText('未训练')
    await expect.poll(() => isDisabled(page, '#pixGenBtn'), { timeout: 15_000 }).toBe(true)

    // 训练内置图案
    await clickWired(page, '#pixTrainBtn')
    await expect
      .poll(async () => (await page.locator('#pixInfo').textContent()) ?? '', { timeout: 40_000 })
      .toMatch(/训练中…\s*\d+\s*\/\s*\d+/)
    // 训练中进度必须是有效整数（不为 NaN/Infinity/空）
    expectFiniteMetrics(
      ((await page.locator('#pixInfo').textContent()) ?? '').replace(/[^\d/.]/g, ''),
      'pixInfo(训练中)'
    )

    await expect(page.locator('#pixStage')).toHaveText('已学会', { timeout: 120_000 })
    await expect.poll(() => isDisabled(page, '#pixGenBtn'), { timeout: 15_000 }).toBe(false)
    await expect(page.locator('#pixInfo')).toContainText('完成')

    // 生成：逐像素动画结束后画布被真正绘制 + 历史缩略图出现
    await clickWired(page, '#pixGenBtn')
    await expect(page.locator('#pixInfo')).toContainText('生成完成', { timeout: 40_000 })
    await expect.poll(() => litPixels(page, '#pixOut'), { timeout: 20_000 }).toBeGreaterThan(64)
    await expect(page.locator('#pixHistory canvas').first()).toBeVisible()

    // --- 手绘板：自己画 → 作为训练数据 ---
    await clickWired(page, '#pixDrawBtn')
    const board = page.locator('#pixBoard')
    await board.waitFor({ state: 'visible', timeout: 10_000 })
    await board.scrollIntoViewIfNeeded()
    const box = await board.boundingBox()
    expect(box, '画板应有实际尺寸').not.toBeNull()
    for (const [fx, fy] of [[0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.5, 0.35]]) {
      await page.mouse.move(box!.x + box!.width * fx, box!.y + box!.height * fy)
      await page.mouse.down()
      await page.mouse.move(box!.x + box!.width * (fx + 0.09), box!.y + box!.height * (fy + 0.06), { steps: 4 })
      await page.mouse.up()
    }
    await expect(page.locator('#pixInfo')).toContainText('你的画已就绪', { timeout: 20_000 })

    // 用手绘图案训练（修复 gridToSeq 后不再产生 -43 非法 token 崩溃）
    await clickWired(page, '#pixTrainBtn')
    await expect(page.locator('#pixStage')).toHaveText('已学会', { timeout: 120_000 })

    // 清空画板开关
    await clickWired(page, '#pixClearBtn')

    // 扩散加噪滑杆往返，无异常
    await setValue(page, '#pixNoise', 1)
    await setValue(page, '#pixNoise', 0)

    expectDiagClean(diag, '图像模型训练+手绘板')
  })
})

test.describe('前端 UI · 语音模型（旋律序列）', () => {

  test('频率滑杆反馈→简谱应用→训练→模型作曲', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    await openTab(page, 'voice')

    // 控件反馈：频率滑杆 → Hz 文案即时刷新
    await setValue(page, '#freqSlider', 880)
    await expect(page.locator('#freqVal')).toHaveText('880 Hz')
    await setValue(page, '#freqSlider', 220)
    await expect(page.locator('#freqVal')).toHaveText('220 Hz')

    // 自定义简谱作为训练数据（wired-input 需经 host .value 写入）
    await setValue(page, '#melText', '1234567')
    await clickWired(page, '#melApplyBtn')
    await expect(page.locator('#melInfo')).toContainText('7 个音')

    await expect(page.locator('#melStage')).toHaveText('未训练')
    await expect.poll(() => isDisabled(page, '#melGenBtn'), { timeout: 15_000 }).toBe(true)

    await clickWired(page, '#melTrainBtn')
    await expect
      .poll(async () => (await page.locator('#melInfo').textContent()) ?? '', { timeout: 40_000 })
      .toMatch(/训练中…\s*\d+\s*\/\s*\d+/)
    expectFiniteMetrics(
      ((await page.locator('#melInfo').textContent()) ?? '').replace(/[^\d/.]/g, ''),
      'melInfo(训练中)'
    )

    await expect(page.locator('#melStage')).toHaveText('已学会', { timeout: 120_000 })
    await expect.poll(() => isDisabled(page, '#melGenBtn'), { timeout: 15_000 }).toBe(false)

    // 模型作曲（不点击播放，避开音频自动播放策略噪声）
    await clickWired(page, '#melGenBtn')
    await expect(page.locator('#melHistory .chip').first()).toBeVisible({ timeout: 20_000 })
    await expect(page.locator('#melInfo')).toContainText('48 个音')

    expectDiagClean(diag, '语音模型训练')
  })
})

test.describe('前端 UI · 多模态混合训练', () => {

  test('混合训练→三模态生成按钮解锁→文本续写/画图/图像转文本', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    await openTab(page, 'multi')

    // 初始化即就绪：token 统计含有效数字
    await expect(page.locator('#mmInfo')).toContainText('混合模型就绪', { timeout: 15_000 })
    expectFiniteMetrics(
      ((await page.locator('#mmInfo').textContent()) ?? '').replace(/[^\d/.]/g, ''),
      'mmInfo(就绪)'
    )
    await expect(page.locator('#mmTxtTrack')).toContainText('月光如流水')

    await clickWired(page, '#mmTrainBtn')
    await expect
      .poll(async () => (await page.locator('#mmInfo').textContent()) ?? '', { timeout: 60_000 })
      .toMatch(/混合训练中…\s*\d+\s*\/\s*3000/)
    expectFiniteMetrics(
      ((await page.locator('#mmInfo').textContent()) ?? '').replace(/[^\d/.]/g, ''),
      'mmInfo(训练中)'
    )

    // 完成：三个生成按钮解除置灰
    for (const id of ['#mmTxtBtn', '#mmImgBtn', '#mmAudBtn']) {
      await expect.poll(() => isDisabled(page, id), { timeout: 150_000 }).toBe(false)
    }

    // 续写文本模态
    await clickWired(page, '#mmTxtBtn')
    await expect
      .poll(async () => ((await page.locator('#mmTxtOut').textContent()) ?? '').trim().length, { timeout: 20_000 })
      .toBeGreaterThan(0)
    // 画图模态：画布被绘制
    await clickWired(page, '#mmImgBtn')
    await expect.poll(() => litPixels(page, '#mmImgOut'), { timeout: 20_000 }).toBeGreaterThan(16)
    // 跨模态转换：图像→文本（修复后的原生 select 默认值可用）
    await clickWired(page, '#mmToTextBtn')
    await expect(page.locator('#mmXOut')).toContainText('的文本形态', { timeout: 15_000 })

    expectDiagClean(diag, '多模态混合训练')
  })
})

test.describe('前端 UI · Transformer 深度解析', () => {

  test('注意力柱状图可见 + softmax 滑杆联动 + 手算四步走查 + 点词切换', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    await openTab(page, 'trans')

    // 注意力权重柱状图（默认演示「我」）：柱体可见且标签为固定占比
    await expect(page.locator('#attnBars .attn-bar')).toHaveCount(3)
    await expect(page.locator('#attnBars .attn-bar .fill').first()).toBeVisible({ timeout: 15_000 })
    const labels = (await page.locator('#attnBars .attn-bar div:last-child').allTextContents()).map((s) => s.trim())
    expect(labels).toEqual(['我 31%', '爱 46%', '你 23%'])

    // softmax 滑杆联动：拖动分数 → 占比柱状图重绘且百分比合计 ≈100%
    await setValue(page, '#sm0', 90)
    const softLabels = (await page.locator('#softBars .soft-bar div:last-child').allTextContents()).map((s) => s.trim())
    expect(softLabels.length).toBe(3)
    const pctSum = softLabels.reduce((s, t) => s + Number(t.match(/(\d+)%/)?.[1] ?? NaN), 0)
    expect(Number.isFinite(pctSum), `softmax 占比应为数字：${softLabels.join(',')}`).toBe(true)
    expect(pctSum).toBeGreaterThanOrEqual(99)
    expect(pctSum).toBeLessThanOrEqual(101)

    // 一步一步算：4 步走查，文案编号递增且互不重复，随后循环回第 1 步
    const seen: string[] = []
    for (let i = 1; i <= 4; i++) {
      await clickWired(page, '#attnCalcBtn')
      const msg = ((await page.locator('#attnCalcMsg').textContent()) ?? '').trim()
      expect(msg, `第 ${i} 次点击应显示第 ${i}/4 步`).toMatch(new RegExp(`第 ${i}/4 步`))
      expect(seen).not.toContain(msg)
      seen.push(msg)
    }
    await clickWired(page, '#attnCalcBtn')
    await expect(page.locator('#attnCalcMsg')).toContainText('第 1/4 步')

    // 点词切换注意力对象
    await page.locator('.attn-demo-token[data-t="爱"]').click()
    await expect(page.locator('.attn-demo-token.on')).toHaveAttribute('data-t', '爱')
    await expect(page.locator('#attnNote')).toContainText('宾语')

    expectDiagClean(diag, 'Transformer 解析')
  })
})

test.describe('前端 UI · 显微镜手算逐字推演', () => {

  test('6 步推演渲染 → 自己算计算器 → 校验与看答案', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')

    // 显微镜依赖 state.model：先构建
    await openTab(page, 'text')
    await expect(page.locator('#modelInfo')).toContainText(/token/, { timeout: 15_000 })

    const msRoot = page.locator('#microscopeRoot')
    await msRoot.scrollIntoViewIfNeeded()
    await setValue(page, '#msInput', '月光')
    await clickWired(page, '#msRun')

    // 逐字推演步骤渲染（查表 → 位置 → 注意力 → 前馈 → 打分 → 选字）
    const steps = msRoot.locator('.ms-step')
    await expect(steps.first()).toBeVisible({ timeout: 20_000 })
    expect(await steps.count(), '显微镜应展示完整手算步骤').toBeGreaterThanOrEqual(6)
    // 注意力权重与候选概率均为有效有限数字
    expectFiniteMetrics(
      ((await msRoot.locator('.ms-attn').first().textContent()) ?? '').replace(/[^\d.]/g, ''),
      'ms-attn'
    )
    expectFiniteMetrics(
      ((await msRoot.locator('.ms-probs').first().textContent()) ?? '').replace(/[^\d.%]/g, ''),
      'ms-probs'
    )
    // 最终预测非空
    expect((await msRoot.locator('.ms-final').textContent()) ?? '').toContain('预测的下一个字')

    // 切换"自己算"：权重表 + 内置计算器
    await clickWired(page, '#msTabCalc')
    await expect(msRoot.locator('#msCalc .ms-table tr').first()).toBeVisible()

    // 计算器 UI：1 + 2 = → 3
    for (const k of ['1', '+', '2', '=']) {
      await msRoot.locator(`.calc-key[data-k="${k}"]`).click()
    }
    await expect(msRoot.locator('#calcDisplay')).toHaveText(/^3(\.0*)?$/)

    // 校验非法输入提示 + 看答案（数值必须有限）
    await setValue(page, '#msAns', 'abc')
    await clickWired(page, '#msCheck')
    await expect(msRoot.locator('#msCalcResult')).toContainText('请输入数字')
    await clickWired(page, '#msSkip')
    const ansRaw = (await msRoot.locator('#msCalcResult').textContent()) ?? ''
    expect(ansRaw).toContain('答案是')
    expectFiniteMetrics(ansRaw.replace('答案是', ''), 'msCalcResult')

    expectDiagClean(diag, '显微镜手算')
  })
})

test.describe('前端 UI · 动画词典与前沿手册', () => {

  test('动画词典：卡片网格 ↔ 详情页往返', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    await openTab(page, 'dict')

    const cards = page.locator('.dict-item.card')
    await expect(cards.first()).toBeVisible({ timeout: 15_000 })
    expect(await cards.count(), '词典应有多张术语卡').toBeGreaterThanOrEqual(5)

    const firstTerm = ((await cards.first().locator('.dict-term').textContent()) ?? '').trim()
    await cards.first().click()
    await expect(page.locator('#dictBack')).toBeVisible()
    await expect(page.locator('#dictDetailCanvas')).toBeVisible()
    await expect(page.locator('.dict-term').first()).toHaveText(firstTerm)

    await clickWired(page, '#dictBack')
    await expect(cards.first()).toBeVisible()

    expectDiagClean(diag, '动画词典')
  })

  test('前沿·推理加速：INT4 量化滑杆联动与量化生成对比', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')

    // 量化基于当前文本模型：等初始构建完成后先记录占位文案
    await openTab(page, 'text')
    await expect(page.locator('#modelInfo')).toContainText(/token/, { timeout: 15_000 })

    await openTab(page, 'frontier')
    await setValue(page, '#quantBits', 32)
    await expect(page.locator('#quantVal')).toHaveText('FP32')
    await setValue(page, '#quantBits', 4)
    await expect(page.locator('#quantVal')).toHaveText(/INT4/)

    const quantOutBefore = (await page.locator('#quantOut').textContent()) ?? ''
    await clickWired(page, '#quantGenBtn')
    // 输出必须从占位文案变为真实生成内容，且信息栏给出 INT4 体积/速度数字
    await expect
      .poll(async () => ((await page.locator('#quantOut').textContent()) ?? '') !== quantOutBefore, { timeout: 30_000 })
      .toBe(true)
    await expect(page.locator('#quantInfo')).toContainText('INT4')
    await expect(page.locator('#quantInfo')).toContainText(/\d+%/)
    expectFiniteMetrics(
      ((await page.locator('#quantInfo').textContent()) ?? '').replace(/[^\d.%]/g, ''),
      'quantInfo'
    )

    expectDiagClean(diag, '量化手册')
  })

  test('论文库 tab 渲染分类与内容', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    await openTab(page, 'papers')
    await expect(page.locator('#tab-papers .card, #tab-papers section').first()).toBeVisible({ timeout: 15_000 })
    expect(await page.locator('#tab-papers button').count()).toBeGreaterThanOrEqual(2)
    expectDiagClean(diag, '论文库')
  })
})
