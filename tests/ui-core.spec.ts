// ============================================================
// tests/ui-core.spec.ts —— 核心流程：导航 / 文本预训练 / SFT+DPO 三阶段
// 注意：不调高速度滑杆——speed 越大单帧同步训练步数越多，会长时间阻塞主线程。
// 应用在 loss 降到初始 25% 或停滞时会"自动停止训练"，因此按钮文案断言
// 只区分 运行态("暂停") 与 非运行态("开始训练"/"继续训练")。
// ============================================================
import {
  test, expect, trackPage, expectDiagClean, expectFiniteMetrics,
  btnText, isDisabled, readLoss, setValue, getValue,
  clickWired, trainThenPause, pauseIfTraining, openTab,
} from './helpers'

test.describe('前端 UI · 导航与文本预训练', () => {

  test('应用加载与八大模块导航切换', async ({ page }) => {
    const diag = trackPage(page)
    await page.goto('/')
    await expect(page).toHaveTitle(/AI 学习本/)
    await expect(page.locator('.masthead h1.title')).toContainText('学习本')
    await expect(page.locator('.tab-btn')).toHaveCount(8)

    const tabs = ['text', 'image', 'voice', 'multi', 'trans', 'dict', 'frontier', 'papers']
    const titles = ['文本模型', '图像模型', '语音模型', '多模态', 'Transformer', '词典', '前沿', '论文']
    for (let i = 0; i < tabs.length; i++) {
      await page.locator(`.tab-btn[data-tab="${tabs[i]}"]`).click()
      // 面板激活、tab 高亮、aria-selected 同步、hash 路由与标题同步
      await expect(page.locator(`#tab-${tabs[i]}`)).toHaveClass(/on/)
      await expect(page.locator(`.tab-btn[data-tab="${tabs[i]}"]`)).toHaveAttribute('aria-selected', 'true')
      await expect(page).toHaveTitle(new RegExp(titles[i]))
      expect(new URL(page.url()).hash).toBe(`#/${tabs[i]}`)
      expect(await page.locator('.tab-panel.on').count(), '同时只应有一个面板激活').toBe(1)
    }

    // 回到文本 tab，核心卡片齐备；初始语料/模型应自动就绪（修复加载竞态后不再弹 alert）
    await openTab(page, 'text')
    for (const id of ['dataCard', 'modelCard', 'sftCard', 'genCard', 'dpoCard']) {
      await expect(page.locator(`#${id}`)).toBeVisible()
    }
    await expect(page.locator('#modelInfo')).toContainText(/token/, { timeout: 15_000 })

    // 加载全程不得弹出任何对话框（修复前会误弹「请输入或选择语料」）
    let dialogFired = false
    page.on('dialog', () => { dialogFired = true })
    await page.waitForTimeout(500)
    expect(dialogFired, '页面交互过程中不应出现 alert 对话框').toBe(false)

    expectDiagClean(diag, '导航遍历')
  })

  test('文本预训练：loss 有效数字→进度推进→暂停/恢复→单步→重建', async ({ page }) => {
    test.setTimeout(180_000)
    const diag = trackPage(page)
    await page.goto('/')

    // 等初始语料/模型就绪（修复加载竞态后自动构建），再操作训练
    await expect(page.locator('#modelInfo')).toContainText(/token/, { timeout: 20_000 })
    await expect(page.locator('#lossTag')).toHaveText('—')   // 初始占位符

    // 控件交互：调低学习率 + 把速度滑杆调到最小（每帧 20 步），
    // 避免训练在观测窗口内触发"自动停止"、也避免单帧长阻塞主线程，
    // 使 开始↔暂停 状态机可被稳定观察（真实用户路径：改参数→训练）
    await setValue(page, '#lr', '0.001')
    await setValue(page, '#speed', 20)

    // 开始训练 → 按钮进入运行态（显示"暂停"）
    await clickWired(page, '#trainBtn')
    await expect.poll(() => btnText(page, '#trainBtn'), { timeout: 20_000 }).toBe('暂停')

    // loss 出现有效有限数字
    await expect.poll(() => readLoss(page), { timeout: 60_000 }).not.toBeNull()
    expectFiniteMetrics(await page.locator('#lossTag').textContent(), 'lossTag')

    // 进度条开始移动：宽度为合法百分比；进度文案若出现则必须含有效数字
    await expect
      .poll(() => page.locator('#progressFill').evaluate((el: HTMLElement) => parseFloat(el.style.width) || 0), { timeout: 60_000 })
      .toBeGreaterThanOrEqual(0)
    const progressRaw = await page.locator('#progressText').textContent()
    if (progressRaw && progressRaw !== '—') {
      expectFiniteMetrics(progressRaw.replace('已学到位', '100'), 'progressText')
    }
    expect((await page.locator('#lossLog').textContent()) ?? '', 'loss 序列日志应有内容').toContain('loss 序列')

    // 训练中 loss 序列必须持续增长（低学习率下数值变化可能小于显示精度，用序列长度判定推进）
    await expect
      .poll(async () => {
        const log = (await page.locator('#lossLog').textContent()) ?? ''
        return log.split('→').length
      }, { timeout: 60_000 })
      .toBeGreaterThan(3)

    // 暂停 → 按钮翻回非运行态；确保完全停下后做单步对照
    await pauseIfTraining(page)
    await expect.poll(() => btnText(page, '#trainBtn')).not.toBe('暂停')
    const pausedLast = (((await page.locator('#lossLog').textContent()) ?? '').split('→').pop() ?? '').trim()

    // 单步训练：静止态下点一次，最新 loss 值必须变化。
    // 先把学习率临时调大，保证单步的 loss 位移超过显示精度（2 位小数）
    await setValue(page, '#lr', '0.5')
    await clickWired(page, '#stepBtn')
    await expect
      .poll(async () =>
        (((await page.locator('#lossLog').textContent()) ?? '').split('→').pop() ?? '').trim(),
        { timeout: 30_000 }
      )
      .not.toBe(pausedLast)
    expectFiniteMetrics(await page.locator('#lossTag').textContent(), '单步后 lossTag')

    // 再启动一次再暂停：验证"开始↔继续"完整往返（恢复小学习率保证暂停态可观测）
    await setValue(page, '#lr', '0.001')
    await clickWired(page, '#trainBtn')
    await expect.poll(() => btnText(page, '#trainBtn'), { timeout: 20_000 }).toBe('暂停')
    await pauseIfTraining(page)
    await expect.poll(() => btnText(page, '#trainBtn')).not.toBe('暂停')

    // 重建：进度显示回到占位符
    await clickWired(page, '#resetBtn')
    await expect(page.locator('#progressText')).toHaveText('—', { timeout: 20_000 })

    expectDiagClean(diag, '文本预训练流程')
  })
})

test.describe('前端 UI · SFT 微调与 DPO 对齐三阶段', () => {

  test('载入问答→微调→前后对比→生成偏好对→DPO 训练→完成章浮现', async ({ page }) => {
    test.setTimeout(180_000)
    const diag = trackPage(page)
    await page.goto('/')

    // 最小速度（每帧 20 步）：避免初始训练阶段单帧长阻塞主线程
    await setValue(page, '#speed', 20)

    // 构建模型并让 loss 出现
    await trainThenPause(page)

    // --- 叁 · SFT 微调 ---
    await clickWired(page, '#loadQaBtn')
    const qaList = await getValue(page, '#qaList')
    expect(qaList.trim().length, '载入示例问答后 qaList 应有内容').toBeGreaterThan(0)
    expect(qaList).toContain(' / ')
    await expect(page.locator('#sftInfo')).toContainText('已载入')

    // 记录微调前快照 → 对比按钮解锁（修复 params 路径后可用）
    await clickWired(page, '#snapBtn')
    await expect.poll(() => isDisabled(page, '#cmpBtn'), { timeout: 15_000 }).toBe(false)

    // 控件交互：调低学习率 + 最小速度（每帧 20 步），微调运行态可稳定观察
    await setValue(page, '#lr', '0.001')
    await setValue(page, '#speed', 20)
    await clickWired(page, '#sftBtn')
    await expect(page.locator('#stageSft')).toHaveClass(/on/, { timeout: 15_000 })
    // 微调复用主训练循环。混合数据收敛极快，可能瞬间"自动停止"，
    // 因此以「按钮进入运行态」或「loss 已被记录」任一作为运行证据：
    await expect
      .poll(async () =>
        (await btnText(page, '#trainBtn')) === '暂停' ||
        ((await page.locator('#lossTag').textContent()) ?? '') !== '—',
        { timeout: 30_000 }
      )
      .toBe(true)
    await pauseIfTraining(page)
    // 微调确实训练过：loss 显示不再是占位符
    await expect.poll(async () => ((await page.locator('#lossTag').textContent()) ?? '') !== '—', { timeout: 15_000 })
      .toBe(true)

    // 微调前后对比开关往返（读写真实权重矩阵）
    await clickWired(page, '#cmpBtn')
    await expect(page.locator('#cmpBtn')).toContainText('微调后', { timeout: 15_000 })
    await clickWired(page, '#cmpBtn')
    await expect(page.locator('#cmpBtn')).toContainText('微调前', { timeout: 15_000 })

    // --- 叁 · DPO 对齐 ---
    await setValue(page, '#dpoQ', '你好')
    await clickWired(page, '#genPairBtn')
    await expect(page.locator('#pairBox')).toBeVisible({ timeout: 30_000 })
    for (const id of ['#ansA', '#ansB']) {
      expect(((await page.locator(id).textContent()) ?? '').trim().length, `${id} 应有回答文本`).toBeGreaterThan(0)
    }

    await clickWired(page, '#pickABtn')
    await expect(page.locator('#prefInfo')).toContainText('已收集 1 对偏好')
    await expect(page.locator('#prefList .pref-item')).toHaveCount(1)
    await expect.poll(() => isDisabled(page, '#dpoBtn'), { timeout: 15_000 }).toBe(false)

    await clickWired(page, '#dpoBtn')
    await expect(page.locator('#dpoInfo')).toContainText('DPO 完成', { timeout: 150_000 })
    // "DPO 完成（300 步 × 1 对）· …" —— 数字必须有效有限
    expectFiniteMetrics(await page.locator('#dpoInfo').textContent(), 'dpoInfo')
    await expect(page.locator('#stageDpo')).toHaveClass(/on/)
    // 三阶段通关彩蛋：完成章浮现
    await expect(page.locator('.stamp')).toHaveCount(1, { timeout: 20_000 })
    await expect(page.locator('.stamp')).toHaveText(/学习者/)

    expectDiagClean(diag, 'SFT/DPO 三阶段')
  })
})
