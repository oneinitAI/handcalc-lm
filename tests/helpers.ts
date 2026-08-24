// ============================================================
// tests/helpers.ts —— 公共工具：诊断收集 / wired 控件兼容 / 断言辅助
//
// ⚠️ 本应用用 wired-elements 把 button/range/input/textarea 升级成了自定义元素：
//    断言与输入必须走 host 元素属性（.value/.textContent/[disabled]），
//    不能依赖 Playwright 对原生控件的 fill()/inputValue()/toBeDisabled()。
//    （select 在修复后保留原生实现，可用标准 API。）
// ============================================================
import { test, expect, type Page } from '@playwright/test'

// ------------------------------------------------------------
// 诊断收集：控制台 Error 级日志 + 未捕获异常 + 4xx/5xx 响应（排除 Vite HMR）
// ------------------------------------------------------------
export type Diagnostics = {
  consoleErrors: string[]
  pageErrors: string[]
  badResponses: { url: string; status: number }[]
}

export function trackPage(page: Page): Diagnostics {
  const diag: Diagnostics = { consoleErrors: [], pageErrors: [], badResponses: [] }
  page.on('console', (msg) => {
    if (msg.type() === 'error') diag.consoleErrors.push(msg.text())
  })
  page.on('pageerror', (err) => diag.pageErrors.push(String(err)))
  page.on('response', (res) => {
    const url = res.url()
    // 排除 HMR 热更新相关请求（@vite/client 的 ws 不产生 response 事件，这里再挡一层 fetch 轮询）
    if (url.includes('@vite/client') || url.includes('/@vite/') || url.toLowerCase().includes('hmr')) return
    if (res.status() >= 400) diag.badResponses.push({ url, status: res.status() })
  })
  return diag
}

/** 断言诊断干净：控制台无 Error、页面无未捕获异常、网络无 4xx/5xx */
export function expectDiagClean(diag: Diagnostics, scope: string) {
  expect(diag.consoleErrors, `${scope}：浏览器控制台出现 Error 级日志`).toEqual([])
  expect(diag.pageErrors, `${scope}：出现未捕获的页面异常（pageerror）`).toEqual([])
  expect(diag.badResponses, `${scope}：出现 4xx/5xx 网络响应`).toEqual([])
}

/**
 * 校验"指标显示框"内是有效有限数字：
 * 非空、不含 NaN/Infinity/null 字样、且解析出的数字全部有限。
 * 返回解析出的数字列表供进一步断言。
 */
export function expectFiniteMetrics(raw: string | null | undefined, label: string): number[] {
  const text = (raw ?? '').trim()
  expect(text, `${label} 不应为空字符串`).not.toBe('')
  const lower = text.toLowerCase()
  for (const bad of ['nan', 'infinity', 'null', 'undefined']) {
    expect(lower.includes(bad), `${label} 含非法值「${bad}」：「${text}」`).toBe(false)
  }
  const nums = (text.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi) ?? []).map(Number)
  expect(nums.length, `${label} 中应包含数字：「${text}」`).toBeGreaterThan(0)
  for (const n of nums) {
    expect(Number.isFinite(n), `${label} 含非有限数字：「${text}」`).toBe(true)
  }
  return nums
}

/** 提取文本中的全部有限数字（不做非空断言，用于"训练中 N/M"这类局部校验） */
export function finiteNumbers(raw: string | null | undefined): number[] {
  return ((raw ?? '').match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi) ?? []).map(Number)
}

// ------------------------------------------------------------
// wired 自定义元素兼容层
// ------------------------------------------------------------
/** wired-button 的可视文本（host textContent，压缩空白后比对） */
export const btnText = (page: Page, sel: string) =>
  page.locator(sel).evaluate((el) => (el.textContent || '').replace(/\s+/g, '').trim())

/** wired-button 的 disabled 状态以 [disabled] 属性为准（自定义元素无原生语义） */
export const isDisabled = (page: Page, sel: string) =>
  page.locator(sel).evaluate((el) => el.hasAttribute('disabled'))

/** 从 lossTag 文本中提取当前 loss 数值 */
export async function readLoss(page: Page): Promise<number | null> {
  const raw = await page.locator('#lossTag').textContent()
  const m = raw?.match(/loss\s+(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)/i)
  return m ? Number(m[1]) : null
}

/**
 * 设置控件值并触发 input/change 事件。
 * 兼容原生 input 与 wired-slider/wired-input/wired-textarea host
 * （均暴露 .value 属性；wired-slider 内部 change 会转发为宿主 input 事件）。
 */
export async function setValue(page: Page, selector: string, value: string | number) {
  await page.locator(selector).evaluate((el, v) => {
    ;(el as HTMLInputElement).value = String(v)
    el.dispatchEvent(new Event('input', { bubbles: true }))
    el.dispatchEvent(new Event('change', { bubbles: true }))
  }, value)
}

export const getValue = (page: Page, selector: string) =>
  page.locator(selector).evaluate((el) => (el as HTMLInputElement).value)

/**
 * 点击 wired-button：先尝试真实鼠标点击（覆盖 shadow 桥接链路），
 * 高负载下动作性检查可能长时间不通过，超时后降级为对宿主派发 click
 * （应用监听器就绑在宿主上，事件语义一致，仅跳过命中检查）。
 */
export async function clickWired(page: Page, selector: string, timeout = 3_000) {
  const loc = page.locator(selector)
  try {
    await loc.click({ timeout })
  } catch {
    await loc.dispatchEvent('click')
  }
}

/** 等待训练循环脱离"暂停"运行态（自动停止或被手动暂停都算结束） */
export async function pauseIfTraining(page: Page) {
  for (let i = 0; i < 8; i++) {
    if ((await btnText(page, '#trainBtn')) !== '暂停') return
    await clickWired(page, '#trainBtn')
    await page.waitForTimeout(300)
  }
}

/** 开始训练直到 loss 出现（≥10 步），然后确保脱离运行态（容忍自动停止/恢复竞态） */
export async function trainThenPause(page: Page) {
  await clickWired(page, '#trainBtn')
  await expect.poll(() => readLoss(page), { timeout: 60_000 }).not.toBeNull()
  await pauseIfTraining(page)
}

export async function openTab(page: Page, tab: string) {
  await page.locator(`.tab-btn[data-tab="${tab}"]`).click()
  await expect(page.locator(`#tab-${tab}`)).toHaveClass(/on/, { timeout: 10_000 })
}

/** 读取 canvas 已绘制的非透明像素数 */
export const litPixels = (page: Page, selector: string) =>
  page.locator(selector).evaluate((canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d')
    if (!ctx || !canvas.width || !canvas.height) return -1
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data
    let lit = 0
    for (let i = 3; i < data.length; i += 4) if (data[i] !== 0) lit++
    return lit
  })

// re-export 方便各 spec 引入
export { test, expect }
