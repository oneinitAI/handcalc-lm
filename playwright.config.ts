// ============================================================
// playwright.config.ts
// Handcalc-LM E2E 测试配置
// 运行：npm run test:e2e   （即 `playwright test`）
// ============================================================
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  // 测试文件目录
  testDir: './tests',

  /* 并行策略：用例相互独立（独立浏览器上下文）可并行；
     worker 数量刻意受限——训练循环是 CPU 密集的同步矩阵运算，并发过高会互相饿死主线程 */
  fullyParallel: true,
  workers: process.env.CI ? 2 : 3,

  /* 全局超时：单个用例最长 120 秒；整套测试最长 15 分钟 */
  timeout: 120_000,
  globalTimeout: 15 * 60_000,
  expect: { timeout: 10_000 },

  /* 失败时不立即放弃重试排查信息 */
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,

  /* 报告：终端列表 + HTML 报告（不自动打开浏览器） */
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report', open: 'never' }],
  ],

  use: {
    baseURL: 'http://localhost:5173/',
    trace: 'retain-on-failure',      // 失败时保留 Trace
    screenshot: 'only-on-failure',   // 失败时截图
    video: 'retain-on-failure',      // 失败时保留视频
    locale: 'zh-CN',
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
  },

  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1440, height: 900 },
      },
    },
  ],

  /* 自动启动 Vite dev server，等待 5173 可用后再跑测试。
     - BROWSER=none：抑制 vite.config.js 的 open:true 在宿主机弹出浏览器窗口
     - --strictPort：5173 被占用时直接报错，而不是悄悄换端口导致测试连错地址 */
  webServer: {
    command: 'npm run dev -- --port 5173 --strictPort',
    url: 'http://localhost:5173/',
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
    stdout: 'pipe',
    stderr: 'pipe',
    env: {
      ...process.env,
      BROWSER: 'none',
    },
  },
})
