# AI 学习本 · AI Notebook

> 翻开这本笔记，把 AI 从头看懂。

在浏览器里**亲手训练一个迷你大模型**的交互式教学实验台——零安装、零依赖、真实计算、没有任何预设剧本。

🔗 **在线体验：https://ai-notebook-seven.vercel.app**

## 它是什么

一个只会"猜下一个字"的几百参数小模型，在你的浏览器里从零开始**真实训练**。从词嵌入、自注意力、多头到位置编码，每一颗螺丝都拆开讲透，配得上"手把手"三个字——不，是"手算"。

## 功能板块

| 板块 | 内容 |
|---|---|
| **文本模型** | 预训练 → SFT 微调 → DPO 对齐 三阶段训练，loss 曲线、权重热力图、流式生成、语义空间 PCA |
| **图像模型 / 语音模型** | 像素序列与音高序列训练，同一个 Transformer 通吃 |
| **多模态** | 一个混合模型同时学习文本 + 像素 + 音高 |
| **Transformer** | 深度解析现代 LLM 基石技术：与 CNN 对比、核心零件逐个拆、论文《Attention Is All You Need》深度解析 |
| **动画词典** | 14 个术语动画解释，任意板块的术语点击即可跳转 |
| **前沿** | 推理加速手册：KV 缓存 / 量化 / 蒸馏 / Flash Attention，可动手量化自己训练好的模型 |
| **论文** | 前沿论文库，从 Transformer 到 DeepSeek-R1 |
| **显微镜** | 把"猜下一个字"拆成 6 步手算，亲手算出模型的内部计算 |
| **成就 & 彩蛋** | 游戏化学习 + 隐藏彩蛋 |

## 亮点

- **真实训练，不是动画**：所有权重真实存在、梯度真实传播、loss 真实下降
- **每个概念都能动手**：滑杆、手算、量化、注意力直播——理解靠动手，不靠背诵
- **纯手写实现**：矩阵运算、Transformer、反向传播全部手写（~300 参数级，代码可读）
- **手绘素描本 UI**：wired-elements 手绘控件 + 演草纸风，零卡片设计，板板正正

## 快速开始

```bash
npm install
npm run dev      # 本地开发 → http://localhost:5173
npm run build    # 构建到 dist/
```

### 验证脚本（Node 直跑，不需要浏览器）

```bash
node scripts/gradcheck.js        # 数值梯度 vs 解析梯度（4 配置验证 backward）
node scripts/train-demo.js       # 预训练演示（loss 下降 + 生成）
node scripts/sft-demo.js         # SFT 演示（防遗忘修复）
node scripts/dpo-demo.js         # DPO 演示（偏好对齐）
node scripts/m4-demo.js          # 显微镜 + 注意力直播数据源
node scripts/finetune-check.js   # SFT 防遗忘验证
```

## 技术栈

- **Vite** + 纯 JavaScript（零框架、零 ML 库）
- 手写矩阵运算 / Transformer / 反向传播（SGD + AdamW）
- **wired-elements**（Rough.js）手绘 UI 控件
- 浏览器 localStorage 持久化（模型、快照、成就、主题）

## 项目结构

```
src/
  matrix.js       纯 JS 矩阵运算库
  model.js        微型 Transformer（~300 参数级）
  train.js        训练循环 + 优化器（SGD / AdamW）
  backward.js     反向传播主流程
  sft.js / dpo.js SFT / DPO 训练
  microscope*.js  显微镜模式（逐步前向手算）
  attn.js         注意力直播数据源
  embed.js        语义空间 PCA 可视化
  pixel.js / melody.js / mm.js  图像 / 语音 / 多模态
  dict*.js        动画词典（术语 + 动画 + 详情）
  papers.js       论文库
  transformer.js  Transformer 深度解析板块
  ach.js          成就系统
```

## 部署

```bash
vercel --prod      # 已配置 vercel.json（Vite 预设 + SPA rewrite）
```

---

*没错，这个关于 LLM 的科普，也是 LLM 写的。*

MIT License