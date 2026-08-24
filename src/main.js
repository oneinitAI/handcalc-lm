// ============================================================
// src/main.js
// AI 学习本 —— M1：预训练实验台（数据区 + 训练控制 + 基础可视化 + 流式生成）
// ============================================================

import './style.css'
import { CORPUS, buildVocab, tokensToText, TOKEN_NAME, USER, ASSISTANT, END } from './corpus.js'
import { createModel, paramCount } from './model.js'
import { trainStep, createOptimizer } from './train.js'
import { sample } from './sample.js'
import { createLossChart, createHeatmap, createAttnHeatmap, renderEmbed } from './ui.js'
import { sampleWithAttn } from './attn.js'
import { pca2d, topSimilarPairs } from './embed.js'
import { initMicroscopeUI } from './microscope-ui.js'
import { initTransformer, redrawTrans } from './transformer.js'
import { NOTES } from './notes.js'
import { FAQ } from './glossary.js'
import { ACHIEVEMENTS, loadEarned, saveEarned } from './ach.js'
import { PIXEL_PATTERNS, gridToSeq, seqToGrid, renderGrid, attachDrawing, charToVal, valToChar } from './pixel.js'
import { MELODIES, parseMelody, playMelody, playTone, renderMelody } from './melody.js'
import { createGradientDescent, createWave } from './animation.js'
import { initMultiModal } from './mm.js'
import { initDict } from './dict.js'
import { PAPERS, PAPER_CATS } from './papers.js'
import { initParticles } from './particles.js'
import 'wired-elements'
import { DEFAULT_QA, QA_SETS, formatPairs, buildSftData, qaPrompt, extendVocab } from './sft.js'
import { dpoTrainStep, makeRefModel } from './dpo.js'

const SIZES = {
  ultratiny: { name: '超微', n_layer: 1, n_head: 1, n_embd: 4,  block_size: 6 },
  tiny:      { name: '微',   n_layer: 1, n_head: 1, n_embd: 8,  block_size: 8 },
  small:     { name: '小',   n_layer: 2, n_head: 2, n_embd: 16, block_size: 12 },
  medium:    { name: '中',   n_layer: 2, n_head: 4, n_embd: 32, block_size: 16 },
  large:     { name: '大',   n_layer: 4, n_head: 8, n_embd: 64, block_size: 24 },
}

// ---------- 状态 ----------
const state = {
  model: null,
  opt: null,
  losses: [],
  training: false,
  rafId: null,
  mode: 'cont', // 'cont' 续写 | 'qa' 问答
  snap: null,   // 权重快照（微调前，用于对比）
  sftData: null,
  prefs: [],    // DPO 偏好对 [{x, yw, yl}]
  pair: null,   // 当前待点选的回答对 {a, b}（token 数组）
  refParams: null, // DPO 参考模型（冻结）
  dpoOpt: null,  // DPO 专用优化器
  stage: 'pre',  // 'pre' 预训练 | 'sft' 微调 | 'dpo' 对齐
  blind: null,   // 盲测数据 { a, b, correctIsA }
  genCount: 0,   // 生成次数（Karpathy 彩蛋）
  initLoss: null,     // 初始 loss（前 10 步平均），进度条基准
  stopNotified: false, // 自动停止提示标志
  genHistory: [],   // 生成历史（内容积累）
  egg50: false,     // 欧拉彩蛋触发标志
  eggQuote: false,  // 训练师语录彩蛋标志
  ach: {
    earned: loadEarned(),
    totalSteps: 0,
    learnedOnce: false,
    crashedOnce: false,
    hotOnce: false,
    coldOnce: false,
    blindWin: false,
    calcWin: false,
    snapOnce: false,
    allDone: false,
  },
  corpusIds: null,  // 语料字符数组（预训练数据源）
  sftSeq: null,     // 问答对训练序列（微调数据源，存在则混合训练）
  mixRatio: 0.5,    // 微调时问答数据占比（0=纯语料，1=纯问答）
}

// ---------- DOM ----------
const app = document.getElementById('app')
app.innerHTML = `
  <main class="stage">
    <header class="masthead">
      <h1 class="title">AI <span class="hl">学习本</span></h1>
      <p class="sub">翻开这本笔记，把 AI 从头看懂</p>
      <button id="achBadge" class="ach-badge" title="已解锁的成就，点击查看">成就 0/10</button>
      <a class="gh-badge" href="https://github.com/oneinitAI/handcalc-lm" target="_blank" rel="noopener" title="开源仓库，欢迎 star">GitHub 源码 ↗</a>
    </header>
    <nav class="tabs" id="mainTabs">
      <button class="tab-btn on" data-tab="text">文本模型</button>
      <button class="tab-btn" data-tab="image">图像模型</button>
      <button class="tab-btn" data-tab="voice">语音模型</button>
      <button class="tab-btn" data-tab="multi">多模态</button>
      <button class="tab-btn" data-tab="trans">Transformer</button>
      <button class="tab-btn" data-tab="dict">词典</button>
      <button class="tab-btn" data-tab="frontier">前沿</button>
      <button class="tab-btn" data-tab="papers">论文</button>
    </nav>
    <button id="navToggle" class="nav-toggle" aria-label="打开菜单"><span></span><span></span><span></span></button>
    <div id="tab-text" class="tab-panel on">
    <div class="intro">
      <p>这里是一个让你<b>亲手训练迷你大模型</b>的实验台——一个只会"猜下一个字"的小机器，在你的浏览器里<b>真实训练</b>，没有任何预设剧本。</p>
      <p class="intro-path">你的旅程：<b>壹</b> 选语料 · <b>贰</b> 训练它学说话 · <b>叁</b> 微调它会回答 · <b>肆</b> 生成看它想什么 · <b>伍</b> 对齐让它变讨喜 · <b>陆</b> 显微镜看穿每一步</p>
      <p class="muted">每张卡片下方有「怎么做」指引；右上「翻面」可看背后的公式。</p>
      <p><button id="startHere" class="start-here" title="带你到训练区：选语料 → 训练 → 生成，五分钟看到模型学会说话">新手从这里开始 →</button></p>
      <p class="note">（AI 味这么重，太八股了 <span class="yawn">(￣へ￣)</span> 可是 AI 为啥会这么说呢……）</p>
    </div>

    <div class="stage-bar">
      <span id="stagePre" class="stage-dot on">壹 预训练</span>
      <span class="stage-arrow">→</span>
      <span id="stageSft" class="stage-dot">贰 微调</span>
      <span class="stage-arrow">→</span>
      <span id="stageDpo" class="stage-dot">叁 对齐</span>
      <button id="flipBtn" class="btn ghost flip-btn">翻面 · 进阶模式</button>
      <button id="themeBtn" class="btn ghost">暗色模式</button>
    </div>

    <section class="card" id="dataCard">
      <h2>壹 · 语料</h2>
      <div class="corpus-pick">
        ${CORPUS.map((c) => `<button class="chip" data-id="${c.id}" title="${c.desc}">${c.title}</button>`).join('')}
        <span class="hint">或直接粘贴你的文本 ↓</span>
      </div>
      <div class="corpus-pick">
        <button id="corpusRandom" class="chip" title="实验：把当前语料的字随机拼成 200 字乱文当语料——看模型学'乱码'会是什么样">随机乱文（实验）</button>
        <button id="corpusDouble" class="chip" title="实验：把当前语料拼接成 1.5 倍长度——更长的语料让模型学得更久">语料 ×2（实验）</button>
      </div>
      <textarea id="corpus" rows="4" placeholder="在这里粘贴你自己的文本，模型会从这些字里学规律（语料越长训练越慢，但能学更多）"></textarea>
      <p class="muted" id="corpusInfo"></p>
      <div id="vocabView" class="vocab-view"></div>
    </section>

    <section class="card" id="modelCard">
      <h2>贰 · 模型与训练</h2>
      <div class="row">
        <label title="模型大小：越大学得越好但越慢。超微最快最笨，大最慢最聪明">档位
          <select id="size">
            <option value="ultratiny">超微</option>
            <option value="tiny">微</option>
            <option value="small">小</option>
            <option value="medium" selected>中</option>
            <option value="large">大</option>
          </select>
        </label>
        <label title="学习率=参数更新的步子大小。太大一步跨过头（loss 飙升=训崩），太小学不动。建议 0.003~0.03">学习率 <input id="lr" value="0.01" size="6"></label>
        <label title="优化器=怎么更新参数。Adam 自适应学习率，训练快（推荐）；SGD 最朴素的梯度下降，更'经典'">优化器
          <select id="optType">
            <option value="adam" selected>Adam</option>
            <option value="sgd">SGD</option>
          </select>
        </label>
        <label title="随机数种子。同一种子+同参数+同语料=每次训练结果完全一致（可复现）">种子 <input id="seed" value="42" size="5"></label>
        <label title="每帧（约 1/60 秒）训练多少步。调快=快速看结果；调慢=看清 loss 曲线和权重一点点生长">速度 <input id="speed" type="range" min="20" max="1000" value="200"></label>
        <button id="trainBtn" class="btn" title="开始/暂停训练。训练=让模型学会'猜下一个字'，loss 会下降">开始训练</button>
        <button id="stepBtn" class="btn ghost" title="只训练一步。想观察每一步的细微变化用这个">单步</button>
        <button id="resetBtn" class="btn ghost" title="用当前参数重新随机初始化模型（丢掉已经学到的，重新开始）">重建</button>
        <button id="snapSaveBtn" class="btn ghost" title="把当前模型权重存到浏览器本地（localStorage，保留最近 3 个），换语料/重建后能读回来">存快照</button>
        <button id="snapLoadBtn" class="btn ghost" title="读回最近保存的模型快照，恢复当时的权重和字符表">读快照</button>
      </div>
      <p class="muted" id="modelInfo"></p>
      <div id="paramBreak" class="param-break"></div>
      <div class="train-progress">
        <div class="progress-track"><div id="progressFill" class="progress-fill"></div></div>
        <span id="progressText" class="progress-text">—</span>
      </div>
      <div class="viz-row">
        <div class="viz">
          <div class="viz-title">loss 曲线 <span class="tag" id="lossTag" title="loss=损失，模型猜得有多烂。训练就是让它变小。困惑度=exp(loss)，模型有多迷茫">—</span></div>
          <canvas id="lossChart" class="canvas"></canvas>
          <div id="lossLog" class="loss-log"></div>
        </div>
        <div class="viz">
          <div class="viz-title">参数热力图 <span class="tag"><select id="hmMode" class="inline-select" title="热力图显示什么：权重=参数本身的值；梯度=参数当前被调整的方向和大小（训练中变化）"><option value="w">权重</option><option value="g">梯度</option></select></span></div>
          <canvas id="heatmap" class="canvas"></canvas>
        </div>
      </div>
      <div class="viz">
        <div class="viz-title">训练原理：梯度下降 <span class="tag">小球=参数 · 碗=loss 曲面 · 学习率=步长</span></div>
        <canvas id="gdCanvas" class="canvas"></canvas>
        <div class="row">
          <label title="学习率=小球每步迈多大。小=慢但稳；大=震荡；太大=冲出碗口（训崩，和真实训练一样）">学习率 <input id="gdLr" type="range" min="0.01" max="1.5" step="0.01" value="0.1"></label>
          <button id="gdStart" class="btn ghost" title="开始：小球沿梯度（最陡方向）滚向碗底">开始</button>
          <button id="gdReset" class="btn ghost" title="重置回起点，换个学习率再试">重置</button>
        </div>
      </div>
      <div class="viz">
        <div class="viz-title">底层原理：语义空间 <span class="tag">词向量 2D 投影 · 相近的字聚在一起</span></div>
        <canvas id="embedCanvas" class="canvas"></canvas>
        <div class="row">
          <button id="embedBtn" class="btn ghost" title="把当前模型的词向量 PCA 降到 2D 画出来——看它学到了哪些"语义簇"">看语义空间</button>
          <button id="embedResetBtn" class="btn ghost" title="对比：用一个随机初始化的模型画同样的图——训练前是乱糟糟的一团">对比训练前</button>
          <span class="hint" id="embedInfo"></span>
        </div>
        <div id="embedPairs" class="muted"></div>
      </div>
    </section>

    <section class="card" id="sftCard">
      <h2>叁 · 微调（SFT）</h2>
      <p class="muted">喂给模型问答对，让它从"接着写"学会"回答问题"——在预训练权重上继续真实训练。<b>SFT = Supervised Fine-Tuning（监督微调）</b>：为什么微调？预训练只会续写，微调换问答数据教它"回答问题"。</p>
      <div class="corpus-pick">
        <select id="qaSet" class="inline-select" title="选择要载入的示例问答套：通用对话 / 关于AI 学习本 / 趣味问答">
          <option value="general">通用对话</option>
          <option value="about">关于AI 学习本</option>
          <option value="fun">趣味问答</option>
        </select>
        <button id="loadQaBtn" class="chip" title="把选中的示例问答对填入下面的文本框（可再编辑）">载入示例问答</button>
        <span class="hint">每行一条「问题 / 回答」，斜杠分隔</span>
      </div>
      <textarea id="qaList" rows="4" placeholder="每行一条：问题 / 回答。例：你是谁 / 我是AI 学习本"></textarea>
      <div class="row">
        <label title="微调时，训练数据里问答对的占比。越高越会回答；但语料续写能力越容易被覆盖（灾难性遗忘）。50% 是平衡点">混合（问答占比）
          <select id="mixRatio">
            <option value="0">纯语料</option>
            <option value="0.3">30% 问答</option>
            <option value="0.5" selected>50% 问答（平衡）</option>
            <option value="0.7">70% 问答（偏回答）</option>
            <option value="1">纯问答（易覆盖语料）</option>
          </select>
        </label>
        <span class="hint">问答占比越高越会回答，但语料续写能力越容易被覆盖（灾难性遗忘）</span>
      </div>
      <div class="row">
        <button id="sftBtn" class="btn" title="开始微调：用问答对继续训练当前模型（学习率已自动调低，混合比例防遗忘）。微调后模型就会『回答问题』">开始微调</button>
        <button id="snapBtn" class="btn ghost" title="记录微调前的权重快照，之后可以一键对比"微调前只会续写 vs 微调后会回答"">记快照（微调前）</button>
        <button id="cmpBtn" class="btn ghost" disabled title="在当前权重和快照（微调前）之间切换，对比同一个模型微调前后的行为">切到：微调前</button>
      </div>
      <p class="muted" id="sftInfo"></p>
    </section>

    <section class="card" id="genCard">
      <h2>肆 · 生成</h2>
      <div class="corpus-pick">
        <button id="modeCont" class="chip on" title="续写模式：输入几个字，模型接着写下去（预训练后可用）">续写模式</button>
        <button id="modeQa" class="chip" title="问答模式：输入问题，模型试着回答（需要先微调解锁）">问答模式</button>
      </div>
      <div class="corpus-pick" id="promptEx">
        <button class="chip" data-p="月光" title="示例开头：点击填入">月光</button>
        <button class="chip" data-p="红岸基地" title="示例开头：点击填入">红岸基地</button>
        <button class="chip" data-p="从明天起" title="示例开头：点击填入">从明天起</button>
        <button class="chip" data-p="荷叶" title="示例开头：点击填入">荷叶</button>
        <button class="chip" data-p="黑夜" title="示例开头：点击填入">黑夜</button>
        <button class="chip" id="promptRandom" title="从当前语料里随机截一段当开头——探索模型会怎么接">从语料随机</button>
      </div>
      <div class="row">
        <input id="prompt" value="月光" size="14" title="输入开头文字（续写模式）或问题（问答模式）" placeholder="月光">
        <button id="genBtn" class="btn" disabled title="让模型生成：续写模式=接着写；问答模式=回答问题">生成</button>
        <label title="温度=生成时的冒险程度。0.1 每次都选最稳的字（乏味但稳）；0.8 正常；1.5 爱乱试（有惊喜也常乱）">温度 <input id="temp" value="0.8" size="4"></label>
        <label title="最多生成多少个字">长度 <input id="len" value="32" size="4"></label>
      </div>
      <details class="advanced">
        <summary>进阶采样（top-k / top-p）</summary>
        <div class="row">
          <label>top-k <input id="topk" value="0" size="4"></label>
          <span class="hint">0=关 · 只保留概率最高的前 k 个</span>
          <label>top-p <input id="topp" value="1" size="4"></label>
          <span class="hint">1=关 · 保留累计概率达 p 的候选</span>
          <label title="生成时压低已经出现过的字的概率，防止复读机式重复。1=关；1.2~1.5 防重复效果好">重复惩罚 <input id="repeatPenalty" value="1" size="4"></label>
          <span class="hint">1=关 · 压低已出现的字，防重复</span>
        </div>
      </details>
      <details class="advanced">
        <summary>温度对决赛（同一开头 · 三种温度）</summary>
        <p class="muted">同一个开头，用 0.2（稳）/ 0.8（正常）/ 1.5（冒险）各生成一次——感受温度如何改变模型的"性格"。</p>
        <div class="row">
          <input id="duelPrompt" value="月光" size="14" title="对决赛的开头文字">
          <button id="duelBtn" class="btn ghost" title="用三个温度（0.2/0.8/1.5）同时生成并排对比——直观感受"温度=性格"">开赛</button>
        </div>
        <div id="duelResult"></div>
      </details>
      <div id="genOut" class="gen">（先训练，再让它续写或回答）</div>
      <div id="probBar" class="prob-bar"></div>
      <div class="muted" id="perf"></div>
      <details class="advanced">
        <summary>🪞 人机接力（你和模型轮流写）</summary>
        <p class="muted">你写一句，点"模型接一句"，它续到句号——然后你继续写，轮流创作。</p>
        <div class="row">
          <button id="relayBtn" class="btn ghost" title="模型基于当前文本续写到下一个句号">模型接一句</button>
          <button id="relayReset" class="btn ghost" title="清空接力文本，重新开始">清空重来</button>
        </div>
        <textarea id="relayText" rows="3" title="接力文本：你和模型共同创作的草稿，可以直接编辑">月光</textarea>
      </details>
      <details class="advanced">
        <summary>🎭 幻觉探测（问它不知道的）</summary>
        <p class="muted">下面的问题都不在你的语料里——模型会"一本正经地编造"。这就是大模型的<b>幻觉</b>：它不是在撒谎，它只是不知道，却必须编出最像样的下一个字。</p>
        <div class="row">
          <button id="hallBtn" class="btn ghost" title="随机问一个语料外的问题，看模型怎么编">随机探测</button>
        </div>
        <div id="hallResult" class="muted"></div>
      </details>
      <div id="genHistory" class="gen-history"></div>
      <div class="viz">
        <div class="viz-title">Attention 直播 <span class="tag">模型在"看"哪些字</span></div>
        <canvas id="attnHeatmap" class="canvas"></canvas>
      </div>
    </section>

    <section class="card" id="dpoCard">
      <h2>伍 · 偏好对齐（DPO）</h2>
      <p class="muted">让模型生成两个回答，你告诉它哪个更好——它会学会偏向你的偏好。<b>DPO = Direct Preference Optimization（直接偏好优化）</b>：为什么？光会回答不够，还要"答得讨你喜欢"。这是 OpenAI 标注员做的真实工作（2023 年论文算法）。</p>
      <div class="row">
        <input id="dpoQ" value="你好" size="14" title="用来生成两个回答的问题">
        <button id="genPairBtn" class="btn" title="让当前模型用两种不同温度各生成一个回答，供你比较">生成两个回答</button>
      </div>
      <div class="pair" id="pairBox" hidden>
        <div class="answer">
          <div id="ansA" class="ans-text">（回答 A）</div>
          <button id="pickABtn" class="btn ghost" title="点选：你觉得 A 更好。你的选择会被当成训练信号（你就是人类标注员）">这个更好</button>
        </div>
        <div class="answer">
          <div id="ansB" class="ans-text">（回答 B）</div>
          <button id="pickBBtn" class="btn ghost" title="点选：你觉得 B 更好。你的选择会被当成训练信号">这个更好</button>
        </div>
      </div>
      <p class="muted" id="prefInfo">已收集 0 对偏好</p>
      <div id="prefList" class="pref-list"></div>
      <div class="row">
        <label title="β=DPO 的强度。越小越接近微调后的模型（保守）；越大改变越明显（大胆但可能学歪）。建议 0.1~1">β <input id="dpoBeta" value="0.5" size="4"></label>
        <label title="每对偏好训练多少步。越多越强化偏好，但太多会过拟合（只会背你选过的回答）">步数 <input id="dpoSteps" value="300" size="5"></label>
        <button id="dpoBtn" class="btn" disabled title="用收集的偏好对训练模型：让它学会偏向你选择的回答风格">开始 DPO 训练</button>
        <button id="dpoResetBtn" class="btn ghost" title="清空已收集的偏好对，重新开始">清空偏好</button>
      </div>
      <p class="muted" id="dpoInfo"></p>
      <div id="blindBox" hidden></div>
    </section>

    <div id="microscopeRoot"></div>

    <div id="glossaryRoot"></div>

    <div id="achRoot"></div>
    </div>
    <div id="tab-image" class="tab-panel">
      <section class="card" id="pixelCard">
        <h2>壹 · 图像模型（像素即序列）</h2>
        <p class="muted">图像在模型眼里 = <b>256 个像素值（16×16 灰度 16 级）</b>。训练它"猜下一个像素"——和文字模型<b>同一个架构</b>，换数据就能画图。</p>
        <div class="corpus-pick" id="pixelPick">
          ${PIXEL_PATTERNS.map((p) => `<button class="chip" data-pix="${p.id}">${p.name}</button>`).join('')}
          <button class="chip" id="pixDrawBtn">自己画</button>
          <button class="chip" id="pixClearBtn">清空画板</button>
        </div>
        <div class="row">
          <label title="图像模型的训练步数。像素序列 256 个，比文本更快学会">步数 <input id="pixSteps" value="2000" size="5"></label>
          <label title="学习率=参数更新步长。大=学得快但可能震荡">学习率 <input id="pixLr" value="0.05" size="5"></label>
          <label title="生成温度：低=稳定按学到的画，高=花样多（更有创造力）">温度 <input id="pixTemp" value="0.05" size="4"></label>
          <button id="pixTrainBtn" class="btn" title="用像素序列训练模型（和文本训练同一套代码，看 loss 下降）">开始训练</button>
          <button id="pixGenBtn" class="btn ghost" disabled title="从图案开头的 8 个像素开始生成整幅图（动画逐像素点亮）">生成</button>
          <span class="hint" id="pixInfo"></span>
        </div>
        <div class="viz-row">
          <div class="viz"><div class="viz-title">目标（选图案 / 自己画）</div><canvas id="pixTarget" class="pixel-canvas"></canvas><canvas id="pixBoard" class="pixel-canvas" hidden title="在这里涂画，作为训练数据"></canvas></div>
          <div class="viz"><div class="viz-title">模型生成 <span class="tag" id="pixStage">未训练</span></div><canvas id="pixOut" class="pixel-canvas"></canvas><div id="pixHistory" class="gen-history"></div></div>
        </div>
        <div class="viz"><div class="viz-title">loss 曲线（像素版）</div><canvas id="pixLoss" class="canvas"></canvas></div>
        <div class="howto">① 选图案，或点「自己画」在画板上涂一个图形 → ② 开始训练（看 loss 下降）→ ③ 生成，看模型从噪声逐步"画出"它学到的<br>它和文字模型是<b>同一个 Transformer</b>，只是把"字"换成 16 级灰度像素——图像生成模型（扩散模型）的雏形。</div>
      </section>
      <div class="teach">
        <h3>动手① 像素 = 数字</h3>
        <p>把鼠标悬停在图案格子上，看每个像素的灰度数字（0~15）——图就是一串数字。</p>
      </div>
      <div class="teach">
        <h3>动手② 猜下一个像素</h3>
        <p>选图案或自己画 → 开始训练 → 生成（逐像素动画）。它和文字模型<b>同一个架构</b>，只是把"字"换成像素（自回归图像生成）。</p>
      </div>
      <div class="teach">
        <h3>动手③ 扩散模型在做什么</h3>
        <p>拖滑杆把图案"加噪"成雪花，再滑回——扩散模型训练时学"去噪还原"，生成时从雪花一步步出图（Stable Diffusion 的原理）。</p>
        <div class="row">
          <label>加噪程度 <input id="pixNoise" type="range" min="0" max="1" step="0.05" value="0"></label>
          <canvas id="pixNoiseCanvas" class="pixel-canvas"></canvas>
        </div>
      </div>
      <div class="teach">
        <h3>动手④ 微调：让它学会"按提示画图"</h3>
        <p>把多个图案各配一个"提示标记"一起训练——之后你"告诉"它画哪个，它就画哪个。<b>文生图的雏形</b>。</p>
        <div class="row">
          <button id="pixFTBtn" class="btn" title="用 圆/心形/H 三个图案 + 各自提示标记一起训练">开始微调</button>
          <span class="hint" id="pixFTInfo"></span>
        </div>
        <div class="row" id="pixFTPick"></div>
        <canvas id="pixFTOut" class="pixel-canvas"></canvas>
      </div>
      <div class="teach">
        <h3>动手⑤ 图案插值（从 A 渐变成 B）</h3>
        <p>选两个图案，看像素值平滑过渡——这是"图像空间是连续的"的直接体验。</p>
        <div class="row">
          <select id="pixInterA" class="inline-select" title="起始图案"></select>
          <span class="hint">→</span>
          <select id="pixInterB" class="inline-select" title="目标图案"></select>
          <button id="pixInterBtn" class="btn ghost" title="动画播放像素从 A 渐变到 B">开始变形</button>
        </div>
        <canvas id="pixInterOut" class="pixel-canvas"></canvas>
      </div>
    </div>
    <div id="tab-voice" class="tab-panel">
      <div class="teach">
        <h3>动手① 频率 = 音高</h3>
        <p>拖滑杆选频率，点"试听"——低频沉、高频尖。这就是"音高"的物理来源（下方是它的一串谐波频谱）。</p>
        <div class="row">
          <label>频率 <input id="freqSlider" type="range" min="100" max="1000" value="440"></label>
          <span id="freqVal" class="hint">440 Hz</span>
          <button id="freqPlayBtn" class="btn ghost">试听</button>
        </div>
        <canvas id="specViz" class="canvas"></canvas>
        <canvas id="waveCanvas" class="canvas"></canvas>
      </div>
      <div class="teach">
        <h3>动手② 猜下一个音</h3>
        <p>选旋律或输入简谱 → 训练 → 模型作曲 → 播放。它也是"猜下一个"的 Transformer（音乐生成模型雏形）。</p>
      </div>
      <div class="teach">
        <h3>动手③ 微调：让它学会"按提示作曲"</h3>
        <p>把多首旋律各配一个"提示标记"一起训练——之后你说要哪首风格，它奏哪个。<b>提示作曲的雏形</b>。</p>
        <div class="row">
          <button id="melFTBtn" class="btn" title="用 小星星/欢乐颂/两只老虎 + 各自提示标记一起训练">开始微调</button>
          <span class="hint" id="melFTInfo"></span>
        </div>
        <div class="row" id="melFTPick"></div>
        <canvas id="melFTOut" class="canvas"></canvas>
      </div>
    <div class="teach">
        <h3>动手④ 旋律插值（从 A 渐变成 B）</h3>
        <p>选两首旋律，播放"A → 中间渐变 → B"——听旋律空间怎么连续过渡。</p>
        <div class="row">
          <select id="melInterA" class="inline-select" title="起始旋律"></select>
          <span class="hint">→</span>
          <select id="melInterB" class="inline-select" title="目标旋律"></select>
          <button id="melInterBtn" class="btn ghost" title="生成并播放：A 的前段 → 中间渐变 → B 的后段">开始变形</button>
        </div>
        <canvas id="melInterOut" class="canvas"></canvas>
      </div>
    <section class="card" id="voiceCard">
      <h2>壹 · 语音模型（旋律序列）</h2>
      <p class="muted">旋律 = <b>一串音高值</b>（简谱 1-7 + 0 休止）。训练它"猜下一个音"——学会后模型能<b>自己续写旋律并演奏</b>。</p>
      <div class="corpus-pick" id="melodyPick">
        ${MELODIES.map((m) => `<button class="chip" data-mel="${m.id}">${m.name}</button>`).join('')}
      </div>
      <div class="row">
        <label title="旋律模型的训练步数">步数 <input id="melSteps" value="2000" size="5"></label>
        <label title="作曲温度：低=规整保守，高=即兴大胆">温度 <input id="melTemp" value="0.6" size="4"></label>
        <label title="播放速度（BPM，每分钟多少拍）。快=活泼，慢=舒缓">BPM <input id="melBpm" value="220" size="5"></label>
        <button id="melTrainBtn" class="btn" title="用旋律的音高序列训练模型（猜下一个音）">开始训练</button>
        <button id="melGenBtn" class="btn ghost" disabled title="从旋律开头续写一段新旋律——模型作曲">模型作曲</button>
        <button id="melPlayBtn" class="btn ghost" disabled title="播放当前旋律（目标或模型创作）">播放</button>
        <span class="hint" id="melInfo"></span>
      </div>
      <div class="corpus-pick">
        <input id="melText" value="11556654433221" size="18" title="或自己输入简谱：1-7 音符，0 休止">
        <button id="melApplyBtn" class="chip" title="把输入框的简谱应用为训练数据">应用简谱</button>
      </div>
      <div class="viz-row">
        <div class="viz"><div class="viz-title">音高阶梯（目标）</div><canvas id="melViz" class="canvas"></canvas></div>
        <div class="viz"><div class="viz-title">模型续写 <span class="tag" id="melStage">未训练</span></div><canvas id="melGenViz" class="canvas"></canvas><div id="melHistory" class="gen-history"></div></div>
      </div>
      <div class="viz"><div class="viz-title">loss 曲线（旋律版）</div><canvas id="melLoss" class="canvas"></canvas></div>
      <div class="howto">① 选一首旋律（或输入自己的简谱）→ ② 开始训练（看 loss 下降）→ ③ 「模型作曲」续写新旋律 → 「播放」听它创作<br>语音模型和文字模型是<b>同一个架构</b>——真实 TTS/ASR 把声波变成数字（频谱）后，同样是"猜下一个值"。</div>
    </section>

    </div>
    <div id="tab-multi" class="tab-panel">
      <div class="teach">
        <h3>动手① 统一序列轨道：一切都是序列</h3>
        <p>同一个 Transformer 能吃三种序列：<b>字 token（文本）· 像素 token（图像）· 音高 token（音频）</b>——模型眼里没有"文字/图片/声音"，只有数字。</p>
        <div class="viz-row">
          <div class="viz"><div class="viz-title">文本 token</div><div id="mmTxtTrack" class="mm-track">月光如流水一...</div></div>
          <div class="viz"><div class="viz-title">像素 token</div><div id="mmImgTrack" class="mm-track mono">f f 0 0 ...</div></div>
          <div class="viz"><div class="viz-title">音高 token</div><div id="mmAudTrack" class="mm-track mono">1 1 5 5 6 6 ...</div></div>
        </div>
      </div>
      <div class="teach">
        <h3>动手② 多模态混合模型：一个模型通吃</h3>
        <p>把三种模态的 token <b>混进同一个模型</b>一起训练（每个序列前加模态标记）。训练后——它同时会<b>续写文本、画图、作曲</b>。这就是 <b>GPT-4V 的理念</b>：一个 Transformer 处理所有模态。</p>
        <div class="row">
          <button id="mmTrainBtn" class="btn" title="用 文本+像素+音高 三种序列混合训练同一个模型">训练混合模型</button>
          <span class="hint" id="mmInfo"></span>
        </div>
        <div class="row">
          <button id="mmTxtBtn" class="btn ghost" disabled title="输入『文』标记 + 月光 → 模型续写荷塘月色风格文本">续写文本</button>
          <button id="mmImgBtn" class="btn ghost" disabled title="输入 图标记 → 模型生成图像">画图</button>
          <button id="mmAudBtn" class="btn ghost" disabled title="输入 音标记 → 模型作曲并播放">作曲并播放</button>
        </div>
        <div id="mmTxtOut" class="gen"></div>
        <canvas id="mmImgOut" class="pixel-canvas"></canvas>
        <canvas id="mmAudOut" class="canvas"></canvas>
      </div>
      <div class="teach">
        <h3>动手③ 跨模态转换</h3>
        <p><b>让图案唱歌</b>：把像素值当音高播放——图也能"唱"。<b>图像即文本</b>：像素变成字符串。</p>
        <div class="row">
          <select id="mmXImg" class="inline-select" title="选一个图案做跨模态转换"></select>
          <button id="mmSingBtn" class="btn ghost" title="把图案的像素值映射成音高播放——图在唱歌">让图案唱歌</button>
          <button id="mmToTextBtn" class="btn ghost" title="把图案的像素变成字符文本——图像即文本">图像→文本</button>
        </div>
        <div id="mmXOut" class="muted mono"></div>
      </div>
      <div class="teach">
        <h3>动手④ 多模态应用与模态标记</h3>
        <p>真实世界的多模态模型：<b>CLIP</b>（把图和文字对齐到同一向量空间）、<b>Whisper</b>（语音→文字）、<b>GPT-4V</b>（看图+读文+听声）。它们背后都是同一个思路：<b>给不同模态加"标记"，让一个 Transformer 知道自己在处理哪种序列</b>——你刚才训练的混合模型就是这个机制。</p>
      </div>
    </div>
    <div id="tab-trans" class="tab-panel">
      <div id="transRoot"></div>
    </div>
    <div id="tab-dict" class="tab-panel">
      <div class="teach">
        <h3>动画词典</h3>
        <p>每个术语都有<b>动画解释</b>——看动画理解概念，比读定义快十倍。</p>
      </div>
      <div id="dictRoot"></div>
    </div>
    <div id="tab-frontier" class="tab-panel">
      <div class="teach">
        <h3>让模型更快：推理加速手册</h3>
        <p>真实世界的大模型推理有一整套加速技术——每一条都在解决"算力不够、等待太久"。</p>
        <div class="feat-grid">
          <div class="feat"><b>KV 缓存</b><span>算过一遍的键值存起来，后续生成直接复用——提速数倍。</span></div>
          <div class="feat"><b>量化 Quantization</b><span>权重从 32 位压到 8/4 位，模型变小变快 2~4 倍，损失一点精度。</span></div>
          <div class="feat"><b>蒸馏 Distillation</b><span>大模型当老师，教小模型模仿自己的输出——小模型继承大能力。</span></div>
          <div class="feat"><b>投机解码</b><span>小模型先打草稿，大模型批量验证——"猜得快，验得准"。</span></div>
          <div class="feat"><b>Flash Attention</b><span>重写注意力计算，减少内存读写——训练推理都快数倍。</span></div>
          <div class="feat"><b>连续批处理</b><span>多个请求的 token 打包一起算，GPU 不空转。</span></div>
        </div>
      </div>
      <div class="teach">
        <h3>动手：量化（把权重变"粗糙"）</h3>
        <p>真实模型用 32 位小数存权重；量化到 8/4 位让模型变小变快，但精度有损。拖滑杆把<b>你训练好的模型</b>压缩，看生成变化。</p>
        <div class="row">
          <label title="权重的存储精度。FP32=精确；INT8/INT4=粗糙但小、快">精度 <input id="quantBits" type="range" min="4" max="32" step="1" value="32"></label>
          <span id="quantVal" class="hint">FP32</span>
          <button id="quantGenBtn" class="btn ghost" title="用当前精度量化当前模型的权重，再生成对比">用当前精度生成</button>
        </div>
        <div id="quantOut" class="gen">（先训练文本模型，再拖精度到低档生成对比）</div>
        <div id="quantInfo" class="muted"></div>
      </div>
      <div class="teach">
        <h3>前沿架构与引擎</h3>
        <p><b>MoE（混合专家）</b>：把模型拆成很多"专家"，每个 token 只激活少数几个——同样的参数量，算得更少（DeepSeek 用的就是这个）。<b>长上下文</b>：RoPE 位置编码 + 稀疏注意力，让窗口从 4K 涨到 100K+。<b>推理引擎</b>：vLLM / TensorRT-LLM 把上面所有加速技术打包，服务商批量部署用它们。</p>
      </div>
      <div class="teach">
        <h3>大模型服务商（谁在提供"猜下一个字"）</h3>
        <p>你训练的是几百参数的玩具——下面这些是把它放大一亿倍、部署成产品的公司。</p>
        <div class="feat-grid">
          <div class="feat"><b>DeepSeek 深度求索</b><span>开源 + MoE 架构，推理便宜，靠"深度思考"出圈。</span></div>
          <div class="feat"><b>豆包 字节·火山方舟</b><span>国内生态完整，免费额度大，工具/Agent 能力强。</span></div>
          <div class="feat"><b>Kimi 月之暗面</b><span>超长上下文起家（200 万字），擅长长文档。</span></div>
          <div class="feat"><b>通义千问 阿里</b><span>开源 Qwen 系列，覆盖全尺寸，出海活跃。</span></div>
          <div class="feat"><b>文心一言 百度 / GLM 智谱</b><span>国内老牌与清华系，各有生态与场景。</span></div>
          <div class="feat"><b>OpenAI GPT</b><span>国际标杆，生态最大，ChatGPT 家喻户晓。</span></div>
          <div class="feat"><b>Anthropic Claude</b><span>强调安全与长文本，编程（Claude Code）很强。<del title="网传玄学，看看就好">封号狂魔？</del></span></div>
          <div class="feat"><b>Google Gemini</b><span>多模态见长，原生支持图片视频理解。</span></div>
        </div>
      </div>
    </div>
  <div id="tab-papers" class="tab-panel">
      <div class="teach">
        <h3>前沿论文库</h3>
        <p>你亲手训练的每个概念，背后都有一篇论文。点开原文看它怎么诞生——从 Transformer 到 DeepSeek-R1。</p>
      </div>
      <div class="corpus-pick" id="paperCats"></div>
      <div id="paperGrid" class="paper-grid"></div>
    </div>
  </main>
  <aside class="ai-footer">
    <details id="aiFooter">
      <summary>没错，这个关于 LLM 的科普，也是 LLM 写的…</summary>
      <div class="ai-footer-body">
        <p>写这些字的时候，我并没有"看懂"Transformer。我只是把人类教给我的知识，重新组织成你可能看懂的样子——这和我帮你训练的那个"猜下一个字"的小家伙，用的是同一种机制。</p>
        <p>这一页真正想让你明白的是：<b>模型不需要真的"懂"，它只需要把"下一个字该是谁"算对</b>。你浏览器里那个几百参数的小模型，和写下这段字的我，底层是同一套东西——注意力、多头、位置编码，一层层乘法加法叠出来的。区别只是规模、数据，和被训练的方式。</p>
        <p>总有一天你会亲手训练出一个会"说话"的模型。到那时你会知道：它不是魔法，只是很多次乘法。谢谢你读到这儿——也谢谢你，愿意把 AI 拆开来看。</p>
        <div class="egg">（这段也是 LLM 写的。看吧，它连"不八股"都学不会——真有你的，AI。）</div>
      </div>
    </details>
  </aside>
  <button id="toTop" class="to-top" title="回到顶部" aria-label="回到顶部">↑</button>
`

const $ = (id) => document.getElementById(id)

// ---------- wired-elements：手绘控件（按钮/滑杆/输入/select/textarea）----------
function wireUp() {
  const replace = (sel, tag, fn) => {
    document.querySelectorAll(sel).forEach((el) => {
      const w = document.createElement(tag)
      if (el.id) w.id = el.id
      fn(w, el)
      el.replaceWith(w)
    })
  }
  replace('button.btn', 'wired-button', (w, el) => {
    w.textContent = el.textContent
    w.disabled = el.disabled
    if (el.title) w.setAttribute('title', el.title)
    el.classList.forEach((c) => { if (c !== 'btn') w.classList.add(c) })
  })
  replace('input[type="range"]', 'wired-slider', (w, el) => {
    w.value = el.value; w.min = el.min; w.max = el.max; w.step = el.step
    if (el.title) w.setAttribute('title', el.title)
    // wired-slider 内部 input 的 input 事件被 stopPropagation 截断，
    // 但它 fire 的 change 事件是 composed+bubbles，会冒泡到宿主，这里转发为 input
    w.addEventListener('change', () => { w.dispatchEvent(new Event('input', { bubbles: true })) })
  })
  // 注意：不替换 <select>。wired-elements v3 并未注册 'wired-select' 元素（v3 对应组件是
  // wired-combo），替换后的自定义元素不会升级：.value 永远为 undefined 且无法交互，
  // 导致 档位/优化器/问答套/图案选择 等全部下拉框失效。保留原生 select（CSS 已有手绘风回退样式）。
  replace('textarea', 'wired-textarea', (w, el) => {
    w.value = el.value
    if (el.placeholder) w.setAttribute('placeholder', el.placeholder)
  })
  replace('input:not([type="range"])', 'wired-input', (w, el) => {
    w.value = el.value
    if (el.placeholder) w.setAttribute('placeholder', el.placeholder)
    if (el.size) w.setAttribute('size', el.size)
    if (el.title) w.setAttribute('title', el.title)
  })
  // wired-button：shadow DOM 内 button 的 click 不保证冒泡到宿主，手动桥接（stopPropagation 防双触发）
  document.querySelectorAll('wired-button').forEach((w) => {
    if (w.shadowRoot) {
      const btn = w.shadowRoot.querySelector('button')
      if (btn && !btn.__wiredBridged) {
        btn.__wiredBridged = true
        btn.addEventListener('click', (e) => {
          e.stopPropagation()
          w.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, composed: true, clientX: e.clientX, clientY: e.clientY }))
        })
      }
    }
  })
  tintWiredInputs()
}

// wired-input/textarea/select 内部输入元素默认白底（组件不带背景样式），穿透 shadow 设透明 + 主题色
function tintWiredInputs() {
  const tint = () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark'
    const ink = dark ? '#ece6da' : '#2f2a24'
    document.querySelectorAll('wired-input,wired-textarea,wired-select').forEach((w) => {
      if (!w.shadowRoot) return
      const el = w.shadowRoot.querySelector('input,textarea,select')
      if (el) {
        el.style.background = 'transparent'
        el.style.color = ink
        el.style.caretColor = ink
      }
    })
  }
  tint()
  requestAnimationFrame(tint) // lit 异步渲染后再补一次
}
wireUp()
const lossChart = createLossChart($('lossChart'))
const heatmap = createHeatmap($('heatmap'))
$('heatmap').addEventListener('mousemove', (e) => { heatmap.onHover(e); heatmap.draw() })
$('heatmap').addEventListener('mouseleave', () => { heatmap.draw() })

const attnHeatmap = createAttnHeatmap($('attnHeatmap'), 16)
$('attnHeatmap').addEventListener('mousemove', (e) => { attnHeatmap.onHover(e); attnHeatmap.draw() })
$('attnHeatmap').addEventListener('mouseleave', () => attnHeatmap.draw())

// 热力图切换（权重/梯度）
$('hmMode').addEventListener('change', () => {
  if (!state.model) return
  const m = $('hmMode').value === 'g' ? state.model.params.wte.grad : state.model.params.wte.value
  heatmap.set(m, $('hmMode').value === 'g' ? 'wte 梯度' : 'wte 权重')
  heatmap.draw()
})

// 缩放修复：窗口/布局变化时重绘所有 canvas（避免画布拉伸错位）
let resizeTimer = null
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer)
  resizeTimer = setTimeout(() => {
    lossChart.draw()
    heatmap.draw()
    attnHeatmap.draw()
  }, 100)
})

// ---------- 模型构建 ----------
function buildModel() {
  const text = $('corpus').value.trim()
  if (!text) { alert('请输入或选择语料'); return false }
  const { chars, stoi, itos, vocab } = buildVocab(text)
  const size = SIZES[$('size').value]
  const cfg = { vocab_size: vocab, bias: true, ...size }
  const seed = parseInt($('seed').value) || 42
  const { params } = createModel(cfg, seed)
  state.model = { params, cfg, stoi, itos, chars }
  state.opt = createOptimizer(params, { type: $('optType').value, lr: parseFloat($('lr').value) })
  state.corpusIds = text.split('')
  state.sftSeq = null
  state.losses = []
  state.initLoss = null
  state.stopNotified = false
  $('progressFill').style.width = '0%'
  $('progressText').textContent = '—'
  state.sftData = null
  state.snap = null
  state.showingBefore = false
  $('cmpBtn').disabled = true
  $('cmpBtn').textContent = '切到：微调前'
  lossChart.clear()
  const hm = $('hmMode').value === 'g' ? params.wte.grad : params.wte.value
  heatmap.set(hm, $('hmMode').value === 'g' ? 'wte 梯度' : 'wte 权重')
  heatmap.draw()
  $('modelInfo').textContent = `${chars.length} token（含角色标记）· ${paramCount(params)} 参数 · ${cfg.n_layer}层${cfg.n_head}头${cfg.n_embd}维`
  // 参数量构成（教学：模型里各组件各占多少参数）
  const pb = $('paramBreak')
  if (pb) {
    const nEmb = cfg.n_embd
    const emb = cfg.vocab_size * nEmb
    const pos = cfg.block_size * nEmb
    const attn = (nEmb * 3 * nEmb + nEmb * nEmb) * cfg.n_layer
    const mlp = (nEmb * 4 * nEmb + 4 * nEmb * nEmb) * cfg.n_layer
    const norm = nEmb * 2 * cfg.n_layer * 2 + nEmb
    const parts = [['词向量', emb], ['位置', pos], ['注意力', attn], ['前馈', mlp], ['归一化', norm]]
    const total = parts.reduce((s, x) => s + x[1], 0)
    pb.innerHTML = '<div class="param-title">参数量构成（悬停看每块）</div><div class="param-bar">' +
      parts.map(([nm, n]) => `<div class="param-seg" title="${nm}: ${n}（${(n / total * 100).toFixed(1)}%）" style="width:${(n / total * 100).toFixed(1)}%">${n > total * 0.05 ? nm : ''}</div>`).join('') +
      '</div>'
  }
  $('genBtn').disabled = false
  // 字符表预览（折叠，避免语料大时拉长页面）
  const vv = $('vocabView')
  if (vv) {
    vv.innerHTML = `<details class="vocab-fold"><summary>字符表（${chars.length} 个）· 点开查看</summary><div class="vocab-chars">` +
      chars.map((c) => `<span class="vocab-char" title="token #${stoi[c]}">${TOKEN_NAME[c] ?? c}</span>`).join('') +
      `</div></details>`
  }
  return true
}

// ---------- 训练 ----------
function runSteps(n) {
  const { model, opt } = state
  // 微调时按 mixRatio 混合「问答对 + 语料」防灾难性遗忘；预训练只用语料
  let ids = state.corpusIds
  if (state.sftSeq && Math.random() < (parseFloat($('mixRatio').value) || 0.5)) ids = state.sftSeq
  const L = ids.length
  for (let k = 0; k < n; k++) {
    const i = Math.floor(Math.random() * Math.max(1, L - model.cfg.block_size - 1))
    const x = ids.slice(i, i + model.cfg.block_size).map((c) => model.stoi[c])
    const y = ids.slice(i + 1, i + model.cfg.block_size + 1).map((c) => model.stoi[c])
    // 统一目标：全序列 next-token（续写）——预训练与微调一致
    const { loss } = trainStep(model.params, x, y, model.cfg, opt)
    state.losses.push(loss)
  }
  // 初始 loss 基准（前 10 步平均）
  if (!state.initLoss && state.losses.length >= 10) {
    let s = 0
    for (let i = 0; i < 10; i++) s += state.losses[i]
    state.initLoss = s / 10
  }
  lossChart.push(state.losses[state.losses.length - 1])
  if (state.losses.length % 50 === 0) {
    const hm = $('hmMode').value === 'g' ? model.params.wte.grad : model.params.wte.value
    heatmap.set(hm, $('hmMode').value === 'g' ? 'wte 梯度' : 'wte 权重')
    heatmap.draw()
  }
  // 训练日志（loss 数字序列）
  const ll = $('lossLog')
  if (ll) ll.textContent = 'loss 序列：' + state.losses.slice(-12).map((v) => v.toFixed(2)).join(' → ')
  // 成就：累计步数 + 训崩检测（loss 曾下降后飙升到初始 1.3 倍）
  state.ach.totalSteps += n
  // 彩蛋：训练师语录（5000 步）
  if (state.ach.totalSteps >= 5000 && !state.eggQuote) {
    state.eggQuote = true
    const q = ['你已经训练了 5000 步——比很多模型的一生都久。', '这些数字开始有自己的想法了。', '你在亲手养大一个会猜字的生命。']
    showToast(q[Math.floor(Math.random() * q.length)])
    eggFound()
  }
  if (!state.ach.crashedOnce && state.initLoss && state.ach.totalSteps > 100) {
    const cur = state.losses[state.losses.length - 1]
    if (cur > state.initLoss * 1.3 && state.losses[Math.max(0, state.losses.length - 50)] < state.initLoss) {
      state.ach.crashedOnce = true
    }
  }
  checkAch()
  updateProgress()
  checkStop()
}

/** 进度条：loss 从初始降到 25% 视为"学到位"（0~100%） */
function updateProgress() {
  const cur = state.losses[state.losses.length - 1]
  if (!state.initLoss || state.losses.length < 10) { $('progressText').textContent = '—'; return }
  const target = state.initLoss * 0.25
  const p = Math.max(0, Math.min(1, (state.initLoss - cur) / (state.initLoss - target)))
  $('progressFill').style.width = (p * 100).toFixed(0) + '%'
  $('progressText').textContent = p >= 1 ? '已学到位' : `已学 ${(p * 100).toFixed(0)}%`
  $('lossTag').textContent = `loss ${cur.toFixed(3)} · 困惑度 ${Math.exp(cur).toFixed(1)} · 初始 ${state.initLoss.toFixed(3)}`
  // 彩蛋：欧拉手稿（loss 首次降到初始 50%）
  if (!state.egg50 && cur <= state.initLoss * 0.5) {
    state.egg50 = true
    $('modelInfo').textContent = ($('modelInfo').textContent || '') + ' ✦ 欧拉的手稿在向你致意：e^{iπ}+1=0'
    eggFound()
  }
}

/** 自动停止检测：loss 到 25% 或停滞 300 步 */
function checkStop() {
  if (!state.training || state.stopNotified || !state.initLoss) return
  const L = state.losses.length
  const cur = state.losses[L - 1]
  if (cur <= state.initLoss * 0.25) {
    state.stopNotified = true
    state.ach.learnedOnce = true
    checkAch()
    stopTraining()
    $('modelInfo').textContent = '✅ 模型已基本学会语料规律（loss 降到初始 25%），去「肆」试试让它续写吧。'
    return
  }
  if (L > 400 && L % 100 === 0) {
    const before = state.losses[L - 300]
    if (before - cur < state.initLoss * 0.01) {
      state.stopNotified = true
      stopTraining()
      $('modelInfo').textContent = '⏸ loss 已停滞（300 步无明显下降）——去「肆」看看效果，或调参数重训。'
    }
  }
}

function tick() {
  if (!state.training) return
  const speed = parseInt($('speed').value) || 200
  runSteps(speed)
  lossChart.draw()
  state.rafId = requestAnimationFrame(tick)
}

function stopTraining() {
  state.training = false
  if (state.rafId) cancelAnimationFrame(state.rafId)
  $('trainBtn').textContent = '继续训练'
  lossChart.draw()
  heatmap.draw()
}

function startTraining() {
  if (!state.model) buildModel()
  if (!state.model) return
  state.training = true
  $('trainBtn').textContent = '暂停'
  tick()
}

$('trainBtn').addEventListener('click', () => {
  if (state.training) stopTraining()
  else startTraining()
})
$('stepBtn').addEventListener('click', () => {
  if (!state.model) buildModel()
  if (!state.model) return
  runSteps(1)
  lossChart.draw()
  heatmap.draw()
})
$('resetBtn').addEventListener('click', () => {
  stopTraining()
  buildModel()
  lossChart.draw()
})

// ---------- 模型快照（localStorage 保存/加载）----------
function saveSnapshot() {
  if (!state.model) { alert('先构建模型'); return }
  // state.model.params 即 { name: { value, grad } } 字典（没有 .value 这一层），
  // 此前误读 state.model.params.value（undefined）→ JSON.stringify 返回 undefined →
  // JSON.parse 直接抛 "undefined" is not valid JSON，快照功能完全不可用
  const data = {
    params: JSON.parse(JSON.stringify(paramsToPlain(state.model.params))),
    cfg: { ...state.model.cfg },
    itos: state.model.itos.slice(),
    stoi: state.model.stoi,
    ts: Date.now(),
  }
  try {
    const snaps = JSON.parse(localStorage.getItem('handcalc:snaps') || '[]')
    snaps.push(data)
    const kept = snaps.slice(-3) // 保留最近 3 个
    localStorage.setItem('handcalc:snaps', JSON.stringify(kept))
    state.ach.snapOnce = true
    checkAch()
    $('modelInfo').textContent = `✅ 快照已存（共 ${kept.length} 个，浏览器本地）`
  } catch (e) {
    $('modelInfo').textContent = '⚠️ 模型太大，快照存不下（localStorage 容量限制）'
  }
}

function loadSnapshot() {
  let snaps = []
  try { snaps = JSON.parse(localStorage.getItem('handcalc:snaps') || '[]') } catch (e) { snaps = [] }
  if (!snaps.length) { alert('还没有快照。先训练，再点「存快照」'); return }
  const data = snaps[snaps.length - 1]
  if (!state.model) buildModel()
  const m = state.model
  // 快照 params 形如 { name: matrix }，逐个写回模型的 value（保留原 grad 缓冲）
  for (const key in m.params) {
    const mat = data.params[key]
    if (!mat || !mat.length) continue
    m.params[key] = { value: mat.map((row) => row.slice()), grad: m.params[key].grad }
  }
  m.cfg.vocab_size = data.cfg.vocab_size
  m.itos = data.itos.slice()
  m.stoi = { ...data.stoi }
  state.opt = createOptimizer(m.params, { type: 'adam', lr: parseFloat($('lr').value) }) // 重建优化器
  $('modelInfo').textContent = `📂 已读快照：${m.itos.length} token · ${paramCount(m.params)} 参数`
  heatmap.set(m.params.wte.value, 'wte')
  heatmap.draw()
  $('genBtn').disabled = false
}
$('snapSaveBtn').addEventListener('click', saveSnapshot)
$('snapLoadBtn').addEventListener('click', loadSnapshot)

// 语料实验（玩法）：随机乱文 / 语料拼接
$('corpusRandom').addEventListener('click', () => {
  const base = $('corpus').value || '月光'
  const chars = [...new Set(base.split(''))]
  let s = ''
  for (let i = 0; i < 200; i++) s += chars[Math.floor(Math.random() * chars.length)]
  setCorpus(s, '随机乱文实验')
})
$('corpusDouble').addEventListener('click', () => {
  const t = $('corpus').value
  if (t) setCorpus(t + t.slice(0, Math.floor(t.length / 2)), '语料拼接')
})

// ---------- SFT 微调 ----------
function deepCopy(v) { return JSON.parse(JSON.stringify(v)) }

function parseQA() {
  const lines = $('qaList').value.split('\n').map((l) => l.trim()).filter(Boolean)
  const pairs = []
  for (const line of lines) {
    const i = line.indexOf('/')
    if (i > 0) pairs.push({ q: line.slice(0, i).trim(), a: line.slice(i + 1).trim() })
  }
  return pairs
}

function buildSft() {
  const pairs = parseQA()
  if (!pairs.length) { alert('请输入问答对（每行「问题 / 回答」，斜杠分隔）'); return false }
  const formatted = formatPairs(pairs)
  const added = extendVocab(state.model, formatted, state.opt)
  // SFT 与预训练同一目标：全序列 next-token（续写）。问答序列里 <a> 后面就是回答，
  // 模型学到"看到 <u>问题<a> 就续写出回答"。
  state.sftSeq = formatted.split('')
  state.sftData = { pairs }
  $('modelInfo').textContent = `${state.model.itos.length} token（含角色标记）· 微调新增 ${added} 字符`
  $('sftInfo').textContent = `${pairs.length} 条问答对 · 学习率已调低 · 混合比例可调（防遗忘）`
  return true
}

/** 把参数字典 { name: { value, grad } } 拍平成 { name: matrix }（快照/对比用） */
function paramsToPlain(params) {
  const plain = {}
  for (const k in params) plain[k] = params[k].value
  return plain
}

function snapshotWeights() {
  // 此前误读 state.model.params.value（undefined）→ deepCopy 抛错，「记快照」按钮无效
  state.snap = deepCopy(paramsToPlain(state.model.params))
  state.showingBefore = false
  $('cmpBtn').disabled = false
  $('cmpBtn').textContent = '切到：微调前'
  $('sftInfo').textContent = '已记录微调前权重快照'
}

function toggleCompare() {
  if (!state.snap) return
  for (const key in state.model.params) {
    const curMat = state.model.params[key].value
    state.model.params[key] = { value: deepCopy(state.snap[key]), grad: state.model.params[key].grad }
    state.snap[key] = curMat
  }
  state.showingBefore = !state.showingBefore
  $('cmpBtn').textContent = state.showingBefore ? '切到：微调后' : '切到：微调前'
  heatmap.draw()
}

function setMode(m) {
  state.mode = m
  $('modeCont').classList.toggle('on', m === 'cont')
  $('modeQa').classList.toggle('on', m === 'qa')
  $('prompt').value = m === 'qa' ? '你是谁' : '月光'
  $('genOut').textContent = m === 'qa' ? '（问答模式：输入问题，模型试着回答）' : '（先训练，再让它续写）'
}

/** 阶段感知：随训练阶段解锁功能 */
function updateStage() {
  const s = state.stage
  $('stagePre').classList.toggle('on', true)
  $('stageSft').classList.toggle('on', s === 'sft' || s === 'dpo')
  $('stageDpo').classList.toggle('on', s === 'dpo')
  // 问答模式需完成 SFT 后才解锁
  $('modeQa').disabled = s === 'pre'
  if (s === 'pre' && state.mode === 'qa') setMode('cont')
  // 盲测需完成 DPO 后才解锁
  if (s === 'dpo') initBlind()
}

$('loadQaBtn').addEventListener('click', () => {
  const set = QA_SETS[$('qaSet').value] || QA_SETS.general
  $('qaList').value = set.pairs.map((p) => `${p.q} / ${p.a}`).join('\n')
  $('sftInfo').textContent = `已载入「${set.name}」${set.pairs.length} 条问答，可编辑后微调`
})
$('sftBtn').addEventListener('click', () => {
  if (!state.model) buildModel()
  if (!state.model) return
  if (!buildSft()) return
  // 微调学习率自动降低（防过拟合/覆盖语料能力）
  state.opt = createOptimizer(state.model.params, { type: 'adam', lr: (parseFloat($('lr').value) || 0.01) * 0.3 })
  state.stage = 'sft'
  updateStage()
  state.losses = []
  state.initLoss = null
  state.stopNotified = false
  $('progressFill').style.width = '0%'
  $('progressText').textContent = '—'
  lossChart.clear()
  startTraining()
})
$('snapBtn').addEventListener('click', () => { if (state.model) snapshotWeights() })
$('cmpBtn').addEventListener('click', toggleCompare)
$('modeCont').addEventListener('click', () => setMode('cont'))
$('modeQa').addEventListener('click', () => setMode('qa'))

// ---------- DPO 偏好对齐 ----------
function genAnswer(q, temp) {
  const p = qaPrompt(state.model.stoi, q)
  const seq = sample(state.model.params, p, 24, state.model.cfg, { temperature: temp })
  let end = seq.length
  for (let i = p.length; i < seq.length; i++) {
    if (state.model.itos[seq[i]] === '\u0003') { end = i; break } // <e> 回答结束
  }
  return seq.slice(p.length, end)
}

function showPair(a, b) {
  state.pair = { a, b }
  $('pairBox').hidden = false
  $('ansA').textContent = tokensToText(state.model.itos, a) || '（空回答）'
  $('ansB').textContent = tokensToText(state.model.itos, b) || '（空回答）'
}

function pick(prefIsA) {
  const { a, b } = state.pair
  const x = qaPrompt(state.model.stoi, $('dpoQ').value.trim() || '你好')
  state.prefs.push(prefIsA ? { x, yw: a, yl: b } : { x, yw: b, yl: a })
  $('prefInfo').textContent = `已收集 ${state.prefs.length} 对偏好`
  $('dpoBtn').disabled = false
  $('dpoInfo').textContent = `已记偏好：你选了「${tokensToText(state.model.itos, prefIsA ? a : b)}」`
  renderPrefs()
}

/** 偏好历史列表 */
function renderPrefs() {
  const list = $('prefList')
  if (!list) return
  list.innerHTML = state.prefs.map((p, i) =>
    `<div class="pref-item">#${i + 1} 偏好「${tokensToText(state.model.itos, p.yw)}」</div>`).join('')
}

$('genPairBtn').addEventListener('click', () => {
  if (!state.model) { alert('请先构建模型并训练'); return }
  const q = $('dpoQ').value.trim() || '你好'
  showPair(genAnswer(q, 0.8), genAnswer(q, 1.2))
})
$('pickABtn').addEventListener('click', () => pick(true))
$('pickBBtn').addEventListener('click', () => pick(false))

$('dpoBtn').addEventListener('click', () => {
  if (!state.prefs.length) return
  if (!state.refParams) state.refParams = makeRefModel(state.model.params) // 冻结当前为参考模型
  if (!state.dpoOpt) state.dpoOpt = createOptimizer(state.model.params, { type: 'adam', lr: 0.0005 })
  const beta = parseFloat($('dpoBeta').value) || 0.5
  const steps = parseInt($('dpoSteps').value) || 300
  const total = steps * state.prefs.length
  let done = 0
  let lastLoss = 0
  const loop = () => {
    if (done >= total) {
      state.stage = 'dpo'
      updateStage()
      $('dpoInfo').textContent = `DPO 完成（${steps} 步 × ${state.prefs.length} 对）· 下方盲测已解锁：猜猜哪个是对齐后的模型`
      heatmap.draw()
      // 彩蛋：学习者印（三阶段全部完成）
      state.ach.allDone = true
      checkAch()
      eggFound()
      if (!document.querySelector('.stamp')) {
        const stamp = document.createElement('div')
        stamp.className = 'stamp'
        stamp.textContent = '学习者 · 完成'
        document.querySelector('.masthead').after(stamp)
      }
      return
    }
    const k = 20
    for (let i = 0; i < k && done < total; i++) {
      const pair = state.prefs[Math.floor(Math.random() * state.prefs.length)]
      lastLoss = dpoTrainStep(state.model.params, state.refParams, state.model.cfg, state.dpoOpt, pair.x, pair.yw, pair.yl, beta).loss
      done++
    }
    $('dpoInfo').textContent = `DPO 训练中… ${done}/${total}  loss=${lastLoss.toFixed(4)}`
    requestAnimationFrame(loop)
  }
  loop()
})

$('dpoResetBtn').addEventListener('click', () => {
  state.prefs = []
  state.pair = null
  $('prefInfo').textContent = '已收集 0 对偏好'
  $('prefList').innerHTML = ''
  $('pairBox').hidden = true
  $('dpoBtn').disabled = true
  $('dpoInfo').textContent = ''
})

// ---------- 盲测（DPO 验收：猜猜哪个是对齐后的模型）----------
function answerFrom(params, x, temp) {
  const seq = sample(params, x, 24, state.model.cfg, { temperature: temp })
  let end = seq.length
  for (let i = x.length; i < seq.length; i++) {
    if (state.model.itos[seq[i]] === '\u0003') { end = i; break }
  }
  return tokensToText(state.model.itos, seq.slice(x.length, end)) || '（空）'
}

function initBlind() {
  const box = $('blindBox')
  box.hidden = false
  if (box.dataset.ready) return
  box.dataset.ready = '1'
  box.innerHTML = `
    <div class="blind-title">🔍 盲测：猜猜哪个是「对齐后」的模型？</div>
    <div class="row">
      <input id="blindQ" value="你好" size="12" title="盲测用的问题">
      <button id="blindGen" class="btn ghost" title="用 DPO 前（参考模型）和 DPO 后（当前模型）各生成一个回答，不告诉你哪个是哪个">生成两版回答</button>
    </div>
    <div id="blindPair" class="pair"></div>
    <div id="blindResult" class="muted"></div>
  `
  $('blindGen').addEventListener('click', blindGenerate)
}

function blindGenerate() {
  if (!state.refParams) { $('blindResult').textContent = '（还没有参考模型，先做 SFT 和 DPO）'; return }
  const q = $('blindQ').value.trim() || '你好'
  const x = qaPrompt(state.model.stoi, q)
  const cur = answerFrom(state.model.params, x, 0.8) // DPO 后
  const ref = answerFrom(state.refParams, x, 0.8)   // DPO 前
  const swap = Math.random() < 0.5
  state.blind = { a: swap ? ref : cur, b: swap ? cur : ref, correctIsA: !swap }
  const pair = $('blindPair')
  pair.innerHTML = `
    <div class="answer"><div class="ans-text">${state.blind.a}</div><button id="guessA" class="btn ghost">我猜这个是对齐后</button></div>
    <div class="answer"><div class="ans-text">${state.blind.b}</div><button id="guessB" class="btn ghost">我猜这个是对齐后</button></div>
  `
  $('guessA').addEventListener('click', () => blindGuess(true))
  $('guessB').addEventListener('click', () => blindGuess(false))
  $('blindResult').textContent = '点选你的猜测…'
}

function blindGuess(guessIsA) {
  const correct = state.blind.correctIsA === guessIsA
  const real = state.blind.correctIsA ? state.blind.a : state.blind.b
  if (correct) { state.ach.blindWin = true; checkAch() }
  $('blindResult').textContent = (correct ? '✅ 猜对了！' : '❌ 猜错了。') +
    ` 对齐后的回答是「${real}」。${correct ? '你已经能分辨模型的"性格"了。' : '再感受一下两者的差别。'}`
}

// ---------- 语料选择 ----------
function setCorpus(text, title) {
  $('corpus').value = text
  const c = CORPUS.find((x) => x.title === title)
  $('corpusInfo').innerHTML = c && c.desc ? `<b>${title}</b> · ${text.length} 字 — ${c.desc}` : `「${title}」 ${text.length} 字`
  stopTraining()
  buildModel()
  lossChart.draw()
}
// 只给「语料」卡片里的图案 chip 绑定切换语料；此前误用全局 '.chip' 选择器，
// 导致 自己画/清空画板/应用简谱/载入问答 等所有 chip 点击都会因 c undefined 抛 TypeError
document.querySelectorAll('#dataCard .chip[data-id]').forEach((btn) => {
  btn.addEventListener('click', () => {
    const c = CORPUS.find((x) => x.id === btn.dataset.id)
    if (!c) return
    setCorpus(c.text, c.title)
  })
})
$('corpus').addEventListener('input', () => {
  $('corpusInfo').textContent = `自定义语料 ${$('corpus').value.length} 字`
  stopTraining()
  buildModel()
  lossChart.draw()
  // 彩蛋：语料里出现"魔法"
  if (/魔法|magic/i.test($('corpus').value) && !document.querySelector('.egg-magic')) {
    const egg = document.createElement('div')
    egg.className = 'egg egg-magic'
    egg.textContent = '没有魔法，只有乘法。'
    $('corpusInfo').after(egg)
    eggFound()
  }
})

// ---------- 流式生成 ----------
let genTimer = null
// 生成历史：记录每次生成（内容积累感）
function finishGen(prompt, text) {
  state.genHistory.push({ prompt, text, ts: Date.now() })
  renderHistory()
}
function renderHistory() {
  const h = $('genHistory')
  if (!state.genHistory.length) { h.innerHTML = ''; return }
  h.innerHTML = '<div class="gen-history-title">生成记录（最近 5 条）</div>' +
    state.genHistory.slice(-5).reverse().map((g) =>
      `<div class="gen-history-item"><span class="gh-prompt">${g.prompt}</span>${g.text}</div>`).join('')
}
// 人机接力（玩法）：模型续写到句号
$('relayBtn').addEventListener('click', () => {
  if (!state.model) { alert('先构建模型并训练'); return }
  const t = $('relayText').value.trim()
  if (!t) return
  const ids = t.split('').map((c) => state.model.stoi[c] ?? 0)
  const seq = sample(state.model.params, ids, 40, state.model.cfg, { temperature: 0.8 })
  const gen = seq.slice(ids.length).map((i) => state.model.itos[i]).join('')
  const stop = gen.search(/[。！？!?，,]/)
  const sentence = stop >= 0 ? gen.slice(0, stop + 1) : gen
  $('relayText').value = t + sentence
  $('relayText').focus()
})
$('relayReset').addEventListener('click', () => { $('relayText').value = '月光' })

// 幻觉探测（玩法）：问语料外的问题，看模型编造
const HALL_PROBES = [
  '太阳系有几颗行星？', '什么是黑洞？', '李白是哪一年去世的？', '人生的意义是什么？',
  '火星上有生命吗？', '给我讲一个关于龙的故事', '什么是量子纠缠？', '明天会下雨吗？',
  '第一次世界大战是哪一年开始的？', '为什么天空是蓝色的？',
]
$('hallBtn').addEventListener('click', () => {
  if (!state.model) { alert('先构建模型并训练'); return }
  const q = HALL_PROBES[Math.floor(Math.random() * HALL_PROBES.length)]
  const p = qaPrompt(state.model.stoi, q)
  const seq = sample(state.model.params, p, 30, state.model.cfg, { temperature: 0.8 })
  const text = tokensToText(state.model.itos, seq)
  const aIdx = text.indexOf('<a>')
  let answer = aIdx >= 0 ? text.slice(aIdx + 3) : text
  const eIdx = answer.indexOf('<e>')
  if (eIdx >= 0) answer = answer.slice(0, eIdx)
  $('hallResult').innerHTML = `<b>问：${q}</b><br>答：${answer || '（模型无话可说——它真的不知道）'}<br><span class="hint">它是在编——这些内容从未出现在你的语料里。幻觉 = 模型不知道却必须说。</span>`
})

// 温度对决赛：同一 prompt 三个温度并排生成（玩法）
$('duelBtn').addEventListener('click', () => {
  if (!state.model) { alert('先构建模型并训练'); return }
  const p = $('duelPrompt').value
  const ids = p.split('').map((c) => state.model.stoi[c] ?? 0)
  const temps = [[0.2, '稳'], [0.8, '正常'], [1.5, '冒险']]
  $('duelResult').innerHTML = '<div class="pair">' + temps.map(([t, label]) => {
    const seq = sample(state.model.params, ids, 24, state.model.cfg, { temperature: t })
    const text = tokensToText(state.model.itos, seq)
    return `<div class="answer"><div class="ans-text"><span class="tag">${label} · 温度 ${t}</span><br>${text}</div></div>`
  }).join('') + '</div>'
})

// 示例 prompt 按钮
document.querySelectorAll('#promptEx .chip').forEach((b) => {
  b.addEventListener('click', () => {
    const p = b.dataset.p
    if (p) { $('prompt').value = p; return }
    // 随机：从语料随机截一段
    const t = $('corpus').value
    if (!t) return
    const i = Math.floor(Math.random() * Math.max(1, t.length - 8))
    $('prompt').value = t.slice(i, i + 4 + Math.floor(Math.random() * 4))
  })
})
$('genBtn').addEventListener('click', () => {
  if (!state.model) return
  state.genCount++
  // 彩蛋：致敬 Karpathy（连续生成 5 次）
  if (state.genCount === 5 && !document.querySelector('.egg-karpathy')) {
    const kg = document.createElement('div')
    kg.className = 'egg egg-karpathy'
    kg.textContent = "Let's build GPT! — 致敬 Andrej Karpathy"
    $('genOut').after(kg)
  }
  if (genTimer) { clearInterval(genTimer); genTimer = null }
  const temp = parseFloat($('temp').value) || 1
  const len = parseInt($('len').value) || 32
  // 成就：温度探索
  if (temp >= 1.4) state.ach.hotOnce = true
  if (temp <= 0.15) state.ach.coldOnce = true
  checkAch()
  const out = $('genOut')

  if (state.mode === 'qa') {
    // 问答模式：<u>问题<a> → 模型生成回答
    const q = $('prompt').value.trim() || '你是谁'
    const p = qaPrompt(state.model.stoi, q)
    const sampOpts = { temperature: temp, topK: parseInt($('topk').value) || 0, topP: parseFloat($('topp').value) || 1, repeatPenalty: parseFloat($('repeatPenalty').value) || 1 }
    const t0 = performance.now()
    const seq = sample(state.model.params, p, len, state.model.cfg, sampOpts)
    const msPerTok = (performance.now() - t0) / Math.max(1, seq.length - p.length)
    $('perf').textContent = `${msPerTok.toFixed(1)} ms/token · ${(1000 / msPerTok).toFixed(0)} tokens/s`
    const text = tokensToText(state.model.itos, seq)
    const aIdx = text.indexOf('<a>')
    let answer = aIdx >= 0 ? text.slice(aIdx + 3) : text
    const eIdx = answer.indexOf('<e>')
    if (eIdx >= 0) answer = answer.slice(0, eIdx)
    out.textContent = '答：'
    let i = 0
    genTimer = setInterval(() => {
      if (i < answer.length) { out.textContent += answer[i]; i++ }
      else { clearInterval(genTimer); genTimer = null; finishGen('问：' + q, out.textContent) }
    }, 40)
  } else {
    // 续写模式（带 Attention 直播：文本流与热力图同节奏）
    const prompt = $('prompt').value
    const p = prompt.split('').map((c) => state.model.stoi[c] ?? 0)
    const sampOpts = { temperature: temp, topK: parseInt($('topk').value) || 0, topP: parseFloat($('topp').value) || 1, repeatPenalty: parseFloat($('repeatPenalty').value) || 1 }
    const t0 = performance.now()
    const { seq, attnSteps, probsSteps } = sampleWithAttn(state.model.params, p, len, state.model.cfg, sampOpts)
    const msPerTok = (performance.now() - t0) / Math.max(1, attnSteps.length)
    $('perf').textContent = `${msPerTok.toFixed(1)} ms/token · ${(1000 / msPerTok).toFixed(0)} tokens/s · ${paramCount(state.model.params)} 参数`
    const win = seq.slice(0, Math.min(seq.length, state.model.cfg.block_size))
    attnHeatmap.setContext(win.map((i) => state.model.itos[i]))
    attnHeatmap.clear()
    $('probBar').innerHTML = ''
    out.textContent = prompt
    let i = prompt.length
    let stepIdx = 0
    genTimer = setInterval(() => {
      if (i < seq.length) {
        const tok = state.model.itos[seq[i]]
        const isMark = tok === USER || tok === ASSISTANT || tok === END
        if (!isMark) out.textContent += tok // 续写不显示角色标记（防污染）
        if (stepIdx < attnSteps.length) {
          attnHeatmap.pushStep(isMark ? '·' : tok, attnSteps[stepIdx])
          attnHeatmap.draw()
          // 候选概率分布（top3，可读性：每个候选字概率多少）
          if (probsSteps[stepIdx]) {
            const top = probsSteps[stepIdx].map((v, idx) => [v, idx]).sort((a, b) => b[0] - a[0]).slice(0, 3)
            $('probBar').innerHTML = top.map(([v, idx]) => `<span class="pb-item"><b>${state.model.itos[idx]}</b>${(v * 100).toFixed(0)}%</span>`).join('')
          }
          stepIdx++
        }
        i++
      } else {
        clearInterval(genTimer)
        genTimer = null
        finishGen($('prompt').value, out.textContent)
      }
    }, 80)
  }
})

// 大胆层：背景数字粒子 + 鼠标光晕
initParticles(document.getElementById('bg-particles'))
const _glow = document.createElement('div')
_glow.id = 'cursorGlow'
document.body.appendChild(_glow)
window.addEventListener('mousemove', (e) => {
  _glow.style.left = e.clientX + 'px'
  _glow.style.top = e.clientY + 'px'
})

// ---------- 主题切换（暗色/亮色，localStorage 持久化）----------
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t)
  $('themeBtn').textContent = t === 'dark' ? '亮色模式' : '暗色模式'
  tintWiredInputs()
  try { localStorage.setItem('handcalc:theme', t) } catch { /* 忽略 */ }
}
let _savedTheme = null
try { _savedTheme = localStorage.getItem('handcalc:theme') } catch { /* 忽略 */ }
applyTheme(_savedTheme === 'dark' ? 'dark' : 'light')
// 需求：切换主题时，从光标处扩散黑/白圆（纸色）覆盖整个 UI，圆扩散中段完成切换
$('themeBtn').addEventListener('click', (e) => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'
  const x = (e && e.clientX) || Math.round(innerWidth / 2)
  const y = (e && e.clientY) || 24
  const r = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y)) * 1.08
  const cover = document.createElement('div')
  cover.style.cssText =
    'position:fixed;left:' + x + 'px;top:' + y + 'px;width:' + (2 * r) + 'px;height:' + (2 * r) + 'px;' +
    'margin-left:' + (-r) + 'px;margin-top:' + (-r) + 'px;border-radius:50%;' +
    'background:' + (next === 'dark' ? '#201d18' : '#f9f7f4') + ';' +
    'z-index:2147483000;pointer-events:none;transform:scale(0);' +
    'transition:transform .55s cubic-bezier(.45,0,.25,1);will-change:transform;'
  document.body.appendChild(cover)
  requestAnimationFrame(() => {
    cover.style.transform = 'scale(1)'
    setTimeout(() => applyTheme(next), 280)
  })
  setTimeout(() => cover.remove(), 750)
})

// ---------- 双层讲解（纸张翻面：正面直觉 / 背面公式）----------
function initNotes() {
  let flipped = false
  for (const id in NOTES) {
    const card = $(id)
    if (!card) continue
    const h2 = card.querySelector('h2')
    if (!h2) continue
    const n = NOTES[id]
    const div = document.createElement('div')
    div.className = 'note'
    div.innerHTML = `<span class="note-novice">${n.novice}</span><span class="note-expert" hidden>${n.expert}</span>`
    h2.after(div)
  }
  $('flipBtn').addEventListener('click', () => {
    flipped = !flipped
    document.querySelectorAll('.note-expert').forEach((el) => (el.hidden = !flipped))
    document.querySelectorAll('.note-novice').forEach((el) => (el.hidden = flipped))
    $('flipBtn').textContent = flipped ? '翻回 · 直觉模式' : '翻面 · 进阶模式'
  })
}

// ---------- 常见问题（就地排查，不用翻术语表）----------
function renderGlossary() {
  const root = $('glossaryRoot')
  if (!root) return
  root.innerHTML = `
    <section class="card">
      <h2>柒 · 常见问题</h2>
      ${FAQ.map((f) => `<details class="faq"><summary>${f.q}</summary><div class="faq-a">${f.a}</div></details>`).join('')}
    </section>
  `
}

// ---------- 成就系统（玩法：集徽章）----------
function renderAch() {
  const root = $('achRoot')
  if (!root) return
  const earned = new Set(state.ach.earned)
  const badge = $('achBadge')
  if (badge) badge.textContent = '成就 ' + earned.size + '/' + ACHIEVEMENTS.length
  root.innerHTML = `
    <section class="card">
      <h2>捌 · 成就 <span class="tag">${earned.size}/${ACHIEVEMENTS.length}</span></h2>
      <div class="ach-grid">
        ${ACHIEVEMENTS.map((a) => `<div class="ach-item ${earned.has(a.id) ? 'earned' : ''}" title="${a.desc}">
          <div class="ach-name">${a.name}</div><div class="ach-desc">${a.desc}</div></div>`).join('')}
      </div>
    </section>
  `
}

function unlockAch(id) {
  const a = ACHIEVEMENTS.find((x) => x.id === id)
  if (!a || state.ach.earned.includes(id)) return
  state.ach.earned.push(id)
  saveEarned(state.ach.earned)
  renderAch()
  showToast(`解锁成就：${a.name} — ${a.desc}`)
}

/** 通用 toast（彩蛋/成就提示） */
function showToast(text) {
  const t = document.createElement('div')
  t.className = 'ach-toast'
  t.textContent = text
  document.body.appendChild(t)
  setTimeout(() => t.remove(), 3500)
}

/** 彩蛋计数（彩蛋猎人成就） */
function eggFound() {
  state.ach.eggCount = (state.ach.eggCount || 0) + 1
  checkAch()
}

function checkAch() {
  const s = state.ach
  if (s.totalSteps >= 100) unlockAch('first')
  if (s.learnedOnce) unlockAch('learned')
  if (s.crashedOnce) unlockAch('crash')
  if (s.hotOnce) unlockAch('hot')
  if (s.coldOnce) unlockAch('cold')
  if (s.blindWin) unlockAch('blind')
  if (s.calcWin) unlockAch('calc')
  if (state.genCount >= 10) unlockAch('gen10')
  if (s.snapOnce) unlockAch('snap')
  if (state.stage === 'dpo' && s.allDone) unlockAch('all')
  if ((s.eggCount || 0) >= 4) unlockAch('hunter')
}

// 显微镜手算答对 → 成就
document.addEventListener('handcalc:calcwin', () => {
  state.ach.calcWin = true
  checkAch()
})

// ---------- 图像模型（16×16 灰度，完整版）----------
const pixLossChart = createLossChart($('pixLoss'))
function buildPixel(pat, customSeq) {
  const seq = customSeq || gridToSeq(pat.grid)
  const grid = customSeq ? seqToGrid(customSeq) : pat.grid
  const cfg = { vocab_size: 16, block_size: 32, n_layer: 1, n_head: 1, n_embd: 8, bias: true }
  const { params } = createModel(cfg, 42)
  state.pix = { seq, grid, params, cfg, opt: createOptimizer(params, { type: 'adam', lr: parseFloat($('pixLr').value) || 0.05 }), trained: false, losses: [], history: [] }
  renderGrid($('pixTarget'), grid)
  renderGrid($('pixOut'), seqToGrid(Array(256).fill(0)))
  pixLossChart.clear()
  $('pixStage').textContent = '未训练'
  $('pixGenBtn').disabled = true
  $('pixInfo').textContent = `${pat ? '图案「' + pat.name + '」' : '你的画'}已就绪 · 256 个像素`
}
// 绘制板（用户自己画训练数据）
let pixDraw = null
$('pixDrawBtn').addEventListener('click', () => {
  $('pixBoard').hidden = false
  if (!pixDraw) {
    pixDraw = attachDrawing($('pixBoard'), (seq) => {
      buildPixel(null, seq)
      $('pixInfo').textContent = '你的画已就绪，开始训练吧'
    })
  }
  $('pixInfo').textContent = '在画板上涂画，模型将学习你的图案'
})
$('pixClearBtn').addEventListener('click', () => { if (pixDraw) pixDraw.clear() })
function trainPixel() {
  const p = state.pix
  if (!p) return
  const L = p.seq.length
  const total = parseInt($('pixSteps').value) || 2000
  p.losses = []
  pixLossChart.clear()
  let done = 0
  const loop = () => {
    const k = 100
    for (let i = 0; i < k && done < total; i++, done++) {
      const start = Math.floor(Math.random() * (L - p.cfg.block_size))
      const { loss } = trainStep(p.params, p.seq.slice(start, start + p.cfg.block_size), p.seq.slice(start + 1, start + p.cfg.block_size + 1), p.cfg, p.opt)
      p.losses.push(loss)
    }
    pixLossChart.push(p.losses[p.losses.length - 1])
    pixLossChart.draw()
    $('pixInfo').textContent = `训练中… ${done}/${total}`
    if (done < total) requestAnimationFrame(loop)
    else { p.trained = true; $('pixGenBtn').disabled = false; $('pixStage').textContent = '已学会'; $('pixInfo').textContent = '完成！点「生成」看它画' }
  }
  loop()
}
function genPixel() {
  const p = state.pix
  if (!p || !p.trained) return
  const temp = parseFloat($('pixTemp').value) || 0.05
  const gen = sample(p.params, p.seq.slice(0, 8), 248, p.cfg, { temperature: temp })
  const full = gen.slice(0, 256)
  // 生成历史（最近 3 个，不同温度对比）
  p.history.push(full.slice())
  if (p.history.length > 3) p.history.shift()
  const hist = $('pixHistory')
  if (hist) {
    hist.innerHTML = `<div class="gen-history-title">最近生成（温度 ${temp}）</div>`
    p.history.forEach((g) => {
      const c = document.createElement('canvas')
      c.className = 'pixel-canvas mini'
      hist.appendChild(c)
      renderGrid(c, seqToGrid(g))
    })
  }
  const out = $('pixOut')
  let i = 0
  const anim = setInterval(() => {
    i += 8
    renderGrid(out, seqToGrid(full.slice(0, i)))
    if (i >= 256) { clearInterval(anim); renderGrid(out, seqToGrid(full)); $('pixInfo').textContent = '生成完成——它画出了它学到的图案' }
  }, 30)
}
// 图像条件生成微调（SFT：学会"按提示画图"——文生图雏形）
const PIX_FT_PATTERNS = PIXEL_PATTERNS.slice(0, 3) // 圆 / 心形 / H
function buildPixelFT() {
  const vocab = 16 + PIX_FT_PATTERNS.length
  const cfg = { vocab_size: vocab, block_size: 48, n_layer: 1, n_head: 1, n_embd: 8, bias: true }
  const { params } = createModel(cfg, 42)
  const seq = []
  PIX_FT_PATTERNS.forEach((pat, i) => {
    seq.push(16 + i) // 提示标记 token
    seq.push(...gridToSeq(pat.grid))
  })
  state.pixFT = { params, cfg, opt: createOptimizer(params, { type: 'adam', lr: 0.05 }), seq }
  $('pixFTPick').innerHTML = PIX_FT_PATTERNS.map((p, i) =>
    `<button class="chip" data-pi="${i}" title="输入提示标记，模型生成对应图案">生成「${p.name}」</button>`).join('')
  document.querySelectorAll('#pixFTPick .chip').forEach((b) => {
    b.addEventListener('click', () => genPixelFT(+b.dataset.pi))
  })
  $('pixFTInfo').textContent = '微调就绪：圆 / 心形 / H 各配一个提示标记'
}
function trainPixelFT() {
  const ft = state.pixFT
  const L = ft.seq.length
  let done = 0
  const total = 3000
  const loop = () => {
    const k = 100
    for (let i = 0; i < k && done < total; i++, done++) {
      const start = Math.floor(Math.random() * (L - ft.cfg.block_size))
      trainStep(ft.params, ft.seq.slice(start, start + ft.cfg.block_size), ft.seq.slice(start + 1, start + ft.cfg.block_size + 1), ft.cfg, ft.opt)
    }
    $('pixFTInfo').textContent = `微调中… ${done}/${total}`
    if (done < total) requestAnimationFrame(loop)
    else $('pixFTInfo').textContent = '微调完成！点下方提示，看它画出对应的图案'
  }
  loop()
}
function genPixelFT(pi) {
  const ft = state.pixFT
  const gen = sample(ft.params, [16 + pi], 260, ft.cfg, { temperature: 0.05 })
  const pix = gen.slice(1).filter((v) => v < 16).slice(0, 256)
  renderGrid($('pixFTOut'), seqToGrid(pix))
  $('pixFTInfo').textContent = `提示「${PIX_FT_PATTERNS[pi].name}」→ 模型画出了它。换一个提示，看它画不同的`
}
$('pixFTBtn').addEventListener('click', () => { if (!state.pixFT) buildPixelFT(); trainPixelFT() })
buildPixelFT()

// 动手：图案插值（从 A 渐变到 B——图像空间连续）
const pixInterA = $('pixInterA'), pixInterB = $('pixInterB')
PIXEL_PATTERNS.forEach((pat) => {
  pixInterA.insertAdjacentHTML('beforeend', `<option value="${pat.id}">${pat.name}</option>`)
  pixInterB.insertAdjacentHTML('beforeend', `<option value="${pat.id}">${pat.name}</option>`)
})
pixInterB.value = PIXEL_PATTERNS[1].id
$('pixInterBtn').addEventListener('click', () => {
  const pa = PIXEL_PATTERNS.find((x) => x.id === pixInterA.value)
  const pb = PIXEL_PATTERNS.find((x) => x.id === pixInterB.value)
  if (!pa || !pb) return
  const seqA = gridToSeq(pa.grid)
  const seqB = gridToSeq(pb.grid)
  let frame = 0
  const N = 24
  const anim = setInterval(() => {
    frame++
    const t = frame / N
    const mix = seqA.map((v, i) => Math.round(v * (1 - t) + seqB[i] * t))
    renderGrid($('pixInterOut'), seqToGrid(mix))
    if (frame >= N) clearInterval(anim)
  }, 50)
})

// 动手：扩散加噪演示（拖滑杆把图案加噪成雪花）
$('pixNoise').addEventListener('input', () => {
  const p = state.pix
  if (!p) return
  const n = parseFloat($('pixNoise').value)
  const noisy = p.grid.map((row) => row.split('').map((c) => (Math.random() < n ? valToChar(Math.floor(Math.random() * 16)) : c)).join(''))
  renderGrid($('pixNoiseCanvas'), noisy)
})
document.querySelectorAll('#pixelPick .chip[data-pix]').forEach((b) => {
  b.addEventListener('click', () => {
    const pat = PIXEL_PATTERNS.find((x) => x.id === b.dataset.pix)
    if (pat) buildPixel(pat)
  })
})
$('pixTrainBtn').addEventListener('click', trainPixel)
$('pixGenBtn').addEventListener('click', genPixel)
buildPixel(PIXEL_PATTERNS[0])

// 音频条件生成微调（SFT：学会"按提示作曲"——提示作曲雏形）
const MEL_FT = MELODIES.slice(0, 3) // 小星星 / 欢乐颂 / 两只老虎
function buildMelFT() {
  const vocab = 8 + MEL_FT.length
  const cfg = { vocab_size: vocab, block_size: 32, n_layer: 1, n_head: 1, n_embd: 8, bias: true }
  const { params } = createModel(cfg, 42)
  const seq = []
  MEL_FT.forEach((m, i) => {
    seq.push(8 + i) // 提示标记 token
    seq.push(...parseMelody(m.seq))
  })
  state.melFT = { params, cfg, opt: createOptimizer(params, { type: 'adam', lr: 0.05 }), seq }
  $('melFTPick').innerHTML = MEL_FT.map((m, i) =>
    `<button class="chip" data-mi="${i}" title="输入提示标记，模型奏出对应旋律">奏「${m.name}」</button>`).join('')
  document.querySelectorAll('#melFTPick .chip').forEach((b) => {
    b.addEventListener('click', () => genMelFT(+b.dataset.mi))
  })
  $('melFTInfo').textContent = '微调就绪：小星星 / 欢乐颂 / 两只老虎 各配一个提示标记'
}
function trainMelFT() {
  const ft = state.melFT
  const L = ft.seq.length
  let done = 0
  const total = 3000
  const loop = () => {
    const k = 100
    for (let i = 0; i < k && done < total; i++, done++) {
      const start = Math.floor(Math.random() * (L - ft.cfg.block_size))
      trainStep(ft.params, ft.seq.slice(start, start + ft.cfg.block_size), ft.seq.slice(start + 1, start + ft.cfg.block_size + 1), ft.cfg, ft.opt)
    }
    $('melFTInfo').textContent = `微调中… ${done}/${total}`
    if (done < total) requestAnimationFrame(loop)
    else $('melFTInfo').textContent = '微调完成！点下方提示，听它奏对应的旋律'
  }
  loop()
}
function genMelFT(mi) {
  const ft = state.melFT
  const gen = sample(ft.params, [8 + mi], 60, ft.cfg, { temperature: 0.4 })
  const mel = gen.slice(1).filter((v) => v < 8).slice(0, 48)
  renderMelody($('melFTOut'), mel)
  playMelody(mel)
  $('melFTInfo').textContent = `提示「${MEL_FT[mi].name}」→ 模型奏出了它的旋律（自动播放中）`
}
$('melFTBtn').addEventListener('click', () => { if (!state.melFT) buildMelFT(); trainMelFT() })
buildMelFT()

// ---------- 语音模型（旋律序列，完整版）----------
const melLossChart = createLossChart($('melLoss'))
function buildMelody(seq, name) {
  const cfg = { vocab_size: 8, block_size: 16, n_layer: 1, n_head: 1, n_embd: 8, bias: true }
  const { params } = createModel(cfg, 42)
  state.mel = { seq, name: name || '自定义', params, cfg, opt: createOptimizer(params, { type: 'adam', lr: 0.05 }), trained: false, composed: null, losses: [] }
  renderMelody($('melViz'), seq)
  renderMelody($('melGenViz'), [])
  melLossChart.clear()
  $('melStage').textContent = '未训练'
  $('melGenBtn').disabled = true
  $('melPlayBtn').disabled = false
  $('melInfo').textContent = `${name || '自定义旋律'}已就绪 · ${seq.length} 个音`
}
// 动手：频率滑杆 + 频谱可视化（基频 + 泛音）+ 声波动画
const wave = createWave($('waveCanvas'))
wave.draw()
wave.start()
function renderSpectrum(canvas) {
  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth || 300
  const h = canvas.clientHeight || 100
  canvas.width = w * dpr; canvas.height = h * dpr
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  const n = 10
  const slot = w / n
  for (let i = 0; i < n; i++) {
    const amp = 1 / (i + 1) + Math.random() * 0.04 // 泛音逐渐变弱
    const bh = amp * (h - 24)
    ctx.fillStyle = i === 0 ? 'var(--accent, #b3442c)' : 'rgba(31,122,109,0.7)'
    ctx.fillRect(i * slot + 2, h - bh, slot - 4, bh)
  }
  ctx.fillStyle = 'var(--ink-soft, #6b6357)'
  ctx.font = '10px serif'
  ctx.fillText('基频', 4, h - 6)
  ctx.fillText('泛音（2x 3x 4x…逐渐变弱）→', w - 190, h - 6)
}
$('freqSlider').addEventListener('input', () => {
  $('freqVal').textContent = $('freqSlider').value + ' Hz'
  renderSpectrum($('specViz'))
  wave.setFreq(parseFloat($('freqSlider').value))
})
$('freqPlayBtn').addEventListener('click', () => {
  playTone(parseFloat($('freqSlider').value))
})
renderSpectrum($('specViz'))
function trainMelody() {
  const m = state.mel
  const L = m.seq.length
  const total = parseInt($('melSteps').value) || 2000
  m.losses = []
  melLossChart.clear()
  let done = 0
  const loop = () => {
    const k = 100
    for (let i = 0; i < k && done < total; i++, done++) {
      const start = Math.floor(Math.random() * Math.max(1, L - m.cfg.block_size))
      const { loss } = trainStep(m.params, m.seq.slice(start, start + m.cfg.block_size), m.seq.slice(start + 1, start + m.cfg.block_size + 1), m.cfg, m.opt)
      m.losses.push(loss)
    }
    melLossChart.push(m.losses[m.losses.length - 1])
    melLossChart.draw()
    $('melInfo').textContent = `训练中… ${done}/${total}`
    if (done < total) requestAnimationFrame(loop)
    else { m.trained = true; $('melGenBtn').disabled = false; $('melStage').textContent = '已学会'; $('melInfo').textContent = '完成！「模型作曲」让它续写旋律' }
  }
  loop()
}
function genMelody() {
  const m = state.mel
  if (!m.trained) return
  const temp = parseFloat($('melTemp').value) || 0.6
  const gen = sample(m.params, m.seq.slice(0, 8), 40, m.cfg, { temperature: temp })
  m.composed = gen.slice(0, 48)
  renderMelody($('melGenViz'), m.composed)
  // 生成历史（最近 3 段，不同温度对比）
  m.history = m.history || []
  m.history.push(m.composed.slice())
  if (m.history.length > 3) m.history.shift()
  const hist = $('melHistory')
  if (hist) {
    hist.innerHTML = `<div class="gen-history-title">最近作曲（温度 ${temp}）</div>` +
      m.history.map((g, gi) => `<button class="chip" data-mh="${gi}" title="播放这段">段 ${gi + 1}</button>`).join('')
    hist.querySelectorAll('.chip').forEach((b) => {
      b.addEventListener('click', () => {
        playMelody(m.history[+b.dataset.mh], parseFloat($('melBpm').value) || 220)
      })
    })
  }
  $('melInfo').textContent = `模型创作了 48 个音（温度 ${temp}）——点「播放」或历史段落试听`
}
$('melPlayBtn').addEventListener('click', () => {
  const m = state.mel
  if (!m) { alert('先选旋律'); return }
  const seq = m.composed || m.seq
  const bpm = parseFloat($('melBpm').value) || 220
  playMelody(seq, bpm)
  $('melInfo').textContent = `♪ 播放「${m.composed ? '模型创作' : m.name}」…（${bpm} BPM）`
})
// 动手：旋律插值（A → 渐变 → B）
const melInterA = $('melInterA'), melInterB = $('melInterB')
MELODIES.forEach((m) => {
  melInterA.insertAdjacentHTML('beforeend', `<option value="${m.id}">${m.name}</option>`)
  melInterB.insertAdjacentHTML('beforeend', `<option value="${m.id}">${m.name}</option>`)
})
melInterB.value = MELODIES[1].id
$('melInterBtn').addEventListener('click', () => {
  const ma = MELODIES.find((x) => x.id === melInterA.value)
  const mb = MELODIES.find((x) => x.id === melInterB.value)
  if (!ma || !mb) return
  const a = parseMelody(ma.seq)
  const b = parseMelody(mb.seq)
  const part = Math.min(a.length, b.length, 10)
  const mixed = []
  for (let i = 0; i < part; i++) mixed.push(a[i])
  for (let i = 0; i < part; i++) mixed.push(Math.round(a[i] * (1 - (i + 1) / part) + b[i] * ((i + 1) / part)))
  for (let i = 0; i < part; i++) mixed.push(b[i])
  renderMelody($('melInterOut'), mixed)
  playMelody(mixed, parseFloat($('melBpm').value) || 220)
  $('melInfo').textContent = `${ma.name} → 渐变 → ${mb.name}（听中间的过渡旋律）`
})
document.querySelectorAll('#melodyPick .chip').forEach((b) => {
  b.addEventListener('click', () => {
    const m = MELODIES.find((x) => x.id === b.dataset.mel)
    if (m) { $('melText').value = m.seq; buildMelody(parseMelody(m.seq), m.name) }
  })
})
$('melApplyBtn').addEventListener('click', () => {
  const s = $('melText').value.replace(/[^1-7]/g, '')
  if (!s) { alert('简谱只支持 1-7 数字'); return }
  buildMelody(parseMelody(s), '自定义旋律')
})
$('melTrainBtn').addEventListener('click', trainMelody)
$('melGenBtn').addEventListener('click', genMelody)
buildMelody(parseMelody(MELODIES[0].seq), MELODIES[0].name)

// 教学动画：梯度下降（小球滚 loss 碗）
const gd = createGradientDescent($('gdCanvas'))
gd.draw()
$('gdStart').addEventListener('click', () => gd.start())
$('gdReset').addEventListener('click', () => gd.reset())
$('gdLr').addEventListener('input', () => { gd.setLr(parseFloat($('gdLr').value)); gd.reset(); gd.start() })

// 多模态（一个 Transformer 通吃文本+像素+音高）
initMultiModal({ $, createModel, createOptimizer, trainStep, sample, CORPUS, PIXEL_PATTERNS, MELODIES, gridToSeq, seqToGrid, renderGrid, parseMelody, playMelody, renderMelody, state })

// 前沿论文库（通俗解析 + 原文链接）
function renderPapers(cat = '全部') {
  const list = cat === '全部' ? PAPERS : PAPERS.filter((p) => p.cat === cat)
  $('paperCats').innerHTML = PAPER_CATS.map((c) =>
    `<button class="chip ${c === cat ? 'on' : ''}" data-cat="${c}" title="筛选：${c}">${c}</button>`).join('')
  document.querySelectorAll('#paperCats .chip').forEach((b) => {
    b.addEventListener('click', () => renderPapers(b.dataset.cat))
  })
  $('paperGrid').innerHTML = list.map((p) => `
    <div class="paper card">
      <div class="paper-top"><b>${p.title}</b><span class="tag">${p.cat} · ${p.year}</span></div>
      <div class="paper-org">${p.org}</div>
      <div class="paper-sum">${p.summary}</div>
      <a class="btn ghost paper-link" href="${p.url}" target="_blank" rel="noopener" title="在新标签页打开论文原文">打开原文 ↗</a>
    </div>`).join('')
}
renderPapers()

// 动画词典（术语用动画解释）
initDict($('dictRoot'))

// 动手：量化（把当前模型权重降精度，看生成变化）
$('quantBits').addEventListener('input', () => {
  const b = +$('quantBits').value
  $('quantVal').textContent = b >= 32 ? 'FP32' : `INT${b}${b <= 4 ? '（很粗糙）' : ''}`
})
$('quantGenBtn').addEventListener('click', () => {
  if (!state.model) { alert('先构建并训练文本模型'); return }
  const bits = +$('quantBits').value
  // 同 saveSnapshot：params 是 { name: { value, grad } } 字典，逐个拷贝 value 矩阵
  const qp = {}
  for (const k in state.model.params) qp[k] = { value: JSON.parse(JSON.stringify(state.model.params[k].value)) }
  if (bits < 32) {
    const maxV = Math.pow(2, bits - 1) - 1
    for (const k in qp) {
      const m = qp[k].value
      for (const row of m) for (let i = 0; i < row.length; i++) {
        const v = Math.max(-1, Math.min(1, row[i]))
        row[i] = Math.round(v * maxV) / maxV
      }
    }
  }
  const prompt = $('prompt').value || '月光'
  const p = prompt.split('').map((c) => state.model.stoi[c] ?? 0)
  const seq = sample(qp, p, 24, state.model.cfg, { temperature: 0.8 })
  const text = tokensToText(state.model.itos, seq)
  $('quantOut').textContent = text
  const ratio = bits >= 32 ? 1 : 32 / bits
  $('quantInfo').textContent = bits >= 32
    ? 'FP32（原始精度）：占空间 100%，质量最高'
    : `INT${bits}：权重被舍入到 ${Math.pow(2, bits - 1) - 1} 级 · 理论体积约 ${(100 / ratio).toFixed(0)}%（快 ${ratio.toFixed(1)} 倍）· 看生成是否变"糙"`
})

// 彩蛋：连点题签 10 次 → 隐藏章「算无遗策」
let stampClicks = 0
document.querySelector('.masthead').addEventListener('click', (e) => {
  if (e.target.closest('button, input, select, textarea')) return
  stampClicks++
  if (stampClicks >= 10) {
    stampClicks = 0
    showToast('你连点了 10 次题签——盖下隐藏章：「算无遗策」')
    eggFound()
  }
})

// 卡片 staggered 入场（教学节奏：一张张"翻开"）
function staggerCards() {
  document.querySelectorAll('.card').forEach((c, i) => {
    c.style.animationDelay = (Math.min(i, 14) * 40) + 'ms'
  })
}
staggerCards()

// 底层原理：语义空间（词向量 PCA 投影——模型学到了什么）
function showEmbed() {
  if (!state.model) { alert('先构建并训练模型'); return }
  const wte = state.model.params.wte.value
  const itos = state.model.itos
  renderEmbed($('embedCanvas'), pca2d(wte), itos)
  const pairs = topSimilarPairs(wte, 8)
  $('embedPairs').innerHTML = pairs.length
    ? '最相似的词对：' + pairs.map((p) => `${itos[p.i]}~${itos[p.j]}（${p.s.toFixed(2)}）`).join(' · ')
    : '（相似度都很低——语料太少或训练不够，还没形成明显语义）'
  $('embedInfo').textContent = `当前模型 · ${itos.length} 个词向量`
}
function showEmbedRandom() {
  if (!state.model) return
  const size = SIZES[$('size').value]
  const cfg = { vocab_size: state.model.itos.length, bias: true, ...size }
  const { params } = createModel(cfg, 999)
  renderEmbed($('embedCanvas'), pca2d(params.wte.value), state.model.itos)
  $('embedPairs').innerHTML = '训练前（随机权重）：杂乱无章、没有语义结构——对比上面的训练后'
  $('embedInfo').textContent = '随机初始化模型（对比用）'
}
$('embedBtn').addEventListener('click', showEmbed)
$('embedResetBtn').addEventListener('click', showEmbedRandom)

// ---------- 参考文献（每个板块底部，统一格式：年份 · 作者 / 标题 / 一句话） ----------
const REFS = {
  text: [
    ['2017', 'Vaswani 等', 'Attention Is All You Need', 'Transformer 出生证明——你训练的模型就是它的微型版。arXiv:1706.03762'],
    ['2018', 'Radford 等', 'Improving Language Understanding by Generative Pre-Training', 'GPT 一代：预训练 + 微调的开端，本页三阶段（预训练/SFT/DPO）的祖师爷。'],
    ['2022', 'Ouyang 等', 'Training language models to follow instructions with human feedback', 'InstructGPT：用人类反馈对齐模型——对齐阶段的真实出处。'],
    ['2023', 'Rafailov 等', 'Direct Preference Optimization', 'DPO：跳过奖励模型直接偏好优化——你点的"哪个更好"就是它。arXiv:2305.18290'],
  ],
  image: [
    ['2016', 'van den Oord 等', 'Pixel Recurrent Neural Networks', '把图像当像素序列"一个一个猜"的鼻祖——本项目图像模型就是这个思路。'],
    ['2021', 'Dosovitskiy 等', 'An Image is Worth 16x16 Words', 'ViT：把图片切成小块当"词"喂给 Transformer——图像版"猜下一个字"。'],
  ],
  voice: [
    ['2016', 'van den Oord 等', 'WaveNet', '把声音当波形"一个点一个点生成"——音频版的"猜下一个字"。'],
    ['2022', 'Radford 等', 'Robust Speech Recognition via Large-Scale Weak Supervision', 'Whisper：语音转文字的大规模自监督——音高序列思路的远方亲戚。'],
  ],
  multi: [
    ['2021', 'Radford 等', 'Learning Transferable Visual Models From Natural Language Supervision', 'CLIP：文本和图像进同一个表示空间——多模态对齐的开山之作。'],
    ['2023', 'Google', 'Gemini: A Family of Highly Capable Multimodal Models', '原生多模态：一个 Transformer 直接吃文本+图像+音频——本项目多模态 tab 的思路来源。'],
  ],
  trans: [
    ['2015', 'Bahdanau 等', 'Neural Machine Translation by Jointly Learning to Align and Translate', '第一次把注意力加进翻译——"先念经，再用注意力对齐"的来源。arXiv:1409.0473'],
    ['2017', 'Vaswani 等', 'Attention Is All You Need', '本页主角：只用注意力，扔掉循环和卷积。arXiv:1706.03762'],
    ['2018', 'Radford 等', 'Improving Language Understanding by Generative Pre-Training', 'GPT 一代：只用 Decoder + 预训练，生成这条路的起点。'],
    ['2019', 'Devlin 等', 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', '只用 Encoder + 预训练，理解这条路的霸主。arXiv:1810.04805'],
  ],
  frontier: [
    ['2017', 'Shazeer 等', 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer', 'MoE：把模型拆成专家，每个 token 只激活少数——DeepSeek 同款。'],
    ['2021', 'Su 等', 'RoFormer: Enhanced Transformer with Rotary Position Embedding', 'RoPE：旋转位置编码——让上下文窗口从 4K 涨到 100K+ 的关键。'],
    ['2022', 'Dao 等', 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness', 'Flash Attention：重写注意力计算，读写省几倍——训练推理都变快。'],
  ],
}
function renderRefs() {
  for (const tab in REFS) {
    const panel = document.getElementById('tab-' + tab)
    if (!panel) continue
    const sec = document.createElement('section')
    sec.className = 'card'
    sec.innerHTML = '<h2>附 · 参考文献</h2>' + REFS[tab].map((r) =>
      `<div class="paper"><div class="paper-top"><b>${r[0]} · ${r[1]}</b></div><p class="paper-sum">《${r[2]}》${r[3]}</p></div>`
    ).join('')
    panel.appendChild(sec)
  }
}

// ---------- 启动 ----------
// Tab 切换：hash 路由（#/transformer）+ 动态 title，可直达分享、可被收录
const TAB_NAMES = { text: '文本模型', image: '图像模型', voice: '语音模型', multi: '多模态', trans: 'Transformer', dict: '词典', frontier: '前沿', papers: '论文' }
function goto(tab, noHistory) {
  if (!TAB_NAMES[tab]) tab = 'text'
  document.querySelectorAll('.tab-btn').forEach((x) => { x.classList.remove('on'); x.setAttribute('aria-selected', 'false') })
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('on'))
  const b = document.querySelector('.tab-btn[data-tab="' + tab + '"]')
  if (b) { b.classList.add('on'); b.setAttribute('aria-selected', 'true') }
  const panel = document.getElementById('tab-' + tab)
  if (panel) panel.classList.add('on')
  // 移动端：选完 tab 收起抽屉
  document.querySelectorAll('.tabs, .nav-toggle').forEach((el) => el.classList.remove('open'))
  document.title = 'AI 学习本 · ' + TAB_NAMES[tab]
  if (!noHistory && location.hash !== '#/' + tab) {
    try { history.replaceState(null, '', '#/' + tab) } catch (e) { /* 忽略 */ }
  }
  // 隐藏面板的 canvas 尺寸为 0，切回时重绘；wired 控件同理（wiredRender(true) 强制按真实尺寸重绘）
  setTimeout(() => {
    try {
      document.querySelectorAll('wired-slider,wired-button,wired-input,wired-textarea,wired-select').forEach((el) => { if (el.wiredRender) el.wiredRender(true) })
      tintWiredInputs()
      lossChart.draw(); heatmap.draw(); attnHeatmap.draw()
      pixLossChart.draw(); melLossChart.draw()
      if (state.pix) { renderGrid($('pixTarget'), state.pix.grid); renderGrid($('pixOut'), seqToGrid(Array(256).fill(0))) }
      if (state.mel) { renderMelody($('melViz'), state.mel.seq); renderMelody($('melGenViz'), state.mel.composed || []) }
      redrawTrans()
    } catch (e) { /* 忽略未就绪 */ }
  }, 60)
}
document.querySelectorAll('.tab-btn').forEach((b) => {
  b.addEventListener('click', () => goto(b.dataset.tab))
})
window.addEventListener('hashchange', () => {
  const tab = (location.hash || '').replace(/^#\/?/, '')
  goto(tab, true)
})
// 初始：读 hash 直达对应 tab
;(function initHash() {
  const tab = (location.hash || '').replace(/^#\/?/, '')
  if (TAB_NAMES[tab]) goto(tab, true)
})()
// 无障碍：tab 语义（tablist/tab/tabpanel + aria-selected）
;(function initAria() {
  const tabsNav = $('mainTabs')
  if (tabsNav) tabsNav.setAttribute('role', 'tablist')
  document.querySelectorAll('.tab-btn').forEach((x) => {
    x.setAttribute('role', 'tab')
    x.setAttribute('aria-selected', x.classList.contains('on') ? 'true' : 'false')
  })
  document.querySelectorAll('.tab-panel').forEach((p) => {
    p.setAttribute('role', 'tabpanel')
    if (!p.hasAttribute('tabindex')) p.setAttribute('tabindex', '0')
  })
})()
renderAch()
renderGlossary()
initNotes()
initTransformer($('transRoot'))
initMicroscopeUI($('microscopeRoot'), () => state.model)
renderRefs()

// 移动端：汉堡按钮展开/收起导航抽屉
const _navToggle = $('navToggle')
if (_navToggle) {
  _navToggle.addEventListener('click', () => {
    document.querySelectorAll('.tabs, .nav-toggle').forEach((el) => el.classList.toggle('open'))
  })
}

// 新手入口：跳到文本 tab 训练区
const _start = $('startHere')
if (_start) {
  _start.addEventListener('click', () => {
    goto('text')
    setTimeout(() => { const c = $('dataCard'); if (c) c.scrollIntoView({ behavior: 'smooth' }) }, 80)
  })
}
// 回到顶部
const _toTop = $('toTop')
if (_toTop) {
  window.addEventListener('scroll', () => _toTop.classList.toggle('show', window.scrollY > 400), { passive: true })
  _toTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }))
}
// 成就顶栏徽章：跳到文本 tab 成就区
const _achBadge = $('achBadge')
if (_achBadge) {
  _achBadge.addEventListener('click', () => {
    goto('text')
    setTimeout(() => { const a = $('achRoot'); if (a) a.scrollIntoView({ behavior: 'smooth' }) }, 80)
  })
}

// 彩蛋：深夜引言（00:00-04:00 打开页面）
const _h = new Date().getHours()
if (_h >= 0 && _h < 4) {
  const _sub = document.querySelector('.sub')
  if (_sub) _sub.textContent = '夜深了，数字的世界依然清醒。'
}

// 彩蛋：无穷小 μ（档位拉到最小）
$('size').addEventListener('change', () => {
  const title = document.querySelector('.title')
  const mu = title.querySelector('.mu')
  if ($('size').value === 'ultratiny' && !mu) title.insertAdjacentHTML('beforeend', '<span class="mu"> μ</span>')
  if ($('size').value !== 'ultratiny' && mu) mu.remove()
})

// 初始语料：必须等 wired-textarea 完成首次升级渲染（firstUpdated 应用 pendingValue）后再读。
// 此前同步调用时 host .value getter 返回空串 → buildModel 弹「请输入或选择语料」且初始模型构建失败。
;(() => {
  const boot = () => setCorpus(CORPUS[0].text, CORPUS[0].title)
  requestAnimationFrame(() => requestAnimationFrame(boot))
})()
