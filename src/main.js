// ============================================================
// src/main.js
// 手算LM —— M1：预训练实验台（数据区 + 训练控制 + 基础可视化 + 流式生成）
// ============================================================

import './style.css'
import { CORPUS, buildVocab, tokensToText, TOKEN_NAME, USER, ASSISTANT, END } from './corpus.js'
import { createModel, paramCount } from './model.js'
import { trainStep, createOptimizer } from './train.js'
import { sample } from './sample.js'
import { createLossChart, createHeatmap, createAttnHeatmap } from './ui.js'
import { sampleWithAttn } from './attn.js'
import { initMicroscopeUI } from './microscope-ui.js'
import { NOTES } from './notes.js'
import { FAQ } from './glossary.js'
import { ACHIEVEMENTS, loadEarned, saveEarned } from './ach.js'
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
      <h1 class="title">手算<span class="hl">LM</span></h1>
      <p class="sub">给你权重，亲手算出它的下一句话</p>
    </header>

    <div class="intro">
      <p>这里是一个让你<b>亲手训练迷你大模型</b>的实验台——一个只会"猜下一个字"的小机器，在你的浏览器里<b>真实训练</b>，没有任何预设剧本。</p>
      <p class="intro-path">你的旅程：<b>壹</b> 选语料 · <b>贰</b> 训练它学说话 · <b>叁</b> 微调它会回答 · <b>肆</b> 生成看它想什么 · <b>伍</b> 对齐让它变讨喜 · <b>陆</b> 显微镜看穿每一步</p>
      <p class="muted">每张卡片下方有「怎么做」指引；右上「翻面」可看背后的公式。</p>
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
    </section>

    <section class="card" id="sftCard">
      <h2>叁 · 微调（SFT）</h2>
      <p class="muted">喂给模型问答对，让它从"接着写"学会"回答问题"——在预训练权重上继续真实训练。<b>SFT = Supervised Fine-Tuning（监督微调）</b>：为什么微调？预训练只会续写，微调换问答数据教它"回答问题"。</p>
      <div class="corpus-pick">
        <select id="qaSet" class="inline-select" title="选择要载入的示例问答套：通用对话 / 关于手算LM / 趣味问答">
          <option value="general">通用对话</option>
          <option value="about">关于手算LM</option>
          <option value="fun">趣味问答</option>
        </select>
        <button id="loadQaBtn" class="chip" title="把选中的示例问答对填入下面的文本框（可再编辑）">载入示例问答</button>
        <span class="hint">每行一条「问题 / 回答」，斜杠分隔</span>
      </div>
      <textarea id="qaList" rows="4" placeholder="每行一条：问题 / 回答。例：你是谁 / 我是手算LM"></textarea>
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
        <button id="sftBtn" class="btn" title="开始微调：用问答对继续训练当前模型（学习率已自动调低，混合比例防遗忘）。微调后模型会"回答问题"">开始微调</button>
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
  </main>
`

const $ = (id) => document.getElementById(id)
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
  // 字符表预览（可读性：模型认识哪些字）
  const vv = $('vocabView')
  if (vv) {
    vv.innerHTML = '<span class="vocab-label">模型认识的字符表：</span>' +
      chars.map((c) => `<span class="vocab-char" title="token #${stoi[c]}">${TOKEN_NAME[c] ?? c}</span>`).join('')
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
  const data = {
    params: JSON.parse(JSON.stringify(state.model.params.value)),
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
  for (const key in m.params.value) m.params.value[key] = data.params[key].map((row) => row.slice())
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

function snapshotWeights() {
  state.snap = deepCopy(state.model.params.value)
  state.showingBefore = false
  $('cmpBtn').disabled = false
  $('cmpBtn').textContent = '切到：微调前'
  $('sftInfo').textContent = '已记录微调前权重快照'
}

function toggleCompare() {
  if (!state.snap) return
  const cur = state.model.params.value
  const tmp = deepCopy(cur)
  for (const key in cur) cur[key] = deepCopy(state.snap[key])
  state.snap = tmp
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
      // 彩蛋：手算者印（三阶段全部完成）
      state.ach.allDone = true
      checkAch()
      if (!document.querySelector('.stamp')) {
        const stamp = document.createElement('div')
        stamp.className = 'stamp'
        stamp.textContent = '手算者 · 完成'
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
document.querySelectorAll('.chip').forEach((btn) => {
  btn.addEventListener('click', () => {
    const c = CORPUS.find((x) => x.id === btn.dataset.id)
    setCorpus(c.text, c.title)
  })
})
$('corpus').addEventListener('input', () => {
  $('corpusInfo').textContent = `自定义语料 ${$('corpus').value.length} 字`
  stopTraining()
  buildModel()
  lossChart.draw()
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

// ---------- 主题切换（暗色/亮色，localStorage 持久化）----------
function applyTheme(t) {
  document.documentElement.setAttribute('data-theme', t)
  $('themeBtn').textContent = t === 'dark' ? '亮色模式' : '暗色模式'
  try { localStorage.setItem('handcalc:theme', t) } catch { /* 忽略 */ }
}
let _savedTheme = null
try { _savedTheme = localStorage.getItem('handcalc:theme') } catch { /* 忽略 */ }
applyTheme(_savedTheme === 'dark' ? 'dark' : 'light')
$('themeBtn').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'
  applyTheme(cur)
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
  root.innerHTML = `
    <section class="card">
      <h2>玖 · 成就 <span class="tag">${earned.size}/${ACHIEVEMENTS.length}</span></h2>
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
  const t = document.createElement('div')
  t.className = 'ach-toast'
  t.textContent = `解锁成就：${a.name} — ${a.desc}`
  document.body.appendChild(t)
  setTimeout(() => t.remove(), 3200)
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
}

// 显微镜手算答对 → 成就
document.addEventListener('handcalc:calcwin', () => {
  state.ach.calcWin = true
  checkAch()
})

// ---------- 启动 ----------
renderAch()
renderGlossary()
initNotes()
initMicroscopeUI($('microscopeRoot'), () => state.model)

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

setCorpus(CORPUS[0].text, CORPUS[0].title)
