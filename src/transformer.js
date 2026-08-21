// ============================================================
// src/transformer.js
// AI 学习本 —— Transformer 深度解析板块
// 用最长、最让人听懂的方式拆解现代 LLM 的基石技术，
// 对比卷积网络，并深度解析论文《Attention Is All You Need》
// ============================================================

// ---------- 教学内容（演草纸风格：章节=题目，就地解释，不跳术语表） ----------
const TRANS_CONTENT = `
<section class="card">
  <h2>壹 · 一句话先听懂</h2>
  <p>Transformer 是一个<b>开会讨论的会场</b>：输入里的每一个词都举手发言，同时听全场所有人的发言，然后一起决定——下一个词最该是谁。</p>
  <p>给它一句话「我 爱 你」，它会这样工作：</p>
  <div class="row">
    <div class="mm-track mono" style="flex:1">我 → 爱 → 你（三个词同时进场，互相看一眼全场）</div>
  </div>
  <div class="howto">先记住这个画面：<b>每个词都能一步看到全场</b>。这是它与所有老网络（RNN、CNN）最本质的区别——后面每一节都在解释这句话。</div>
  <p class="muted">你的手算 LM 就是一个微型 Transformer：几百个参数，但注意力、多头、位置编码全都真材实料。这一页把它的每一颗螺丝拆给你看。</p>
</section>

<section class="card">
  <h2>贰 · 它为什么出生（2017 年之前的世界）</h2>
  <p>2017 年之前，做"猜下一个字"（语言模型）主要靠两类网络，各有致命伤：</p>
  <div class="pair">
    <div class="answer">
      <div class="blind-title">RNN（循环神经网络）—— 逐字念经</div>
      <p>一个字一个字地读：读「我」，更新记忆，再读「爱」，再更新……<b>串行</b>——读完 100 个字才能处理第 100 个。</p>
      <p class="ms-slogan">致命伤 ① 慢：GPU 再强，也只能一个字一个字跑。<br>致命伤 ② 记不住：读第 1 个字的记忆传到第 100 个字时早已"淡出"——长句子顾头不顾尾。</p>
    </div>
    <div class="answer">
      <div class="blind-title">CNN（卷积）—— 只看窗口</div>
      <p>拿一个固定大小的"窗口"在序列上滑动，只看窗口内的几个字。看远一点？把窗口加大，或者叠很多层。</p>
      <p class="ms-slogan">致命伤：窗口再大也有限。第 1 个字和第 100 个字的关系，要么堆几十层卷积，要么直接看不到。</p>
    </div>
  </div>
  <div class="teach">
    <h3>Transformer 的破局思路</h3>
    <p>不逐字读、不滑窗口——<b>让所有词同时互相看</b>。第 1 个字想看第 100 个字？<b>一步到位</b>，中间不经过任何中转。这就是自注意力（Self-Attention）。</p>
    <p class="howto">"Attention Is All You Need"——连循环都不要，连卷积都不要，<b>只要注意力就够了</b>。这就是那篇论文的标题，也是它敢叫这个名字的原因。</p>
  </div>
</section>

<section class="card">
  <h2>叁 · 和卷积网络（CNN）比一比</h2>
  <p>CNN 和 Transformer 都做"信息融合"，但哲学完全不同。用一个画面记住：</p>
  <div class="viz-row">
    <div class="viz">
      <div class="viz-title">CNN：放大镜扫图 <span class="tag">局部</span></div>
      <canvas id="cmpCnn" class="canvas"></canvas>
      <p class="muted">卷积核 3x3 滑过图像/序列，每个位置只看到邻居。想看全局 → 多叠几层。</p>
    </div>
    <div class="viz">
      <div class="viz-title">Transformer：高处看全图 <span class="tag">全局</span></div>
      <canvas id="cmpAttn" class="canvas"></canvas>
      <p class="muted">自注意力一步连接所有位置，任意两点的关系直接算。</p>
    </div>
  </div>
  <table class="ms-table" style="width:100%">
    <tr><th>维度</th><th>CNN（卷积）</th><th>Transformer（注意力）</th></tr>
    <tr><td><b>视野</b></td><td>局部窗口（kernel size）</td><td>全局（一步看全序列）</td></tr>
    <tr><td><b>看远距离</b></td><td>要叠很多层才够</td><td>一层就够（一步直达）</td></tr>
    <tr><td><b>并行</b></td><td>可以并行，但感受野靠层数堆</td><td>天然并行（所有位置同时算）</td></tr>
    <tr><td><b>位置信息</b></td><td>卷积自带位置（窗口形状）</td><td>没有——必须手动加<b>位置编码</b></td></tr>
    <tr><td><b>擅长</b></td><td>图像底层特征（边缘、纹理）</td><td>语言、序列、关系建模</td></tr>
    <tr><td><b>复杂度</b></td><td>O(n·k)（k=窗口）</td><td>O(n²)（n=序列长，每个位置看每个位置）</td></tr>
  </table>
  <div class="howto">注意最后一行：注意力 O(n²) 是它的软肋——句子越长越贵。这也是后来 Flash Attention、稀疏注意力、线性注意力都在攻克的问题（前沿 tab 有讲）。</div>
</section>

<section class="card">
  <h2>肆 · 核心零件逐个拆（重头戏）</h2>
  <p>一个 Transformer 层由下面这些零件组成。我们一个一个拆，每个都用"人话 + 数字例子"讲透。</p>

  <div class="teach">
    <h3>零件 ① 词嵌入：查表把字变成向量</h3>
    <p>字不是数字，模型只认数字。词嵌入（Embedding）就是一张大表：<b>每个字对应一行数字（向量）</b>。</p>
    <p class="mono">我 → [0.12, -0.34, 0.55]　爱 → [-0.21, 0.68, 0.10]　你 → [0.43, 0.05, -0.72]</p>
    <p>训练时这张表里的数字不断被调整——意思相近的字，向量方向也相近。去「词典」tab 或训练 tab 的<b>语义空间</b>看真实效果（PCA 把高维压到 2D，你能看见"月光"和"荷塘"挤在一起）。</p>
  </div>

  <div class="teach">
    <h3>零件 ② 位置编码：告诉模型"顺序"</h3>
    <p>注意力不分先后——"我爱你"和"你爱我"在它眼里结构一样。所以必须<b>手动把"位置"信息塞进向量</b>。</p>
    <p>论文用了一组正弦余弦函数，给第 pos 个位置、第 i 个维度加上：</p>
    <p class="mono">PE(pos, 2i)   = sin(pos / 10000^(2i/d))<br>PE(pos, 2i+1) = cos(pos / 10000^(2i/d))</p>
    <p><b>直觉</b>：不同维度用不同"波长"的波——低维（小 i）波长长、摆得慢，高维波长短、摆得快。于是每个位置都有一串独特的"指纹"，且<b>相邻位置指纹相似、远处位置指纹不同</b>。</p>
    <div class="viz">
      <div class="viz-title">位置编码曲线（每个维度一条波，位置 0→50）</div>
      <canvas id="posEncCanvas" class="canvas"></canvas>
      <p class="muted">同一维度上，位置越近值越接近——模型能从中读出"第 3 个和第 4 个挨着"。</p>
    </div>
    <p class="howto">为什么不直接用 1、2、3…？因为模型只见过训练时的数字范围，换个长度就懵；而且 sin/cos 的值有界（-1~1），不会爆炸。今天的模型用更新的办法（RoPE，旋转位置编码），思路相同。</p>
  </div>

  <div class="teach">
    <h3>零件 ③ Q K V：三张票</h3>
    <p>每个词进会场时，随身带三样东西（由嵌入向量乘三个矩阵得到）：</p>
    <p><b>Q（Query 查询）</b>——"我在找什么"。<b>K（Key 键）</b>——"我是什么/我能提供什么"。<b>V（Value 值）</b>——"我真正的内容"。</p>
    <p class="howto">像相亲现场：Q 是"我的要求"，K 是"我的标签"，V 是"我本人"。匹配看 Q 和 K 合不合，融合拿 V。</p>
  </div>

  <div class="teach">
    <h3>零件 ④ 缩放点积注意力：核心中的核心</h3>
    <p>对每个词，用它的 Q 去点所有词的 K（点积 = 两个向量有多对齐），得到一堆<b>分数</b>；分数过 softmax 变成<b>权重</b>；再对 V 加权求和——这就是这个词"看完全场后的新理解"。</p>
    <p class="mono">Attention(Q,K,V) = softmax( Q·Kᵀ / √d_k ) · V</p>
    <p>一个「我 爱 你」的具体例子（我们手工算一遍，向量都是假想的小数字）：</p>
    <div class="ms-step">
      <div class="ms-title">第 1 步：查 Q/K（每个词的查询和键）</div>
      <table class="ms-table">
        <tr><th>词</th><th>Q（查询）</th><th>K（键）</th></tr>
        <tr><td class="ms-tok">我</td><td>[1.0, 0.2]</td><td>[0.5, 0.8]</td></tr>
        <tr><td class="ms-tok">爱</td><td>[0.4, 1.0]</td><td>[1.0, 0.3]</td></tr>
        <tr><td class="ms-tok">你</td><td>[0.3, 0.6]</td><td>[0.2, 0.9]</td></tr>
      </table>
    </div>
    <div class="ms-step">
      <div class="ms-title">第 2 步：点积算分数（「我」去问全场）</div>
      <p class="mono">我·我: 1.0×0.5 + 0.2×0.8 = 0.66<br>我·爱: 1.0×1.0 + 0.2×0.3 = 1.06<br>我·你: 1.0×0.2 + 0.2×0.9 = 0.38</p>
      <p class="muted">「我」认为「爱」和自己最相关（分数最高）——合理，主谓宾里"我"和"爱"确实靠得近。</p>
    </div>
    <div class="ms-step">
      <div class="ms-title">第 3 步：除以 √d_k，再 softmax</div>
      <p class="mono">√d_k = √2 ≈ 1.41　→　[0.47, 0.75, 0.27]<br>softmax →　[0.31, 0.46, 0.23]</p>
      <p class="howto">为什么要除以 √d_k？分数是 d 维向量点积，维度越高分数自然越大，直接 softmax 会"饱和"（一个接近 1、其他几乎 0，学不动）。缩一下让梯度舒服。</p>
    </div>
    <div class="ms-step">
      <div class="ms-title">第 4 步：对 V 加权求和 =「我」的新理解</div>
      <p class="mono">0.31×V(我) + 0.46×V(爱) + 0.23×V(你)</p>
      <p>于是「我」这个位置的输出向量里，混进了 46% 的「爱」、23% 的「你」——<b>它不再孤立，它理解了上下文</b>。所有词同时做这件事，就是一层自注意力。</p>
    </div>
    <div class="row">
      <button id="attnCalcBtn" class="btn ghost" title="再算一遍（值不变，重点是跟着步骤走一遍流程）">再走一遍流程</button>
      <span id="attnCalcMsg" class="muted"></span>
    </div>
  </div>

  <div class="teach">
    <h3>零件 ⑤ 多头注意力：8 个侦探分头查</h3>
    <p>一个头只看一种关系，太单一。论文让模型用<b>多组 Q/K/V（多头）</b>并行算，每组头各管一种关系：</p>
    <div class="feat-grid">
      <div class="feat"><b>头 1</b><span>管语法：谁修饰谁</span></div>
      <div class="feat"><b>头 2</b><span>管指代："它"指谁</span></div>
      <div class="feat"><b>头 3</b><span>管位置：前后距离</span></div>
      <div class="feat"><b>头 4..8</b><span>管各种语义关系……</span></div>
    </div>
    <p>最后把所有头的输出拼起来再过一层线性——模型同时看多种角度。</p>
    <p class="howto">你的手算 LM 也用多头（去训练 tab 看配置；显微镜/注意力直播里能看见每个头在"盯"谁）。</p>
  </div>

  <div class="teach">
    <h3>零件 ⑥ 前馈网络 FFN：每个词独立再想一步</h3>
    <p>注意力负责"词与词之间"的信息交换；<b>前馈网络负责"每个词自己"的加工</b>：一个两层的 MLP（先放大再缩回），每个位置独立过一遍。</p>
    <p class="mono">FFN(x) = W₂ · GELU(W₁·x + b₁) + b₂</p>
    <p class="muted">像批注：会场上聊完（注意力），各自回座位再消化一遍（FFN）。两个零件交替堆叠，就是完整的 Transformer 层。</p>
  </div>

  <div class="teach">
    <h3>零件 ⑦ 残差连接 + LayerNorm：稳定训练的保险</h3>
    <p>每层外面包一圈：<b>残差</b>（输入 + 输出，像传送带，信息不丢、梯度好传）和 <b>LayerNorm</b>（把每层的数值拉回标准范围，像校平器，防止越叠越歪）。</p>
    <p class="mono">x = LayerNorm( x + Attention(x) )<br>x = LayerNorm( x + FFN(x) )</p>
    <p class="howto">没有这两样，几十层叠下去数值就会漂移、训练直接崩。你的手算 LM 的每一层也都是这个"传送带 + 校平器"结构。</p>
  </div>

  <div class="teach">
    <h3>零件 ⑧ 拼成一层（数据流总览）</h3>
    <p class="mono">输入（字） → 词嵌入 + 位置编码 → [多头注意力 + 残差 + LayerNorm] → [FFN + 残差 + LayerNorm] → 输出向量 → 过一个矩阵变成"下一个字"的概率</p>
    <p>中间那两块叠 N 次（论文里 N=6，你的手算 LM 叠 1 层）——叠得越深，理解越抽象。</p>
  </div>
</section>

<section class="card">
  <h2>伍 · Encoder 和 Decoder：读题的与答题的</h2>
  <p>论文原版用在翻译上，分两半：</p>
  <div class="pair">
    <div class="answer">
      <div class="blind-title">Encoder 编码器（读题）</div>
      <p>把整句源语言读一遍，每个位置都能看全句（双向自注意力），产出一批"理解向量"。</p>
    </div>
    <div class="answer">
      <div class="blind-title">Decoder 解码器（答题）</div>
      <p>一个字一个字生成目标语言。它有两个注意力：一个看自己已生成的部分（<b>带 mask，不能偷看未来</b>），一个去看 Encoder 的理解（交叉注意力）。</p>
    </div>
  </div>
  <p class="howto">mask（掩码）很重要：生成"我爱"时，decoder 不许看到"你"——真实世界里未来还没发生。今天的大模型只有 Decoder（只生成），结构更简单，但"不能偷看未来"这条规矩一模一样（你的模型训练时就是这么做的：只预测当前位置之后的一个字）。</p>
</section>

<section class="card">
  <h2>陆 · 它怎么学（很短，因为你在别处练过）</h2>
  <p>训练就是：预测下一个字 → 拿正确答案算<b>交叉熵损失</b> → 反向传播更新所有矩阵（Q/K/V、嵌入、FFN、输出层）。</p>
  <p>这整个流程你在「文本模型」tab 已经亲手跑过了——你训练的那个模型，每一颗螺丝都和这一页讲的一样，只是尺寸迷你。去「显微镜」tab，你能亲眼看到它的注意力给每个 token 打的分数；去「注意力直播」，能看见 Q/K/V 实时流动。</p>
  <div class="howto">这就是"手算 LM"的初心：<b>把黑盒拆成看得见、算得出的零件</b>。Transformer 再大，也是这些零件堆起来的。</div>
</section>

<section class="card">
  <h2>柒 · 论文深度解析：《Attention Is All You Need》</h2>
  <div class="ms-step">
    <div class="ms-title">档案</div>
    <table class="ms-table">
      <tr><td><b>标题</b></td><td>Attention Is All You Need（只需要注意力）</td></tr>
      <tr><td><b>作者</b></td><td>Ashish Vaswani 等 8 人（Google Brain / Google Research）</td></tr>
      <tr><td><b>时间</b></td><td>2017 年 6 月，arXiv:1706.03762</td></tr>
      <tr><td><b>一句话贡献</b></td><td>提出 Transformer：只用注意力做序列建模，弃用循环和卷积</td></tr>
    </table>
  </div>
  <div class="ms-step">
    <div class="ms-title">它回答的问题</div>
    <p>2017 年机器翻译的主流是"RNN + 注意力"（Bahdanau 2015 把注意力加进 RNN 帮助对齐）。但 RNN 天生串行——GPU 再多也只能一个个字跑，长句记忆还会衰减。作者问：<b>如果注意力已经这么有用，为什么还要循环？</b></p>
  </div>
  <div class="ms-step">
    <div class="ms-title">它的三个关键设计</div>
    <p>① <b>缩放点积注意力</b>：Q·Kᵀ/√d → softmax → 加权 V。比加法注意力快、省内存。<br>② <b>多头注意力</b>：并行多组 Q/K/V，模型同时学多种关系。<br>③ <b>位置编码</b>：sin/cos 函数给序列注入顺序——没有循环和卷积后，位置必须显式给。</p>
  </div>
  <div class="ms-step">
    <div class="ms-title">实验结果（当时的世界纪录）</div>
    <table class="ms-table">
      <tr><th>任务</th><th>成绩</th><th>意义</th></tr>
      <tr><td>WMT 英→德翻译</td><td>BLEU 28.4</td><td>当时最强，超过所有 RNN 模型</td></tr>
      <tr><td>WMT 英→法翻译</td><td>BLEU 41.8</td><td>单模型 SOTA，训练成本更低</td></tr>
      <tr><td>训练速度</td><td>8 块 GPU 约 3.5 天</td><td>比当时最好的 RNN 模型快 3 倍以上</td></tr>
    </table>
    <p class="muted">快，是因为注意力所有位置同时算（并行）；强，是因为长距离一步直达。速度和精度双赢，这是它横扫翻译的原因。</p>
  </div>
  <div class="ms-step">
    <div class="ms-title">为什么它是现代 LLM 的基石</div>
    <p>论文发表后，两条路都从它长出来：<br>① <b>BERT（2018）</b>：只用 Encoder + 预训练 → 语言理解任务全面刷榜。<br>② <b>GPT（2018 起）</b>：只用 Decoder + 预训练 → 生成能力一路放大 → GPT-3 → ChatGPT → 今天的整个 LLM 时代。</p>
    <p class="howto">今天所有大模型——GPT、Claude、Gemini、DeepSeek、豆包——架构上都是这篇论文的子孙：<b>堆叠的 Transformer 层，只有注意力，没有循环，没有卷积</b>。"Attention Is All You Need" 就是现代大模型的出生证明。</p>
    <div class="egg">彩蛋：论文标题是作者们深思熟虑的宣言——"你只需要注意力"，连 RNN 的注意力增强都不要了，直接纯注意力开局。</div>
  </div>
</section>

<section class="card">
  <h2>捌 · 下一步：去亲手摸它</h2>
  <p>这一页讲的是"图纸"。真家伙在你的浏览器里：</p>
  <div class="feat-grid">
    <div class="feat"><b>文本模型 tab</b><span>训练一个真正的微型 Transformer，看损失曲线一路下降。</span></div>
    <div class="feat"><b>注意力直播</b><span>生成时逐 token 看 Q/K/V 和注意力分数（同款多头！）。</span></div>
    <div class="feat"><b>显微镜 tab</b><span>把"猜下一个字"拆成 6 步手算，连注意力矩阵都给你列出来。</span></div>
    <div class="feat"><b>语义空间</b><span>看训练后词向量自动聚类——嵌入表在长什么。</span></div>
  </div>
  <div class="howto">读完这一页再回去训练，你会看见每一层在做什么——这就是"看懂"和"会用"的区别。</div>
</section>
`

// ---------- 位置编码曲线 ----------
function drawPosEncoding() {
  const canvas = document.getElementById('posEncCanvas')
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const w = canvas.clientWidth, h = canvas.clientHeight
  const dpr = window.devicePixelRatio || 1
  canvas.width = w * dpr; canvas.height = h * dpr
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, w, h)
  const dark = document.documentElement.getAttribute('data-theme') === 'dark'
  const lineC = dark ? '#b8ae9e' : '#6b6257'
  const palette = ['#c0452c', '#1f7a6d', '#8a5a44', '#9c9285']
  if (dark) palette[0] = '#ff8f5c', palette[1] = '#5cd4c0', palette[2] = '#d6a08a', palette[3] = '#8a8172'
  ctx.strokeStyle = 'rgba(0,0,0,0.15)'; ctx.beginPath()
  for (let gx = 0; gx <= w; gx += 40) { ctx.moveTo(gx, 0); ctx.lineTo(gx, h) }
  for (let gy = 0; gy <= h; gy += 30) { ctx.moveTo(0, gy); ctx.lineTo(w, gy) }
  ctx.stroke()
  const dims = [[0, 1], [2, 1], [4, 2], [8, 2]] // [i, 频率系数]
  dims.forEach(([i, freq], di) => {
    ctx.strokeStyle = palette[di % palette.length]
    ctx.lineWidth = 1.6
    ctx.beginPath()
    for (let pos = 0; pos <= 50; pos++) {
      const x = (pos / 50) * (w - 8) + 4
      const v = i % 2 === 0 ? Math.sin(pos / Math.pow(10000, i / 8) * freq) : Math.cos(pos / Math.pow(10000, i / 8) * freq)
      const y = h / 2 - v * (h / 2 - 10)
      pos === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.fillStyle = lineC
    ctx.font = '12px "Patrick Hand", "Kaiti SC", cursive'
    ctx.fillText('维度 ' + i + '（波长' + (i === 0 ? '长' : i === 2 ? '中' : '短') + '）', 6 + di * 0, 14 + di * 16)
  })
}

// ---------- CNN vs Transformer 对比示意 ----------
function drawCompare() {
  const cnn = document.getElementById('cmpCnn')
  const attn = document.getElementById('cmpAttn')
  if (!cnn || !attn) return
  const dark = document.documentElement.getAttribute('data-theme') === 'dark'
  const ink = dark ? '#b8ae9e' : '#6b6257'
  const accent = dark ? '#ff8f5c' : '#c0452c'
  const glow = dark ? '#5cd4c0' : '#1f7a6d'
  ;[cnn, attn].forEach(c => {
    const dpr = window.devicePixelRatio || 1
    c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr
    const ctx = c.getContext('2d')
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, c.clientWidth, c.clientHeight)
    const w = c.clientWidth, h = c.clientHeight
    const n = 8
    const xs = [], ys = []
    for (let i = 0; i < n; i++) { xs.push((i + 0.5) / n * w); ys.push(h / 2) }
    // 方块
    xs.forEach((x, i) => {
      ctx.fillStyle = i === 3 ? accent : 'rgba(128,120,105,0.25)'
      ctx.fillRect(x - 14, ys[i] - 14, 28, 28)
      ctx.strokeStyle = ink; ctx.lineWidth = 1
      ctx.strokeRect(x - 14, ys[i] - 14, 28, 28)
      ctx.fillStyle = ink; ctx.font = '11px sans-serif'; ctx.textAlign = 'center'
      ctx.fillText(String(i + 1), x, ys[i] + 4)
    })
    if (c === cnn) {
      // CNN：第 4 个只连邻居 3、5
      ctx.strokeStyle = accent; ctx.lineWidth = 2
      ;[3, 5].forEach(j => { ctx.beginPath(); ctx.moveTo(xs[3], ys[3]); ctx.lineTo(xs[j - 1], ys[j - 1]); ctx.stroke() })
      ctx.fillStyle = accent; ctx.font = '12px sans-serif'
      ctx.fillText('看邻居', xs[0], 20)
    } else {
      // Attention：第 4 个连所有
      ctx.strokeStyle = glow; ctx.lineWidth = 1.2
      for (let j = 0; j < n; j++) { if (j === 3) continue; ctx.beginPath(); ctx.moveTo(xs[3], ys[3]); ctx.lineTo(xs[j], ys[j]); ctx.stroke() }
      ctx.fillStyle = glow; ctx.font = '12px sans-serif'
      ctx.fillText('看全场', xs[0], 20)
    }
  })
}

// ---------- 注意力手算引导 ----------
function bindCalc() {
  const btn = document.getElementById('attnCalcBtn')
  if (!btn) return
  btn.addEventListener('click', () => {
    const msg = document.getElementById('attnCalcMsg')
    if (!msg) return
    const steps = [
      '第 1 步：查表，把「我 爱 你」换成 Q 和 K（每个词两维）',
      '第 2 步：「我」的 Q 点所有 K：0.66、1.06、0.38 → 爱最相关',
      '第 3 步：除以 √2 ≈ 1.41 → [0.47, 0.75, 0.27] → softmax → [0.31, 0.46, 0.23]',
      '第 4 步：加权 V 求和 → 「我」吸收了 46% 的「爱」——上下文注入完成'
    ]
    msg.textContent = steps[Math.floor(Math.random() * steps.length)]
  })
}

// ---------- 入口 ----------
export function redrawTrans() {
  drawPosEncoding()
  drawCompare()
}

export function initTransformer(root) {
  if (!root) return
  root.innerHTML = TRANS_CONTENT
  redrawTrans()
  bindCalc()
  window.addEventListener('resize', () => {
    setTimeout(redrawTrans, 100)
  })
}