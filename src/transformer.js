// ============================================================
// src/transformer.js
// AI 学习本 —— Transformer 深度解析板块（人话版）
// 比喻先行、术语即时解释、公式带翻译、互动拉满
// ============================================================

import { dictOpenDetail } from './dict.js'

// ---------- 教学内容（演草纸风格：章节=题目，就地解释，不跳术语表） ----------
const TRANS_P1 = `
<section class="card">
  <h2>壹 · 一句话先听懂</h2>
  <p>Transformer 是一个<b>开会讨论的会场</b>：输入里的每个词都举手发言，同时听全场所有人的发言，然后一起决定——下一个词最该是谁。</p>
  <p>给它一句话「我 爱 你」，它会这样工作：</p>
  <div class="row">
    <div class="mm-track mono" style="flex:1">我 → 爱 → 你（三个词同时进场，互相看一眼全场）</div>
  </div>
  <div class="howto">记住这个画面：<b>每个词都能一步看到全场</b>。这是它与所有老网络最本质的区别——后面每一节都在解释这句话。</div>
  <p class="muted">这一页把 Transformer 的每一颗螺丝拆给你看，<b>全程不说一个不解释的黑话</b>。文里反复出现的"手算 LM"，指的是你在「文本模型」tab 亲手训练的那个微型模型（LM = 语言模型 Language Model，就是"猜下一个字"的模型）——还没训练过也没关系，先看原理。</p>
</section>

<section class="card">
  <h2>贰 · 它为什么出生（2017 年之前的世界）</h2>
  <p>2017 年之前，做"猜下一个字"主要靠两类网络，各有<b>致命伤</b>：</p>
  <div class="pair">
    <div class="answer">
      <div class="blind-title">RNN（循环网络）＝ 逐字念经</div>
      <p>一个字一个字地读：读完「我」才读「爱」，读完「爱」才读「你」——<b>串行</b>，像念经一样排着队。</p>
      <p class="ms-slogan">致命伤 ① 慢：GPU 再强也只能一个字一个字跑。<br>致命伤 ② 健忘：第 1 个字的信息传到第 100 个字时早"淡出"了——长句子顾头不顾尾。</p>
    </div>
    <div class="answer">
      <div class="blind-title"><a class="gloss-link" data-gloss="CNN">CNN</a>（卷积）＝ 拿着放大镜扫</div>
      <p>用一个固定大小的"窗口"在序列上滑动，只看窗口里的几个邻居字。想看远处？把窗口加大，或者叠几十层。</p>
      <p class="ms-slogan">致命伤：窗口再大也有限。第 1 个字和第 100 个字的关系，要么堆几十层才够得着，要么直接看不见。</p>
    </div>
  </div>
  <div class="teach">
    <h3>Transformer 的破局思路</h3>
    <p>不逐字念、不拿放大镜扫——<b>让所有词同时互相看</b>。第 1 个字想看第 100 个字？<b>一步直达</b>，中间不经过任何中转。这就是<b><a class="gloss-link" data-gloss="注意力">自注意力</a></b>（Self-Attention，就是"自己看自己这句话里的每个词"）。</p>
    <p class="howto">论文标题《Attention Is All You Need》直译："你只需要注意力"——循环不要了，卷积也不要了，<b>只要注意力就够</b>。敢取这名字，是因为它真的做到了。</p>
  </div>
</section>

<section class="card">
  <h2>叁 · 和卷积网络（<a class="gloss-link" data-gloss="CNN">CNN</a>）比一比</h2>
  <p>CNN 和 Transformer 都做"把信息融合起来"，但哲学完全不同：</p>
  <div class="viz-row">
    <div class="viz">
      <div class="viz-title">CNN：放大镜扫图 <span class="tag">只看邻居</span></div>
      <canvas id="cmpCnn" class="canvas"></canvas>
      <p class="muted">第 4 个位置只能看到第 3、5 个邻居。想看远处 → 一层层叠。</p>
    </div>
    <div class="viz">
      <div class="viz-title">Transformer：高处看全图 <span class="tag">一步看全</span></div>
      <canvas id="cmpAttn" class="canvas"></canvas>
      <p class="muted">第 4 个位置一步连到所有位置，远近一视同仁。</p>
    </div>
  </div>
  <table class="ms-table" style="width:100%">
    <tr><th>比什么</th><th>CNN（放大镜）</th><th>Transformer（看全场）</th></tr>
    <tr><td><b>视野</b></td><td>只看得见窗口里几个邻居</td><td>一步看见整句所有词</td></tr>
    <tr><td><b>看远处的关系</b></td><td>得叠很多层，一层只传一步</td><td>一层就够，一步直达</td></tr>
    <tr><td><b>算得快不快</b></td><td>能并行，但要叠层补视野</td><td>天然并行（所有位置同时算）</td></tr>
    <tr><td><b>知不知道先后顺序</b></td><td>知道（窗口本身带位置形状）</td><td>不知道！必须手动加<b>座位号</b>（位置编码）</td></tr>
    <tr><td><b>谁在用它</b></td><td>图像底层特征（边缘、纹理）</td><td>语言、序列、大模型全部</td></tr>
    <tr><td><b>贵不贵</b></td><td>便宜（每个位置只算窗口里几个邻居）</td><td>贵（每个词要看每个词，句子越长越贵）</td></tr>
  </table>
  <div class="howto">注意最后一行："每个词看每个词"是注意力的软肋——句子越长越贵。后来一堆加速技术（Flash Attention、稀疏注意力等，大意就是"别白算那么多、只看该看的"）都在攻这个点（「前沿」tab 有讲）。</div>
</section>

<section class="card">
  <h2>肆 · 核心零件逐个拆（重头戏）</h2>
  <p>一个 Transformer 层由下面这些零件组成。我们一个一个拆，<b>每个都用"人话 + 动手"讲透</b>。</p>

  <div class="teach">
    <h3>零件 ① 词嵌入＝查字典翻译官</h3>
    <p><b>问题</b>：模型只会算数字，不会看汉字。怎么办？准备一本<b><a class="gloss-link" data-gloss="Embedding">大字典（嵌入表）</a></b>：每个字对应一串数字（叫<b>向量</b>，你可以理解成"这个字的身份证号"，不过这串号码分好几个<b>格子</b>，每个格子存一个数）。查表这一步，就是在给每个 <a class="gloss-link" data-gloss="Token">token</a>（模型眼里的一个字）编号。</p>
    <p class="mono">我 → [0.12, -0.34, 0.55]　爱 → [-0.21, 0.68, 0.10]　你 → [0.43, 0.05, -0.72]</p>
    <p><b>训练时这本字典会一直改</b>：意思相近的字，号码也越来越像（怎么比像？两个号码逐格做减法，差得少就算像）。去文本模型 tab 点「语义空间」按钮——它把每个字的号码画在平面上，你能亲眼看到"月光"和"荷塘"挤在一起（意思相近的字自己挨到一起，这叫<b>聚类</b>）。</p>
  </div>

  <div class="teach">
    <h3>零件 ② 位置编码＝给每个词发座位号</h3>
    <p><b>问题</b>：注意力不分先后——"我爱你"和"你爱我"在它眼里长得一模一样（都是"你 我 爱"三个字）。顺序丢了可不行。</p>
    <p><b>办法</b>：给每个位置的词加上一个<b>专属的座位号数字</b>，让它知道自己是第几个。论文用了一组正弦余弦函数（下面画给你看）：</p>
    <p class="mono">PE(pos, 2i)   = sin(pos / 10000^(2i/d))<br>PE(pos, 2i+1) = cos(pos / 10000^(2i/d))</p>
    <p class="howto"><b>公式翻译</b>：pos 是第几个位置，i 是号码的第几格，d 是号码总共几格。每个位置拿到一串"波形指纹"——<b>挨得近的位置指纹像，离得远的不像</b>（因为波是连续摆动的：第 3 格和第 4 格落在同一段波上，数值自然接近），模型就分得清先后了。</p>
    <div class="viz">
      <div class="viz-title">座位号长什么样（0→50 号位置，每个维度一条波）</div>
      <canvas id="posEncCanvas" class="canvas"></canvas>
      <p class="muted">同一个格子（维度）上，位置越近数值越接近——模型能读出"第 3 个和第 4 个挨着"。</p>
    </div>
  </div>

  <div class="teach">
    <h3>零件 ③ Q K V＝相亲的三样东西</h3>
    <p>每个词进会场前，都把自己的信息复制成三份（每份乘一个不同的"改数字机器"——<b>矩阵</b>你可以先当成一台固定规则的机器，把一串数字按规则变成另一串，规则是训练时调出来的，现在不用管细节），各自派上用场：</p>
    <div class="feat-grid">
      <div class="feat"><b>Q（Query 查询）</b><span>"我在找谁"——我的要求</span></div>
      <div class="feat"><b>K（Key 键）</b><span>"我能提供什么"——我的标签</span></div>
      <div class="feat"><b>V（Value 值）</b><span>"我真正的内容"——我本人</span></div>
    </div>
    <p class="howto">像相亲：<b>Q 是要求，K 是标签，V 是本人</b>。看两个词合不合，比的是 Q 和 K；真要融合内容，拿的是 V。</p>
  </div>

  <div class="teach">
    <h3>零件 ④ 自注意力＝开会讨论（核心中的核心）</h3>
    <p>每个词拿自己的 Q（要求）去对全场所有词的 K（标签）打分，分数越高越相关；然后按"几成"的比例把大家的 V（内容）混起来，得到自己的新理解。</p>
    <p class="howto"><b>公式翻译</b>（看一眼就好，下面有动手演示）：<span class="mono">Attention = <a class="gloss-link" data-gloss="Softmax">softmax</a>(Q·Kᵀ/√d) · V</span> —— 意思就是"Q 问 K 要分数 → 分数变占比 → 按占比拿 V"。</p>

    <div class="ms-step">
      <div class="ms-title">动手：点一个词，看它在"盯"谁</div>
      <p class="muted">下面假想的注意力权重（真实模型里就是这些数字，去「显微镜」tab 能看真家伙）。</p>
      <div id="attnDemo" class="attn-demo">
        <button class="attn-demo-token" data-t="我">我</button>
        <button class="attn-demo-token" data-t="爱">爱</button>
        <button class="attn-demo-token" data-t="你">你</button>
      </div>
      <div id="attnBars" class="attn-bars"></div>
      <p class="muted" id="attnNote">点上面的词试试。</p>
    </div>

    <div class="ms-step">
      <div class="ms-title">数字例子：从「我」的视角算一遍</div>
      <p class="muted">为了手算方便，下面每个向量只留 2 个格子（真实模型有几十上百格，算法一模一样）。</p>
      <p class="mono">我的 Q=[1.0, 0.2]，全场的 K：我[0.5,0.8] 爱[1.0,0.3] 你[0.2,0.9]<br>点积（逐格相乘再相加，看合不合拍）：<br>我和我: 0.5+0.16=0.66　我和爱: 1.0+0.06=1.06　我和你: 0.2+0.18=0.38<br>→ 「爱」得分最高——主谓宾里"我"和"爱"本来就挨得近。</p>
      <div class="row">
        <button id="attnCalcBtn" class="btn ghost" title="一步一步跟着算，每点一次走一步">一步一步算</button>
        <span id="attnCalcMsg" class="muted"></span>
      </div>
    </div>
  </div>

  <div class="teach">
    <h3>零件 ⑤ <a class="gloss-link" data-gloss="Softmax">softmax</a>＝把分数变成"几成"</h3>
    <p>分数是原始得分（0.66、1.06、0.38），加起来不等于 1，不好直接当比例用。<b>softmax 就是把一堆分数变成"占比"</b>——大的更大、小的更小，而且加起来正好 100%。</p>
    <div class="ms-step">
      <div class="ms-title">动手：拖滑杆改分数，看它变成几成</div>
      <p class="muted">三个滑杆是「我 爱 你」的原始得分（越大=越相关）。</p>
      <div class="row" id="softRows">
        <label>我 <input id="sm0" type="range" min="0" max="100" value="30"></label>
        <label>爱 <input id="sm1" type="range" min="0" max="100" value="60"></label>
        <label>你 <input id="sm2" type="range" min="0" max="100" value="20"></label>
      </div>
      <div id="softBars" class="soft-bars"></div>
      <p class="muted" id="softNote">注意：得分最高的"爱"占了最大头——这就是模型"注意力"的分配方式。</p>
    </div>
  </div>
</section>
`

const TRANS_P2 = `
<section class="card">
  <div class="teach">
    <h3>零件 ⑥ 多头注意力＝8 个侦探分头查</h3>
    <p>一个头只看一种关系，太单调。论文让模型用<b>多组 Q/K/V（多头）</b>同时算，每组头管一种关系：</p>
    <div class="feat-grid">
      <div class="feat"><b>头 1</b><span>管语法：谁修饰谁</span></div>
      <div class="feat"><b>头 2</b><span>管指代："它"指的是谁</span></div>
      <div class="feat"><b>头 3</b><span>管位置：前后隔多远</span></div>
      <div class="feat"><b>头 4 到 8</b><span>管各种语义关系……</span></div>
    </div>
    <p>最后把所有头的结果拼起来，再过一层"合并机器"（<b>线性变换</b>，就是把拼起来的结果再整体加工一遍）——<b>一次会议，多个视角同时看</b>。</p>
    <p class="howto">你的手算 LM 也是多头（去文本模型 tab 看配置）；「注意力直播」能看见每个头在"盯"谁。</p>
  </div>

  <div class="teach">
    <h3>零件 ⑦ 前馈网络 FFN＝散会后各自消化</h3>
    <p>注意力管<b>词与词之间</b>的消息交换；散会后每个词还要<b>自己回座位再想一步</b>——这就是 FFN（前馈网络）：一个两层的小机器，先把信息放大、再缩回来，每个词独立过一遍。</p>
    <p class="howto">公式翻译：<span class="mono">FFN(x) = 第二层( 激活( 第一层(x) ) )</span>——"第一层加工 → 过一道<b>激活</b>（就是把负数压一压，让模型能表达更丰富的东西）→ 第二层加工"，总之就是"先加工再加工"的两步处理。注意力和 FFN 交替叠起来，就是完整的一层。</p>
  </div>

  <div class="teach">
    <h3>零件 ⑧ 残差连接 + LayerNorm＝传送带 + 校平器</h3>
    <p>模型叠很多层时，数字会越传越歪、越传越大，训练直接崩。两个保险：</p>
    <div class="feat-grid">
      <div class="feat"><b>残差连接（传送带）</b><span>把原始输入原样带到后面"加"上——信息不丢（重要内容永远有直达通道），训练信号（梯度，就是教模型往哪调的指令）也传得顺。</span></div>
      <div class="feat"><b>LayerNorm（校平器）</b><span>每算完一层，把整层数字拉回"正常范围"——不膨胀、不塌缩。</span></div>
    </div>
    <p class="howto">没有这两样，几十层叠下去数值就漂了。你的手算 LM 每层也是"传送带 + 校平器"结构。</p>
  </div>

  <div class="teach">
    <h3>零件 ⑨ 拼成一层（数据流总览）</h3>
    <p class="mono">字 → 查字典(嵌入) + 发座位号(位置编码) → [开会(多头注意力) → 传送带+校平] → [各自消化(FFN) → 传送带+校平] → 出来一个"下一个字"的候选分数</p>
    <p>上面那块叠 N 次（论文叠 6 层，你的手算 LM 叠 1 层）——叠得越深，理解越抽象。</p>
  </div>
</section>

<section class="card">
  <h2>伍 · Encoder 和 Decoder＝读题的人和答题的人</h2>
  <p>论文原版做翻译，所以分两半：</p>
  <div class="pair">
    <div class="answer">
      <div class="blind-title">Encoder 编码器（读题）</div>
      <p>把整句原文从头到尾读一遍，每个位置都能看全句，产出"理解向量"（读懂了的意思）。</p>
    </div>
    <div class="answer">
      <div class="blind-title">Decoder 解码器（答题）</div>
      <p>一个字一个字写译文。写的时候有两个注意力：一个看自己已经写的部分（<b>masked，蒙住未来</b>），一个去看 Encoder 的理解（交叉注意力）。</p>
    </div>
  </div>
  <p class="howto"><b>mask 翻译</b>：就是"作弊禁止"——写「我爱」的时候不许偷看「你」，因为现实中未来还没发生。今天的大模型只有 Decoder（只管生成），结构更简单，但"不能偷看未来"这条规矩一模一样。</p>
</section>

<section class="card">
  <h2>陆 · 论文深度解析：《Attention Is All You Need》</h2>
  <div class="ms-step">
    <div class="ms-title">档案</div>
    <table class="ms-table">
      <tr><td><b>标题</b></td><td>Attention Is All You Need（只需要注意力）</td></tr>
      <tr><td><b>作者</b></td><td>Ashish Vaswani 等 8 人（Google Brain / Google Research）</td></tr>
      <tr><td><b>时间</b></td><td>2017 年 6 月，arXiv:1706.03762</td></tr>
      <tr><td><b>一句话贡献</b></td><td>提出 Transformer：只用注意力做序列建模，彻底扔掉循环和卷积</td></tr>
    </table>
  </div>
  <div class="ms-step">
    <div class="ms-title">它回答的问题</div>
    <p>2017 年机器翻译的主流是"RNN + 注意力"（先念经，再用注意力对齐）。但 RNN 天生串行——GPU 再多也只能一个字一个字跑，长句还会健忘。作者们反问：<b>既然注意力这么好用，为什么还要循环？</b></p>
  </div>
  <div class="ms-step">
    <div class="ms-title">它的三个关键设计</div>
    <p>① <b>缩放点积注意力</b>（Q 问 K 打分 → softmax → 拿 V），比老式算法快、省内存。<br>② <b>多头注意力</b>（多组同时查，一次看多种关系）。<br>③ <b>位置编码</b>（sin/cos 座位号）——扔掉循环和卷积后，顺序必须显式给。</p>
  </div>
  <div class="ms-step">
    <div class="ms-title">实验结果（当时的"世界纪录"）</div>
    <table class="ms-table">
      <tr><th>任务</th><th>成绩</th><th>意义</th></tr>
      <tr><td>WMT 英→德翻译</td><td>BLEU 28.4</td><td>当时最强，超过所有 RNN</td></tr>
      <tr><td>WMT 英→法翻译</td><td>BLEU 41.8</td><td>单模型最强，成本还更低</td></tr>
      <tr><td>训练速度</td><td>8 块 GPU 约 3.5 天</td><td>比当时最强 RNN 快 3 倍以上</td></tr>
    </table>
    <p class="howto"><b>BLEU 翻译</b>：机器翻译的"考试打分"，看译文和标准答案重合多少。<b>世界纪录</b>就是 SOTA——当时没有比它更好的。</p>
    <p class="muted">快，因为所有位置同时算（并行）；强，因为长距离一步直达。速度和精度双赢，横扫翻译榜。</p>
  </div>
  <div class="ms-step">
    <div class="ms-title">为什么它是现代大模型的基石</div>
    <p>论文发表后，两条路都从它长出来：<br>① <b>BERT（2018）</b>：只用 Encoder + 预训练（先读海量书打基础）→ 语言理解全面刷榜。<br>② <b>GPT（2018 起）</b>：只用 Decoder + 预训练 → 生成能力一路放大 → GPT-3 → ChatGPT → 今天的整个大模型时代。</p>
    <p class="howto">今天所有大模型——GPT、Claude、Gemini、DeepSeek、豆包——架构上都是这篇论文的子孙：<b>堆叠的 Transformer 层，只有注意力，没有循环，没有卷积</b>。它就是现代大模型的"出生证明"。</p>
    <div class="egg">彩蛋：这标题是作者们的宣言——"你只需要注意力"，连增强 RNN 的注意力都不要，直接纯注意力开局。</div>
  </div>
</section>

<section class="card">
  <h2>柒 · 下一步：去亲手摸它</h2>
  <p>这一页讲的是"图纸"，纸上谈兵到此为止——<b>还没训练过模型也没关系</b>，真家伙就在你的浏览器里，照着下面动手就行：</p>
  <div class="feat-grid">
    <div class="feat"><b>文本模型 tab</b><span>训练一个真正的微型 Transformer，看损失曲线一路下降。</span></div>
    <div class="feat"><b>注意力直播</b><span>生成时逐 token 看 Q/K/V 和注意力分数（同款多头！）。</span></div>
    <div class="feat"><b>显微镜 tab</b><span>把"猜下一个字"拆成 6 步手算，连注意力矩阵都列给你。</span></div>
    <div class="feat"><b>语义空间</b><span>把每个字的号码画在平面上，意思相近的字自己挨到一起（聚类）——看字典长什么样。</span></div>
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
  if (dark) { palette[0] = '#ff8f5c'; palette[1] = '#5cd4c0'; palette[2] = '#d6a08a'; palette[3] = '#8a8172' }
  ctx.strokeStyle = 'rgba(0,0,0,0.15)'; ctx.beginPath()
  for (let gx = 0; gx <= w; gx += 40) { ctx.moveTo(gx, 0); ctx.lineTo(gx, h) }
  for (let gy = 0; gy <= h; gy += 30) { ctx.moveTo(0, gy); ctx.lineTo(w, gy) }
  ctx.stroke()
  const dims = [[0, 1], [2, 1], [4, 2], [8, 2]]
  dims.forEach(([i, freq], di) => {
    ctx.strokeStyle = palette[di % palette.length]
    ctx.lineWidth = 1.6
    ctx.beginPath()
    for (let pos = 0; pos <= 50; pos++) {
      const x = (pos / 50) * (w - 8) + 4
      const v = i % 2 === 0 ? Math.sin((pos / Math.pow(10000, i / 8)) * freq) : Math.cos((pos / Math.pow(10000, i / 8)) * freq)
      const y = h / 2 - v * (h / 2 - 10)
      pos === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.fillStyle = lineC
    ctx.font = '12px "Patrick Hand", "Kaiti SC", cursive'
    ctx.fillText('格子 ' + i + '（波长' + (i === 0 ? '长' : i === 2 ? '中' : '短') + '）', 6, 14 + di * 16)
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
    for (let i = 0; i < n; i++) { xs.push(((i + 0.5) / n) * w); ys.push(h / 2) }
    xs.forEach((x, i) => {
      ctx.fillStyle = i === 3 ? accent : 'rgba(128,120,105,0.25)'
      ctx.fillRect(x - 14, ys[i] - 14, 28, 28)
      ctx.strokeStyle = ink; ctx.lineWidth = 1
      ctx.strokeRect(x - 14, ys[i] - 14, 28, 28)
      ctx.fillStyle = ink; ctx.font = '11px sans-serif'; ctx.textAlign = 'center'
      ctx.fillText(String(i + 1), x, ys[i] + 4)
    })
    if (c === cnn) {
      ctx.strokeStyle = accent; ctx.lineWidth = 2
      ;[3, 5].forEach(j => { ctx.beginPath(); ctx.moveTo(xs[3], ys[3]); ctx.lineTo(xs[j - 1], ys[j - 1]); ctx.stroke() })
      ctx.fillStyle = accent; ctx.font = '12px sans-serif'
      ctx.fillText('看邻居', xs[0], 20)
    } else {
      ctx.strokeStyle = glow; ctx.lineWidth = 1.2
      for (let j = 0; j < n; j++) { if (j === 3) continue; ctx.beginPath(); ctx.moveTo(xs[3], ys[3]); ctx.lineTo(xs[j], ys[j]); ctx.stroke() }
      ctx.fillStyle = glow; ctx.font = '12px sans-serif'
      ctx.fillText('看全场', xs[0], 20)
    }
  })
}

// ---------- softmax 滑杆演示 ----------
const SOFT_TOKENS = ['我', '爱', '你']
function initSoftmax() {
  const rows = ['sm0', 'sm1', 'sm2']
  const bars = document.getElementById('softBars')
  const note = document.getElementById('softNote')
  if (!bars || !note) return
  function softmax(arr) {
    const mx = Math.max(...arr)
    const ex = arr.map(v => Math.exp(v - mx))
    const s = ex.reduce((a, b) => a + b, 0)
    return ex.map(v => v / s)
  }
  function render() {
    const vals = rows.map(id => parseFloat(document.getElementById(id).value))
    const probs = softmax(vals)
    bars.innerHTML = ''
    probs.forEach((p, i) => {
      const d = document.createElement('div')
      d.className = 'soft-bar'
      const fill = document.createElement('div')
      fill.className = 'fill'
      fill.style.height = Math.max(4, Math.round(p * 100)) + '%'
      const label = document.createElement('div')
      label.textContent = SOFT_TOKENS[i] + ' ' + (p * 100).toFixed(0) + '%'
      d.appendChild(fill)
      d.appendChild(label)
      bars.appendChild(d)
    })
    const mxIdx = vals.indexOf(Math.max(...vals))
    note.textContent = '得分最高的是「' + SOFT_TOKENS[mxIdx] + '」，占了最大头——模型把注意力主要给了它。'
  }
  rows.forEach(id => {
    const el = document.getElementById(id)
    if (el) el.addEventListener('input', render)
  })
  render()
}

// ---------- 点词看注意力 ----------
const ATTN_DATA = {
  '我': { '我': 0.31, '爱': 0.46, '你': 0.23 },
  '爱': { '我': 0.20, '爱': 0.30, '你': 0.50 },
  '你': { '我': 0.25, '爱': 0.35, '你': 0.40 }
}
const ATTN_NOTE = {
  '我': '「我」认为「爱」最值得注意——主语最关心动词，这样才知道自己要干什么。',
  '爱': '「爱」把最大注意力给了「你」——动词最关心宾语（爱的对象）。',
  '你': '「你」比较平均，但最在意「爱」——宾语也要回看动词，句子才完整。'
}
function initAttnDemo() {
  const demo = document.getElementById('attnDemo')
  const bars = document.getElementById('attnBars')
  const note = document.getElementById('attnNote')
  if (!demo || !bars || !note) return
  function show(t) {
    demo.querySelectorAll('.attn-demo-token').forEach(b => b.classList.toggle('on', b.dataset.t === t))
    const w = ATTN_DATA[t]
    bars.innerHTML = ''
    Object.keys(w).forEach(k => {
      const d = document.createElement('div')
      d.className = 'attn-bar'
      const fill = document.createElement('div')
      fill.className = 'fill'
      fill.style.height = Math.max(4, Math.round(w[k] * 100)) + '%'
      const label = document.createElement('div')
      label.textContent = k + ' ' + (w[k] * 100).toFixed(0) + '%'
      d.appendChild(fill)
      d.appendChild(label)
      bars.appendChild(d)
    })
    note.textContent = ATTN_NOTE[t]
  }
  demo.querySelectorAll('.attn-demo-token').forEach(b => b.addEventListener('click', () => show(b.dataset.t)))
  show('我')
}

// ---------- 注意力手算：一步一步 ----------
function bindCalc() {
  const btn = document.getElementById('attnCalcBtn')
  const msg = document.getElementById('attnCalcMsg')
  if (!btn || !msg) return
  const steps = [
    '第 1/4 步：查字典，把「我 爱 你」换成数字向量（每个词 2 个数）。',
    '第 2/4 步：「我」的 Q（要求）去对全场的 K（标签）打分：0.66、1.06、0.38 → 「爱」最合拍。',
    '第 3/4 步：softmax 把分数变成占比 → 「我」的注意力分配：31%、46%、23%。',
    '第 4/4 步：按占比拿 V 混合 → 「我」的新理解里混进了 46% 的「爱」——上下文注入完成。'
  ]
  let i = 0
  btn.addEventListener('click', () => {
    msg.textContent = steps[i]
    i = (i + 1) % steps.length
  })
}

// ---------- 入口 ----------
export function redrawTrans() {
  drawPosEncoding()
  drawCompare()
}

export function initTransformer(root) {
  if (!root) return
  root.innerHTML = TRANS_P1 + TRANS_P2
  // 术语链接：点击跳到词典看动画
  root.querySelectorAll('.gloss-link').forEach((a) => {
    a.addEventListener('click', (e) => { e.preventDefault(); dictOpenDetail(a.dataset.gloss) })
  })
  redrawTrans()
  bindCalc()
  initSoftmax()
  initAttnDemo()
  window.addEventListener('resize', () => {
    setTimeout(redrawTrans, 100)
  })
}