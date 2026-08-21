// ============================================================
// src/corpus.js
// 手算LM —— 内置现代文语料（3 篇，可切换 + 用户粘贴自定义）
// ============================================================

export const CORPUS = [
  {
    id: 'hetang',
    title: '荷塘月色 · 朱自清',
    desc: '现代散文经典——月光、荷塘、薄雾。语言优美，适合看模型学"意境"。',
    text: '月光如流水一般，静静地泻在这一片叶子和花上。薄薄的青雾浮起在荷塘里。叶子和花仿佛在牛乳中洗过一样；又像笼着轻纱的梦。虽然是满月，天上却有一层淡淡的云，所以不能朗照；但我以为这恰是到了好处——酣眠固不可少，小睡也别有风味的。月光是隔了树照过来的，高处丛生的灌木，落下参差的斑驳的黑影，峭楞楞如鬼一般；弯弯的杨柳的稀疏的倩影，却又像是画在荷叶上。',
  },
  {
    id: 'santi',
    title: '三体 · 刘慈欣',
    desc: '科幻——红岸基地、仰望星空。科技感文本，配"手稿算前沿"的反差。',
    text: '红岸基地的巨大天线矗立在半山腰，像一只仰望星空的钢铁巨手。叶文洁站在天线脚下，第一次感到自己是多么渺小。她想起父亲说过，人类之所以伟大，不在于制造了多少工具，而在于敢于仰望星空，并追问那些没有答案的问题。太阳即将落山，天边泛起一片绯红。她按下了那个按钮，然后，在静默中等待了八年。这八年里，她收到了那个来自宇宙深处的回信：不要回答！不要回答！不要回答！',
  },
  {
    id: 'shige',
    title: '现代诗 · 顾城与海子',
    desc: '现代诗——短、凝练、字少。训练最快，语感最强。',
    text: '黑夜给了我黑色的眼睛，我却用它寻找光明。我有一所房子，面朝大海，春暖花开。从明天起，做一个幸福的人，喂马，劈柴，周游世界。从明天起，关心粮食和蔬菜。那都是些普通的日子，像一棵草一样生长。我把石头还给石头，让胜利的胜利。风从远方来，带着海的气息，也带着你未说出口的话语。',
  },
]

// 特殊角色标记（SFT 用），用控制字符避免与正文冲突
export const USER = '\u0001'
export const ASSISTANT = '\u0002'
export const END = '\u0003'
export const SPECIAL = [USER, ASSISTANT, END]
export const TOKEN_NAME = { [USER]: '<u>', [ASSISTANT]: '<a>', [END]: '<e>' }

/** 从文本构建字符表（vocab 固定预留特殊 token 在前） */
export function buildVocab(text) {
  const chars = [...new Set(text.split(''))].sort()
  const all = [...SPECIAL, ...chars]
  const stoi = Object.fromEntries(all.map((c, i) => [c, i]))
  return { chars: all, stoi, itos: all, vocab: all.length }
}

/** token 序列转可读文本（特殊 token 显示为 <u>/<a>/<e>） */
export function tokensToText(itos, ids) {
  return ids.map((i) => TOKEN_NAME[itos[i]] ?? itos[i]).join('')
}
