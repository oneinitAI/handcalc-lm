// ============================================================
// src/ach.js
// 手算LM —— 成就系统（游戏化玩法，localStorage 持久化）
// ============================================================

export const ACHIEVEMENTS = [
  { id: 'first', name: '初见', desc: '完成第一次训练（满 100 步）' },
  { id: 'learned', name: '学到位', desc: '训练到进度条满（loss 自动停）' },
  { id: 'crash', name: '训崩艺术家', desc: '把模型训练训崩（loss 飙升）' },
  { id: 'hot', name: '温度疯子', desc: '温度拉到 1.5 生成' },
  { id: 'cold', name: '温度保守派', desc: '温度拉到 0.1 生成' },
  { id: 'blind', name: '火眼金睛', desc: '盲测猜对一次' },
  { id: 'calc', name: '纸笔译者', desc: '显微镜手算答对一次' },
  { id: 'gen10', name: '高产作者', desc: '生成满 10 次' },
  { id: 'snap', name: '存档大师', desc: '保存一次模型快照' },
  { id: 'all', name: '三阶段通关', desc: '完成 预训练 → 微调 → 对齐 全旅程' },
  { id: 'hunter', name: '彩蛋猎人', desc: '发现 4 个不同的彩蛋' },
]

const KEY = 'handcalc:ach'

export function loadEarned() {
  try { return JSON.parse(localStorage.getItem(KEY) || '[]') } catch { return [] }
}
export function saveEarned(arr) {
  try { localStorage.setItem(KEY, JSON.stringify(arr)) } catch { /* 忽略 */ }
}