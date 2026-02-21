# Career Path Agent 前端 UI 实现交付文档

## 1. 项目目标
- 将当前 `CLI` 对话式职业规划流程升级为可交互的 Web 前端。
- 保持后端既有对话逻辑不变（`workflow/graph_chat.py`），前端负责体验、状态展示、输入输出编排。
- 支持从“信息采集 -> 追问确认 -> 生成最终方案”的完整闭环。

## 2. 当前后端事实（必须对齐）
- 当前入口为 `main.py`，运行模式是命令行多轮对话。
- 核心状态来源：`user_state`、`progress_state`、`memory`、`goal_constraints`、`evaluation`。
- 会话阶段字段：`question_stage`，取值为：
  - `collecting`：基础信息采集
  - `confirming`：中等置信度信息确认（yes/no）
  - `targeted`：定向追问（可选信息补齐）
  - `done`：问答阶段结束，进入最终方案
- 对话结束标记：`conversation_complete=true`。

## 3. 前端实现范围
1. 单页主流程（必须）
- 聊天区（用户/助手消息）
- 输入区（文本输入 + 发送）
- 右侧状态区（已提取画像、缺失字段、当前阶段、进度）
- 结果区（最终职业方案渲染）

2. 会话管理（必须）
- 初始化会话后自动展示首条助手问题。
- 每轮提交用户输入，渲染最新助手回复。
- `conversation_complete=true` 时切换到“最终方案展示态”。

3. 追问与确认（必须）
- 当 `question_stage=confirming` 时，UI 显示“确认模式”样式（突出 yes/no）。
- 展示 `pending_slots`、`current_slot`，让用户知道系统在补哪些信息。

4. RAG 信息感知（建议）
- 若返回 `rag_context` 非空，可在侧栏提供“参考依据”折叠面板。

## 4. 信息架构（页面）
- 页面 A：`/chat`
  - Header：项目名 + 会话状态
  - Main Left：消息流
  - Main Right：用户画像与流程进度
  - Footer：输入框与发送按钮
- 页面 B：`/result`（可选，或在 A 页内切换）
  - 展示最终方案全文（`last_answer`）
  - 展示结构化摘要（从 `memory`、`goal_constraints` 提取）

## 5. 核心字段映射（前端必须消费）
- `messages`: 渲染聊天记录（role: `user/assistant/system`）
- `next_question`: 当前轮系统准备发出的下一个问题（可用于 UI 提示）
- `question_stage`: 当前阶段（驱动状态标签）
- `progress_state.pending_slots`: 未完成必填槽位
- `progress_state.current_slot`: 当前重点采集槽位
- `missing_fields`: 后端评估出的缺失字段
- `memory`: 当前冻结/累计的用户画像
- `goal_constraints`: 目标与约束结构体
- `conversation_complete`: 是否完成会话
- `last_answer`: 最终方案正文

## 6. 对接接口约定（建议，供前后端联调）
说明：当前仓库暂无 HTTP 服务层，前端先按以下契约开发，后端再补 API 适配。

1. `POST /api/chat/session`
- 作用：创建会话并返回首条问题
- 请求体：
```json
{}
```
- 响应体（示例）：
```json
{
  "session_id": "uuid",
  "messages": [{"role": "assistant", "content": "What is your education background and major?"}],
  "question_stage": "collecting",
  "conversation_complete": false,
  "user_state": {},
  "progress_state": {}
}
```

2. `POST /api/chat/turn`
- 作用：提交用户消息并返回本轮更新状态
- 请求体：
```json
{
  "session_id": "uuid",
  "message": "我目前是统计学本科，每周可以投入10小时"
}
```
- 响应体（示例）：
```json
{
  "messages": [
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "next_question": "What is your current experience level (entry/junior/mid/senior)?",
  "question_stage": "collecting",
  "conversation_complete": false,
  "user_state": {"profile": {}},
  "progress_state": {"pending_slots": ["experience_level", "location"], "current_slot": "experience_level"},
  "memory": {},
  "goal_constraints": {},
  "missing_fields": ["experience_level", "location"]
}
```

3. `GET /api/chat/session/{session_id}`（可选）
- 作用：刷新后恢复会话状态

## 7. 前端状态管理要求
- 必须本地维护：
  - `session_id`
  - `messages`
  - `question_stage`
  - `conversation_complete`
  - `user_state/profile`
  - `progress_state`
- 发送按钮在请求中禁用，避免并发提交同一轮。
- 出错时保留输入框内容，支持重试。

## 8. UI/交互验收标准
1. 可完整走通一次会话，最终拿到 `last_answer` 并展示。
2. 不同 `question_stage` 有明显视觉差异（collecting/confirming/targeted/done）。
3. 侧栏能实时展示 `pending_slots` 与已提取画像。
4. 桌面端与移动端可用（移动端消息区不溢出、输入框可用）。
5. 接口失败有错误提示，不丢历史消息。

## 9. 技术建议（前端）
- 框架：`React + TypeScript + Vite`
- 数据请求：`axios` 或 `fetch`（封装 API client）
- 状态管理：轻量优先（`zustand` 或 React Context）
- 样式：模块化 CSS 或 Tailwind，保证响应式。

## 10. 实施里程碑（建议）
1. Day 1：项目初始化 + 页面骨架 + Mock 数据联调
2. Day 2：会话状态流、消息渲染、输入提交流程
3. Day 3：侧栏画像/进度、阶段样式、错误处理
4. Day 4：真接口联调、修复边界、移动端适配
5. Day 5：验收与文档补全

## 11. 非本期范围
- 不在前端实现职业规划算法与模型推理。
- 不在前端做 RAG 检索逻辑，仅做展示。
- 不改动 `workflow` 的业务决策规则。
