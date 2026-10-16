# AI 模拟面试官（RAG + Agent）- 产品需求文档

## Overview

- **Summary**：构建一个"AI 模拟面试官"应用。用户上传简历（PDF/Markdown）与目标岗位 JD（或选择内置示例），系统基于高级 RAG 管线（查询改写 → 向量+BM25 混合检索 → RRF 融合 → Rerank 重排 → Agentic 自评重检索）召回简历、JD 与八股题库内容，由 LangGraph 编排的面试 Agent 完成"开场 → 结合简历出题 → 根据回答动态追问 → 多维评分 → 生成改进报告"的完整模拟面试流程。FastAPI 提供 SSE 流式接口，Streamlit 提供 Web 界面，Docker Compose 一键部署到云服务器并公网可访问。
- **Purpose**：作为求职作品集项目，在 AI 应用开发/后端岗位面试中证明具备 RAG 深度优化、Agent 编排、工程化评测与云端部署的完整 0→1 能力；同时本人求职前可用它做真实模拟训练。
- **Target Users**：求职者本人（演示与训练用）；面试官/评审者（查看项目完成度与工程质量）。

## Goals

- RAG 不是"切块+向量"的玩具 demo：具备查询改写、混合检索、重排、检索质量自评与重检索，且答案带证据溯源。
- Agent 有真实必要性：面试流程为有状态图编排，支持基于候选人回答的动态追问、轮次控制、结构化评分。
- 有量化评测：RAGAS 评测集 + 基线指标，可回归。
- 完整工程闭环：配置管理、会话持久化、可观测（可选）、pytest 测试、Docker Compose、云服务器部署脚本、示例数据开箱即用。

## Non-Goals

- 不做用户注册登录/多租户权限体系（单用户/演示级，不持久化敏感个人信息）。
- 不做语音面试、摄像头、实时音视频。
- 不做爬虫自动抓取题库/招聘网站 JD。
- 不做模型微调与本地 LLM 部署（LLM 全部走 DeepSeek API）。
- 不做域名、HTTPS、ICP 备案与生产级高可用（公网 `IP:端口` 访问即可）。
- 不做移动端原生 App。

## Background & Context

- 目录当前为空，从零初始化。开发环境为 Windows + PowerShell 5；部署目标为 Linux 云服务器。
- 已确认技术决策：
  1. Embedding/Rerank 使用**硅基流动 SiliconFlow**（OpenAI 兼容）：`BAAI/bge-m3` 向量、`BAAI/bge-reranker-v2-m3` 重排；
  2. Chat/推理使用 **DeepSeek**（OpenAI 兼容接口）；
  3. 部署到**用户自有的云服务器**，Docker Compose 编排，公网 IP:端口访问；
  4. 前端使用 **Streamlit**；
  5. 向量库使用 **Qdrant**（容器化）；关键词检索使用 BM25（本地实现，如 rank_bm25）；
  6. 编排使用 **LangGraph**；评测使用 **RAGAS**；依赖管理使用 **uv**；Python 3.11；
  7. Langfuse 链路追踪为**可选开关**，未配置时自动降级为本地结构化日志。
- 历史项目经验要求规避：前后端字段名不一致、错误信息不透出、API 可用性结论必须真实调用验证、密钥不得硬编码或提交入库。

## Functional Requirements

- **FR-1 项目骨架与配置**：通过 `uv sync` 完成依赖安装；`.env.example` 声明全部配置项（DeepSeek/SiliconFlow 的 key、base_url、模型名、Qdrant 地址、Langfuse 开关等）；配置集中加载并做启动期校验（缺失关键 key 给出明确报错）；提供 `/health` 健康检查。
- **FR-2 知识库摄入**：支持 PDF、Markdown/TXT 文档；按"简历 / JD / 题库"三类知识库（kb_type）管理；文档经解析、结构化切块（带 overlap、保留标题层级元数据）后写入 Qdrant；每条 chunk 含 `chunk_id / source / kb_type / heading / text` 元数据；提供命令行摄入脚本与内置示例数据（示例简历、模拟 JD、Python/后端/LLM 八股题库 30-50 条），支持重建索引（幂等）。
- **FR-3 模型客户端**：封装 DeepSeek 聊天客户端与 SiliconFlow embedding/rerank 客户端（OpenAI 兼容协议），支持按用途配置模型名/温度（出题 Agent、评分 Agent 可不同参数），统一超时、重试与可诊断的错误信息（区分 401 key 错误 / 429 限流 / 模型不存在）。
- **FR-4 高级 RAG 检索**：
  - 查询改写：支持 Multi-Query（生成多个改写查询）与 HyDE（假设性答案）两种策略，可配置开关；
  - 混合检索：向量检索（bge-m3 + Qdrant）与 BM25 关键词检索并行召回，使用 RRF 融合排序；
  - Rerank：融合结果送 bge-reranker-v2-m3 精排，取 Top-K；
  - 支持按 `kb_type` 元数据过滤（如"只在简历+JD 中检索"）；
  - 返回结果携带溯源字段（source/heading/chunk_id/各阶段分数）。
- **FR-5 Agentic RAG（检索质量自评）**：出题/追问前，由 LLM 评判"检索上下文是否足以支撑提问/回答"；不足时自动改写查询并重新检索，设最大重试次数与终止条件；整个决策过程在 trace/日志中可见。
- **FR-6 面试 Agent（LangGraph）**：状态机包含节点：`开场 → 出题（结合 RAG 证据）→ 候选人回答 → 回答质量研判 → 动态追问（或进入下一题）→ 轮次/结束判断 → 多维评分 → 报告生成`；支持可配置的面试时长/题目数量与考察方向；同一问题最多追问 2 轮；会话状态持久化，进程重启后可从中断点继续。
- **FR-7 评分报告**：面试结束输出结构化报告：维度（如岗位匹配度、基础知识、项目深度、表达逻辑、问题解决）各 1-5 分、每个评分必须关联面试中的具体问答证据、亮点与不足、针对性改进建议与推荐复习知识点（可回链题库来源）；输出经 Pydantic 校验，LLM 返回非法结构时自动重试。
- **FR-8 FastAPI 后端与流式**：提供创建会话、上传/选择知识库、发送候选人回答（SSE 流式返回面试官 token 与阶段事件）、提前结束、获取报告等接口；统一错误响应结构；接口 schema 与前端共享同一份 Pydantic 定义。
- **FR-9 Streamlit 前端**：页面包含：① 开始页（上传简历/JD 或选择示例、选择岗位方向与题目数、填写 API key 或使用服务端环境变量）；② 面试页（对话式 UI、流式输出、显示当前阶段）；③ 报告页（雷达图/维度条、证据引用、改进建议）；所有后端错误必须在界面可见且可读。
- **FR-10 评测**：内置 ≥40 条评测集（问题 + 期望命中来源/参考答案要点），用 RAGAS 计算 faithfulness、context recall/precision、answer relevancy，并提供一键运行脚本输出指标报告；记录基线数值供回归对比。
- **FR-11 可观测性**：Langfuse 配置开关：有 key 时上报完整 RAG/Agent 链路（检索、改写、rerank、节点跳转、LLM 调用）；无 key 时降级为本地结构化日志文件，关键步骤均可排查。
- **FR-12 部署上线**：提供 `Dockerfile`（后端+Streamlit 镜像）与 `docker-compose.yml`（app + qdrant 卷持久化）；提供云服务器初始化/部署脚本与文档；在干净 Linux 主机上部署后，公网 `IP:端口` 可完整走通一次模拟面试；密钥仅通过服务器环境变量/`.env`（不入库）注入。

## Non-Functional Requirements

- **NFR-1 性能**：单次混合检索（含 rerank）P95 ≤ 3s（题库规模 ≤ 2000 chunk）；SSE 首 token 时间 ≤ 2s（受 DeepSeek API 影响除外，需在日志中分段计时）。
- **NFR-2 可靠性**：外部 API 调用具备超时（≥30s 流式）、指数退避重试（最多 2 次）与友好错误；索引重建幂等不产生重复数据。
- **NFR-3 安全**：所有密钥仅从环境变量读取，代码与示例文件不含真实 key；`.env` 加入 `.gitignore`；上传文件做类型/大小限制（PDF ≤ 10MB）。
- **NFR-4 可移植性**：Windows 可直接本地开发运行（路径处理使用 pathlib，不依赖 Linux-only 特性）；Linux 容器可部署。
- **NFR-5 可测试性**：核心模块（RRF 融合、切块、状态机分支、报告结构、API）具备 pytest 用例；LLM/外部调用使用 mock 或标记集成测试；核心工具函数测试覆盖率 ≥ 70%。
- **NFR-6 可维护性**：分层清晰（config / core clients / rag / agent / api / ui / eval），模块间通过接口/schema 耦合；README 含架构图、快速开始、接口说明、面试讲解话术要点。

## Constraints

- **Technical**：Python 3.11；LLM 仅用 DeepSeek API；Embedding/Rerank 仅用 SiliconFlow API；向量库 Qdrant；编排 LangGraph；前端 Streamlit；容器 Docker Compose；开发机 Windows PowerShell 5（脚本需兼容，不使用 `&&` 等 Bash 语法）。
- **Business**：1-2 周（约 10 个工作日）内完成 0→1 并上线；定位为作品集项目，非生产 SaaS。
- **Dependencies**：DeepSeek API key（用户提供）；SiliconFlow API key（用户注册获取，有免费额度）；云服务器（用户提供，建议 ≥ 2 vCPU / 4GB RAM / 20GB 磁盘，已装 Docker，开放访问端口）；示例简历/JD/题库随仓库提供。

## Assumptions

- 用户具备基础 Python 与命令行能力，能自行注册两个平台并创建 API key。
- SiliconFlow 的 `bge-m3` 与 `bge-reranker-v2-m3` 在项目期内保持 OpenAI 兼容接入与可用免费额度；若模型 ID 变更，仅需改配置项。
- 云服务器为 Linux（Ubuntu 22.04 为佳），可用密码/SSH key 登录；域名与备案不在范围内，公网 IP + 端口可接受。
- 评测 LLM 评判默认复用 DeepSeek（RAGAS 的 LLM/Embedding 可配置）。
- 上传内容为用户本人简历，不涉及他人隐私数据；数据存储于服务器 Docker 卷，不做额外加密。

## Acceptance Criteria

### AC-1: 开发环境一键就绪
- **Type**: `rule`
- **Given**: 一台安装 Python 3.11 与 uv 的 Windows 机器，代码已克隆
- **When**: 执行 `uv sync`、复制 `.env.example` 为 `.env` 并填入两个 API key、`docker compose up -d qdrant`、运行摄入脚本后启动后端
- **Then**: 依赖安装无报错；Qdrant 集合创建成功且包含示例数据 chunk；`GET /health` 返回 200 及各组件状态
- **Pass Condition**: 在干净虚拟环境按 README 步骤执行，全部命令成功且 `/health` 为 200
- **Evidence**: 命令执行输出截图/日志、`/health` 响应体

### AC-2: 知识摄入正确且幂等
- **Type**: `rule`
- **Given**: 内置示例简历、模拟 JD 与题库（30-50 条）
- **When**: 执行摄入脚本并再执行一次
- **Then**: Qdrant 中 chunk 数两次一致（幂等，无重复）；每条 chunk 含 chunk_id/source/kb_type/heading/text 元数据；PDF 与 Markdown 均被正确解析
- **Pass Condition**: 集合点数两次相同且 > 0，抽样 chunk 元数据字段完整
- **Evidence**: 摄入脚本日志、Qdrant count/scroll 查询结果、pytest 用例

### AC-3: 混合检索管线正确工作
- **Type**: `rule`
- **Given**: 已完成索引的知识库
- **When**: 以一个岗位相关问题调用检索接口（开启改写+混合+rerank）
- **Then**: 返回结果同时包含向量路与 BM25 路召回证据；RRF 融合与 rerank 排序逻辑有单测验证；结果按 rerank 分数降序且携带 source/heading/chunk_id/分数溯源；kb_type 过滤生效
- **Pass Condition**: 单测覆盖 RRF 计算、rerank 排序、元数据过滤；集成检索返回结构符合 schema 且 Top-1 与问题语义相关
- **Evidence**: pytest 输出、一次真实检索的完整返回 JSON

### AC-4: Agentic RAG 自评重检索生效
- **Type**: `rule`
- **Given**: 一个检索结果明显不足的问题（如题库中不存在的冷门主题）
- **When**: 运行检索-自评节点（LLM 使用 mock/录制响应）
- **Then**: 系统判定"不足"并触发查询改写与重新检索，且不超过配置的最大次数；判定为"充足"时不触发；决策过程写入日志/trace
- **Pass Condition**: mock 驱动的单测可分别稳定复现"重检索"与"不重检索"两条分支，并断言最大次数边界
- **Evidence**: pytest 用例与日志片段

### AC-5: 面试 Agent 完整流程与状态恢复
- **Type**: `rule`
- **Given**: 已选择简历/JD 与题库的新会话
- **When**: 候选人依次回答直至达到设定题数
- **Then**: 状态机依次经过开场、出题、回答、至少一次基于回答内容的动态追问、结束节点；每轮问题/回答与 RAG 使用的证据写入持久化存储；在任意时刻重启服务后可恢复会话继续面试
- **Pass Condition**: 端到端测试（LLM 可 mock）断言节点序列包含追问分支；模拟中断后 GET 会话状态与历史完整
- **Evidence**: pytest 端到端用例、持久化数据库记录、一次真实完整面试的事件流

### AC-6: 结构化评分报告
- **Type**: `rule`
- **Given**: 一场已完成的面试
- **When**: 请求报告
- **Then**: 返回包含全部约定维度的 1-5 分、每分关联具体问答证据、亮点/不足/改进建议非空、推荐知识点带来源；报告通过 Pydantic 校验；LLM 返回非法 JSON 时触发重试且最终成功
- **Pass Condition**: 报告 schema 校验通过；注入一次非法返回的 mock 测试验证重试逻辑
- **Evidence**: 真实报告 JSON、pytest 用例

### AC-7: 后端 API 与 SSE 流式契约
- **Type**: `rule`
- **Given**: 后端运行中
- **When**: 依次调用创建会话、发送回答（SSE）、结束会话、获取报告
- **Then**: SSE 按"阶段事件 + token 流 + 结束事件"格式返回；错误响应结构统一；请求/响应字段与前端及 Pydantic schema 完全一致
- **Pass Condition**: TestClient 集成测试覆盖全部端点与至少一个错误分支；流式事件可被逐事件解析
- **Evidence**: pytest 集成测试输出、SSE 原始事件样本

### AC-8: Streamlit 端到端体验
- **Type**: `rubric`
- **Dimension**: 前端完整度与可用性
- **Scale**: 1-5
- **Anchors**: 1 = 页面无法跑通或错误不可见；3 = 能完成面试但流式/报告/错误处理有明显短板；5 = 三页流程顺畅、流式自然、报告可视化清晰、异常提示友好、零控制台报错
- **Pass Threshold**: >= 4
- **Evidence**: 浏览器实操走查截图/录屏清单、Streamlit 控制台无异常堆栈

### AC-9: RAG 评测质量达标
- **Type**: `rubric`
- **Dimension**: RAG 回答质量（RAGAS 加权）
- **Scale**: 1-5（由指标映射：faithfulness ≥ 0.85、context recall ≥ 0.8、answer relevancy ≥ 0.85 记 5；任一低于 0.6 记 ≤ 2）
- **Anchors**: 1 = 三项指标大面积不达标；3 = 基本可用但有明显幻觉/漏召回；5 = 三项均达阈值且有基线报告
- **Pass Threshold**: >= 4（即三项指标全部 ≥ 0.75 且 faithfulness ≥ 0.85）
- **Evidence**: ≥40 条评测集、`run_eval` 生成的指标报告 JSON/Markdown 与基线记录

### AC-10: 可观测开关与降级
- **Type**: `rule`
- **Given**: 分别在"未配置 Langfuse key"与"配置有效 key"两种环境
- **When**: 完成一次完整面试
- **Then**: 无 key 时本地结构化日志完整记录检索/改写/节点/LLM 分段耗时；有 key 时同名 span/trace 出现在 Langfuse；两种模式业务行为一致
- **Pass Condition**: 两种配置下分别验证日志文件与（有 key 时）Langfuse trace 存在
- **Evidence**: 日志片段、Langfuse trace 截图

### AC-11: 云端部署公网可用
- **Type**: `rule`
- **Given**: 一台干净的 Linux 云服务器（Docker 已安装、端口已开放）
- **When**: 按部署文档执行初始化与部署脚本，注入服务器端 `.env`
- **Then**: app 与 qdrant 容器健康运行；通过 `http://服务器公网IP:端口` 可在浏览器完成一次完整模拟面试并看到报告；仓库与镜像中不含真实密钥
- **Pass Condition**: 公网地址端到端走查通过；`git ls-files` 与镜像内扫描无真实 key
- **Evidence**: 部署执行日志、公网访问走查记录、密钥扫描结果

### AC-12: 工程质量与面试可讲性
- **Type**: `rubric`
- **Dimension**: 代码工程质量
- **Scale**: 1-5
- **Anchors**: 1 = 脚本堆叠、无测试、跑不起来；3 = 结构尚可但测试/文档薄弱；5 = 分层清晰、核心模块测试齐备、README 架构与讲解要点完整、静态检查无错误
- **Pass Threshold**: >= 4
- **Evidence**: 目录结构、pytest 覆盖率报告（核心工具 ≥70%）、README、独立评审走查

## Open Questions

- [ ] 云服务器具体规格与开放端口（建议 2C4G 以上，端口暂定 8501/8000，部署前确认安全组）——不阻塞开发，部署阶段确认。
- [ ] 是否需要把 Qdrant、后端、Streamlit 全部容器化（当前默认：Qdrant + 后端 + Streamlit 三容器 compose）。
- [ ] 评测集语言以中文为主（默认中文简历/JD/题库），是否需要英文 JD 示例（默认附带 1 份英文 JD 以体现双语能力，非阻塞）。
