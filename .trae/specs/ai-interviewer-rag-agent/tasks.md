# AI 模拟面试官（RAG + Agent）- 实施计划

> 预计 10 个工作日。每个任务均含可自验的测试要求（TR）；`rule` 为可客观判定的二元条件，`rubric` 为评分项。依赖以任务 ID 表示。

## Task 1: 项目骨架、配置与健康检查
- **Status**: `completed`
- **Completion Evidence**:
  - `uv sync` 在 Python 3.11.9（`.python-version` 固定）全新虚拟环境零错误；核心依赖（fastapi/openai/langgraph/qdrant-client/rank-bm25/jieba/pypdf/streamlit/plotly）+ dev 组安装完成；ragas/langfuse 在可选组按需安装。
  - 配置：`app/config/settings.py`（pydantic-settings 集中加载、派生模型配置、路径统一解析到项目根、`config_status()` 不泄露密钥）；`.env.example` 覆盖全部变量；`.gitignore` 忽略 `.env` 且保留 `.env.example`/`.python-version` 入库。
  - 缺 key 校验：`validate_runtime()` 抛含变量名（DEEPSEEK_API_KEY/SILICONFLOW_API_KEY）与 .env 指引的中文 `ConfigurationError`（测试覆盖）。
  - `GET /health`：实测 HTTP 200，返回 config 与 components.qdrant（local 模式探测数据目录可写、server 模式探 /healthz）；未配 key 时整体 degraded 但进程存活；httpx 验证中文 UTF-8 正确（PS5 乱码仅为客户端显示问题）。
  - `pytest -q`：11 passed；`ruff check app tests`：All checks passed。
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 使用 uv 初始化 Python 3.11 项目（`pyproject.toml`），锁定依赖：fastapi、uvicorn、pydantic、pydantic-settings、langchain（含 openai/text-splitters 等必要子包）、langgraph、qdrant-client、rank-bm25、pypdf、streamlit、plotly、httpx、ragas、pytest、pytest-asyncio 等。
  - 建立分层目录：`app/{config,core,rag,agent,api,storage,observability,eval}`、`frontend/`、`scripts/`、`tests/`、`data/{knowledge,eval}`、`deploy/`。
  - 实现集中配置（`app/config.py`，pydantic-settings）：DeepSeek/SiliconFlow 的 api_key/base_url/model、Qdrant、检索参数、Langfuse 开关、上传限制；启动期校验缺失关键配置并给出中文可读报错。
  - 编写 `.env.example`（无真实 key）、`.gitignore`（含 `.env`、数据目录、`.venv` 等）、最小 FastAPI 应用与 `/health`（检查配置与 Qdrant 连通性）。
- **Acceptance Criteria Addressed**: AC-1, AC-12
- **Test Requirements**:
  - `rule` TR-1.1: 全新虚拟环境执行 `uv sync` 零错误；`uv run uvicorn` 启动后 `GET /health` 返回 200 且 JSON 含 qdrant 状态字段。
  - `rule` TR-1.2: 缺失 DEEPSEEK_API_KEY 时启动/配置加载产生明确报错信息（包含变量名），而非堆栈式 KeyError；`.env.example` 与 `.gitignore` 存在且 `.env` 不会被 git 跟踪。
  - `rule` TR-1.3: pytest 基线可运行（含一个配置加载单测），`pytest -q` 通过。

## Task 2: 示例数据与文档摄入管线
- **Status**: `completed`
- **Completion Evidence**:
  - TR-2.1 真实摄入（2026-10-16，真实 SiliconFlow embedding）：`ingest_kb.py --rebuild` → resume=13 / jd=19 / qa=41，集合总点数=73；不带 --rebuild 再跑一次 → 各类"替换旧点 N 个"、总点数仍为 73（幂等成立）。
  - TR-2.2/TR-2.3：chunker/loader/ingestion 共 17 个离线用例通过；抽样 chunk 含 chunk_id/source/kb_type/heading/text 五字段（test_qdrant_ingestion.py 覆盖）。
- **Progress Notes**（实现要点存档）:
  - 示例数据：中文简历（埋入 RAG/高并发项目细节）、中文 JD、英文 JD、八股题库 40 条（Python×10/后端×10/LLM×10/RAG-Agent×10）。
  - loader（PDF 用 pypdf 按页、MD 自研标题解析、TXT）、chunker（标题层级 + overlap 滑窗 + 句号边界回退 + 确定性 chunk_id）。
  - QdrantStore：local/server 双模式、uuid5 稳定 point id、delete_source 幂等、count/scroll/search/reset；规避了 Qdrant local 同名集合重建数据复活的坑（reset 改为按 point 删除）。
  - ingestion 服务 + `scripts/ingest_kb.py`（--rebuild、分类计数）。
  - 离线测试：chunker/loader（含动态生成的最小 PDF fixture）/local Qdrant 端到端幂等等 17 个用例通过。
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 编写内置示例数据：中文示例简历（Markdown）、模拟后端/AI 应用岗位 JD（中文 1 份 + 英文 1 份）、八股题库 30-50 条（Python/后端/LLM/RAG 方向，Markdown 结构化条目，每题含标题、答案要点、标签）。
  - 实现文档加载器（PDF 用 pypdf；Markdown/TXT 原生解析）与结构化切块器：保留标题层级（heading）、可配置 chunk_size/overlap、题库按条目切块；生成确定性 chunk_id（source + 序号哈希）。
  - 实现 Qdrant 访问层：按 kb_type 建/用集合（或单集合 + kb_type 字段），upsert 幂等（按 chunk_id 去重，重建前删除同 source 旧 chunk）、count、按元数据 scroll/delete。
  - 提供 `scripts/ingest_kb.py`：一键摄入 `data/knowledge`，支持 `--rebuild`，输出分类计数。
- **Acceptance Criteria Addressed**: AC-2, AC-12
- **Test Requirements**:
  - `rule` TR-2.1: 摄入脚本成功执行，按三类 kb_type 输出 chunk 计数且题库条目数与文件一致；连续执行两次 Qdrant 总点数不变（幂等）。
  - `rule` TR-2.2: 切块器单测：标题层级保留、overlap 生效、chunk_id 确定性（同输入两次一致）；PDF/Markdown 各用一个 fixture 文档通过解析测试。
  - `rule` TR-2.3: 抽样 scroll 出的 chunk 完整包含 chunk_id/source/kb_type/heading/text 五个字段。

## Task 3: DeepSeek / SiliconFlow 模型客户端
- **Status**: `completed`
- **Completion Evidence**:
  - TR-3.1 真实验活（2026-10-16）：`check_models.py` 三接口全部 HTTP 200——chat(deepseek-flash) 返回 'pong'；embedding(bge-m3) 维度=1024；rerank(bge-reranker-v2-m3) 降序 #0=0.9960/#1=0.0000。（注：首次验证曾因 SiliconFlow 余额不足 402，充值后通过；用户 .env 中对话模型配置为 deepseek-flash。）
  - TR-3.2/TR-3.3：mock 测试 10 个用例通过（401 不重试/429 重试/超时重试/流式拼接/embedding 保序/rerank 排序）。
- **Progress Notes**（实现要点存档）:
  - 统一异常体系（401/400/404 不重试，429/超时/5xx 指数退避 2 次，中文排查建议，不回显 key）；共享 http_utils。
  - DeepSeekChatClient：chat/stream_chat（SSE）、按用途绑定模型与温度、json_mode；SiliconFlowEmbedder（批处理保序、1024 维）；SiliconFlowReranker（/rerank 降序结果）。
  - `scripts/check_models.py` 三接口真实验活（表格/--json、缺失 key 友好退出码 2）。
  - mock 测试 10 个：成功、401 不重试、429 重试至成功/耗尽、超时重试、流式拼接、embedding 维度与顺序、rerank 排序与 401。
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 实现 `app/core/llm.py`：基于 OpenAI 兼容协议的 DeepSeek 聊天客户端（同步 + 异步、流式接口），按用途（出题/追问/评分/改写）提供不同默认参数的工厂方法，统一超时与最多 2 次指数退避重试。
  - 实现 `app/core/embedding.py`（SiliconFlow bge-m3，批量 embedding）与 `app/core/reranker.py`（bge-reranker-v2-m3，HTTP 调用 rerank 端点）。
  - 统一异常分类：401（key 无效）、404（模型不存在）、429（限流）、超时，转换为含中文说明与排查建议的自定义异常；错误信息不得泄露 key。
  - 提供 `scripts/check_models.py`：真实调用三个接口各一次，输出结构化可用性结论（模型名/接口类型/HTTP 状态/结论），供部署期验活。
- **Acceptance Criteria Addressed**: AC-1, AC-3
- **Test Requirements**:
  - `rule` TR-3.1: 配置真实 key 时 `check_models.py` 对 chat/embedding/rerank 三个接口均输出"可用"（以实际 HTTP 200 与返回体为证据，禁止臆测）。
  - `rule` TR-3.2: 使用 mock HTTP 层模拟 401/429/超时，断言抛出对应自定义异常且重试次数正确（429/超时重试、401 不重试）。
  - `rule` TR-3.3: reranker 对固定 query+文档列表返回降序分数与文档索引；embedding 批处理返回维度正确（1024）。

## Task 4: 混合检索与查询改写
- **Status**: `completed`
- **Completion Evidence**:
  - TR-4.1：RRF 手算对照单测通过（`test_rrf_hand_computed`：k=10 下 a/b/c/d 分数与手算 1/11+1/13 等逐项一致；`test_rrf_k_value_changes_scores` 验证 k 值对分数与平缓度的影响；同分确定性、空输入、非法 k 均覆盖）。
  - TR-4.2 真实集成（2026-10-16，真实 Qdrant 73 chunks + SiliconFlow + DeepSeek 改写）：`pytest -m integration` 通过。真实检索 "Redis 缓存穿透和缓存雪崩有什么区别？如何防范？"（kb_types=["qa"]）：Multi-Query 生成 3 条改写（共 4 路向量查询），Top-1 命中题库 Q14（缓存穿透/击穿/雪崩），`vector_rank=1 且 bm25_rank=1`（双路命中溯源成立），rerank_score 降序 0.9990 / 0.0092 / 0.0068；kb_types=["resume"] 过滤后结果全部 kb_type=resume。
  - TR-4.3：mock LLM（MockTransport + 真实 DeepSeekChatClient）验证开关——改写全关时 embedder 输入仅原查询且零 LLM 调用；Multi-Query 开时 embedding 输入 = 原查询+2 改写（3 路）；HyDE 开时追加假设答案向量一路；改写命中记入 matched_queries 溯源。
  - 基线：`pytest -q` 52 passed（新增 14 个用例）；`ruff check app tests` All checks passed。
- **Progress Notes**（实现要点存档）:
  - `app/rag/fusion.py`：RRF 纯函数（rank 从 1 计、同分按 ID 字典序保证确定性）。
  - `app/rag/bm25.py`：rank_bm25 + jieba 分词（英文/数字小写化），heading+text 参与打分，内存索引（≤2000 chunk 毫秒级构建，不持久化），0 分结果剔除，持有 chunk_id→Chunk 映射供 kb_type 过滤。
  - `app/rag/rewrite.py`：Multi-Query（json_mode + 行解析兜底、原查询居首去重截断 n+1）与 HyDE；任一策略 LLM 失败降级不中断检索；开关读 settings。
  - `app/rag/retriever.py`：`Retriever.retrieve(query, kb_types, top_k)` → `RetrievalResult(chunks, trace)`；RetrievedChunk 含 vector_score/vector_rank/bm25_rank/rrf_score/rerank_score/matched_queries；流程=改写→每改写查询一路向量召回+HyDE 追加一路→BM25 原查询一路→RRF(k=settings.retrieval_rrf_k)→rerank 融合前 max(4*top_k,20) 候选取 Top-K；BM25 索引懒构建 + `refresh_bm25()` 供摄入后刷新。
- **Priority**: high
- **Depends On**: Task 2, Task 3
- **Description**:
  - 实现查询改写器：Multi-Query（一次生成 N 个改写）与 HyDE（生成假设答案再以其向量检索），各自可通过配置开关；改写结果与策略记入检索溯源。
  - 实现双路召回：向量路（Qdrant，含 kb_type 过滤）+ BM25 路（索引随集合构建/持久化，rank_bm25，支持中文按字符或 jieba 分词——优先简单可靠方案并在文档注明）。
  - 实现 RRF 融合（k 参数可配），对多路结果融合去重；融合后调用 SiliconFlow reranker 精排取 Top-K。
  - 暴露统一 `Retriever.retrieve(query, kb_types, top_k)` 返回 `RetrievedChunk` 列表（含 vector_score/bm25_rank/rrf_score/rerank_score 与来源元数据）。
- **Acceptance Criteria Addressed**: AC-3, AC-12
- **Test Requirements**:
  - `rule` TR-4.1: RRF 纯函数单测：用手工构造的两路排名列表验证融合排序与 k 值影响，结果与手算一致。
  - `rule` TR-4.2: 集成测试（可打集成标记，真实 Qdrant + SiliconFlow）：给定岗位问题，返回 Top-K 按 rerank_score 降序，且溯源字段中能同时看到向量路与 BM25 路的命中文档；kb_type 过滤后结果不含其他类型 chunk。
  - `rule` TR-4.3: mock LLM 验证 Multi-Query/HyDE 开关分别生效/关闭时检索调用次数与查询内容符合预期。

## Task 5: Agentic RAG 检索质量自评与重检索
- **Status**: `completed`
- **Completion Evidence**:
  - TR-5.1：mock LLM（MockTransport 队列 no→yes）断言发生一次重检索且第二轮查询文本变为评估给出的改进查询（"Redis 缓存穿透 布隆过滤器"）；首轮 yes 时只检索一次、chat 恰好 1 次；评估未给改进查询时兜底为"原查询+理由"拼接，保证文本改变（`test_fallback_query_guarantees_text_change`）。
  - TR-5.2：连续 no 达上限（adaptive_max_rewrite_rounds=2，共 3 次检索）断言循环终止、`stopped_reason=max_rounds_reached`、返回各轮中 Top-1 rerank 分数最优轮（0.3/0.8/0.5 中取 0.8），caplog 含"达到最大轮次"。
  - TR-5.3：LLM 连续返回非法输出，重试一次后保守放行（视为充足、不重检索、/chat/completions 恰好 2 次、不抛未处理异常），caplog 含"保守放行"。
  - 解析单测：JSON yes/no、关键词兜底（是/否/yes/no）、`insufficient` 不被子串 "sufficient" 误判、`cannot` 类不被 "no" 误命中（词边界正则 + 开头窗口）、垃圾输出返回 None。
  - 真实联调（2026-10-16，真实 Qdrant 73 chunks + SiliconFlow + DeepSeek eval）：`pytest -m integration` 通过。真实查询 "Redis 缓存穿透和缓存雪崩有什么区别？如何防范？"（kb_types=["qa"]）：第 1 轮即 `sufficient=True`、top_score=0.9990，DeepSeek 理由"上下文[1]直接对比了缓存穿透、击穿、雪崩的区别并给出了各自的防范方案，足以回答该问题"，轮次日志结构化输出。
  - 基线：`pytest -q -m "not integration"` 58 passed；`ruff check app tests` All checks passed。
- **Progress Notes**（实现要点存档）:
  - `app/rag/adaptive.py`：`SufficiencyVerdict`（sufficient/reason/improved_query）；`_parse_verdict` JSON 优先 + 词边界关键词兜底；评估节点 `purpose="eval"` + json_mode，上下文取 Top-3 各 200 字摘要，非法输出重试一次后保守放行、调用异常亦保守放行（不中断检索主流程）。
  - `AdaptiveRetriever.retrieve(query, kb_types, top_k) -> AdaptiveResult(retrieval, rounds, stopped_reason, verdicts)`：复用 Task 4 `Retriever` 不改其内部；每轮判定/查询/top 分数写 logger.info；达上限返回各轮 Top-1 rerank 分数最优轮；每轮 verdict 留存供 trace。
- **Priority**: high
- **Depends On**: Task 4
- **Description**:
  - 实现检索评估节点：LLM 依据"问题 + 召回上下文"判定是否 sufficient（结构化 yes/no + 理由），带 JSON 输出约束与解析兜底。
  - 实现自适应检索循环：不足时基于反馈改写查询重新走混合检索，最多 `max_rewrite_rounds`（默认 2）次；充足或达上限即返回；每轮判定与命中变化写结构化日志/trace。
  - 封装为 `AdaptiveRetriever`，供出题/追问节点调用。
- **Acceptance Criteria Addressed**: AC-4
- **Test Requirements**:
  - `rule` TR-5.1: mock LLM 连续返回 no→yes，断言发生一次重检索且第二轮查询文本改变；返回 yes 时断言只检索一次。
  - `rule` TR-5.2: 连续 no 达上限时断言循环终止、返回当前最优结果且日志记录"达到最大轮次"。
  - `rule` TR-5.3: LLM 返回非法 JSON 时有兜底处理（默认重试一次后保守放行），不抛未处理异常。

## Task 6: 会话状态与持久化
- **Status**: `completed`
- **Completion Evidence**:
  - TR-6.1：临时 SQLite 文件（tmp_path）仓储测试 5 项通过——会话创建/读取全字段相等、状态更新、消息按插入序返回且证据引用（EvidenceRef 含 score）往返一致、state 快照（含中文/浮点/嵌套 dict）反序列化后与原对象相等且 upsert 覆盖生效、报告 InterviewReport 保存/读取/覆盖相等、仓储重复构造幂等。
  - TR-6.2：schema 校验 2 项通过——合法样例（含 SSE 四类事件 encode 为 `event: <type>\ndata: <json>\n\n` 线格式）全部接受；非法样例全部 ValidationError（score=0/6 越界、缺 score/track、空证据、空 track、题目数 0/11、highlights/weaknesses/suggestions 为空、done reason 非法值）。
  - 基线：`pytest -q -m "not integration"` 65 passed；`ruff check app tests` All checks passed。
- **Progress Notes**（实现要点存档）:
  - `app/api/schemas.py`（前后端唯一契约）：CreateSessionRequest（track/num_questions 1-10/candidate_name）、SessionRecord、MessageRecord（role ∈ interviewer/candidate/system、stage 与 Task 7 节点对齐的 InterviewStage、evidence 引用）、SseEvent（stage/token/error/done + encode()）、InterviewReport（REPORT_DIMENSIONS 五维度、DimensionScore 1-5 分且 evidence 非空、亮点/不足/建议非空、RecommendedTopic 带 source 回链）。
  - `app/storage/db.py`：SessionRepository（stdlib sqlite3，每次操作独立连接对多线程安全，CREATE TABLE IF NOT EXISTS）；sessions/messages/states/reports 四表；save_state 为 sort_keys JSON upsert；报告以 model_dump_json 存取。
  - 复用 settings.sqlite_path（./data/interview.db），未新增配置。
- **Priority**: medium
- **Depends On**: Task 1
- **Description**:
  - SQLite（标准库 sqlite3 或 SQLModel）存储会话：session、message（角色/内容/阶段/证据引用）、state 快照（LangGraph checkpoint 兼容的 JSON）、报告。
  - 设计 Pydantic schema（`app/api/schemas.py` 作为前后端唯一契约来源）：创建会话请求/响应、SSE 事件（stage/token/error/done）、报告 schema。
  - 提供仓储层接口：create_session / append_message / save_state / load_state / save_report / get_session。
- **Acceptance Criteria Addressed**: AC-5, AC-6, AC-7, AC-12
- **Test Requirements**:
  - `rule` TR-6.1: 仓储层单测（临时 SQLite 文件）：完整 CRUD 往返一致；state 快照序列化/反序列化后与原对象相等。
  - `rule` TR-6.2: schemas 对合法/非法样例分别通过/拒绝校验（如维度分越界、缺字段）。

## Task 7: LangGraph 面试状态机
- **Status**: `completed`
- **Completion Evidence**:
  - TR-7.1：mock LLM 队列（MockTransport）跑完整图 2 题含 1 次追问——`node_trace` 精确等于 [greeting, ask_question, receive_answer, assess_answer, followup, receive_answer, assess_answer, ask_question, receive_answer, assess_answer, final_evaluation, report]；追问不计题数（questions_asked=2）；消息按 7 条角色序落库、出题消息带 ev-1 证据引用、追问消息 stage=followup；报告落库且五维度齐全、会话状态 finished；评估汇总 quality_counts={good:2, partial:1}。追问上限用例：连续 partial/poor 时 followup 恰好出现 2 次后结束。出题 3 次重试全 500 → 降级证据模板题（含"考点1"、evidence_ids=["ev-1"]）；研判垃圾输出 → 默认 good 放行。
  - TR-7.2：start + 答 1 题后"崩溃"，InterviewEngine.restore(同一 session_id) 恢复——node_trace/已问题数/历史回答完整，继续答第 2 题正常结束、消息总数 5 条；未知 session 恢复返回 None。
  - 提前结束：finish() 走 final_evaluation→report 且幂等，结束后提交回答抛 RuntimeError。
  - 基线：`pytest -q -m "not integration"` 72 passed；`ruff check app tests` All checks passed。
  - TR-7.3（rubric 追问相关性）：评分 5/5（阈值 ≥4，达成）。2026-10-16 Task 9 真实全链路（test_real_full_chain_via_api，空泛作答触发）取 3 个真实追问样例人工评查：①"你简历里写从0到1负责RAG知识库落地，那请具体说一下文档切块策略和混合检索是怎么实现的？"——精确锁定回答回避点并引用简历证据；②"……如果切块和混合检索不是你做的，那你在该RAG项目中具体交付了什么？"——二次施压追问真实贡献边界；③"既然你提到加缓存，那在 AI 后端做缓存重建时，如果多个实例同时发现缓存失效并并发回源，你会用什么机制保证只有一个实例去重建……？"——承接候选人原话"加缓存"并结合分布式锁考点。三者均精确针对回答漏洞且引用简历/题库证据（anchor 5）。
- **Progress Notes**（实现要点存档）:
  - `app/agent/state.py`：InterviewState（TypedDict，纯 JSON 类型，直接作为 Task 6 save_state 快照）；initial_state 工厂。
  - `app/agent/graph.py`：START 路由器实现断点恢复（未问候→greeting / pending_answer→receive_answer / finish_requested→final_evaluation / done→END），不依赖 LangGraph 内置 checkpointer；拓扑 greeting→ask_question→END / receive_answer→assess_answer→{followup | ask_question | final_evaluation}→report→END。
  - 出题/追问证据约束：AdaptiveRetriever 召回 6 取 3（used_evidence_ids 去重，耗尽允许复用）；`_parse_question` 校验 evidence_ids ⊆ 召回集、空则兜底 Top-1；LLM/检索失败三级兜底（模板题→无证据项目题）。研判 `_parse_assess` 失败默认 good 放行，面试不中断。
  - 副作用约定：`_say()` 统一落消息 + 发 SseEvent（stage + 12 字分段 token）；report 节点 save_report + 会话置 finished + done 事件；每次图执行结束 save_state。
  - 报告生成器可注入（ReportGenerator 协议）；默认 default_report_generator 产出 schema 合法骨架（Task 8 以 LLM 结构化生成替换）。event_sink 预留（Task 9 接 SSE 队列）。
  - MAX_FOLLOWUPS=2 同题追问上限；followup 文本直接取研判输出的 followup_question（研判 prompt 含证据，追问 grounded）。
- **Priority**: high
- **Depends On**: Task 5, Task 6
- **Description**:
  - 定义面试状态（简历/JD 上下文、题库证据、历史问答、当前阶段、已问问题去重、每题追问计数、剩余题数、候选人画像）。
  - 实现节点与边：`greeting → ask_question（AdaptiveRetriever 取证据 + 结构化出题）→ receive_answer → assess_answer → conditional(追问/下一题/结束) → final_evaluation → report`；条件路由基于回答质量与轮次计数；同题追问 ≤ 2 轮、总题数可配。
  - 出题/追问 prompt 要求必须结合至少一条 RAG 证据（简历项目或题库），问题以结构化对象输出（题干、考察点、引用证据 id）。
  - 每步状态落库（checkpoint），支持传入 thread/session id 从中断点继续。
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `rule` TR-7.1: 用录制/mock 的 LLM 响应跑完整图，断言节点序列包含 greeting→ask→answer→assess→follow_up（至少一次）→final→report，且追问不超过 2 轮、题数达到配置即结束。
  - `rule` TR-7.2: 中途"中断"（仅落 checkpoint）后用同一 session_id 恢复，历史与剩余轮次正确。
  - `rubric` TR-7.3: 追问相关性；scale 1-5；anchors 1=模板化无视回答，3=部分承接，5=精确针对回答漏洞且引用简历/题库证据；threshold >= 4；evidence=3 个真实追问样例人工评查。

## Task 8: 多维评分与报告生成
- **Status**: `completed`
- **Completion Evidence**:
  - TR-8.2：`tests/test_report.py` 6 个 mock 用例通过——首次残缺 JSON 第二次合法 → 重试成功且第二次请求携带"未通过校验"修正反馈；维度缺失（4/5）→ 重试；分数越界（9）→ 重试；连续 3 次垃圾输出 → 抛 `ReportGenerationError("连续 3 次未能从 LLM 获得合法报告…")` 且 reports_dir 零文件（脏数据不落盘）；成功路径断言 prompt 携带问答原文与知识来源、Markdown 落盘含五维度/改进建议/推荐知识点。
  - TR-8.1 真实联调（2026-10-16，真实 DeepSeek + FakeAdaptive 固定证据，2 题完整面试含真实 greeting/出题/研判）：报告通过 schema 校验——五维度齐全、分数均在 1-5、每维度 evidence 非空、highlights/weaknesses/suggestions 非空；真实输出维度分 {岗位匹配度:1, 基础知识:1, 项目深度:1, 表达逻辑:1, 问题解决:1}（候选人兜底式回答，评分诚实）；推荐知识点 source 精确回链 `qa_backend.md`；Markdown 写入 `{session_id}.md` 含"岗位匹配度"。
  - 联调中发现并修复两个真实问题：①候选人表现弱时模型诚实返回 `highlights: []` 违反 min_length=1 且同 prompt 重试无效 → prompt 明确"禁止空数组"+ 重试时把非法输出与校验错误追加为上下文定向修正；②推荐知识点 source 曾整行复制 → prompt 收紧为"只填来源文件名"。
  - 基线：`pytest -q -m "not integration"` 78 passed；`ruff check app tests` All checks passed。
- **Progress Notes**（实现要点存档）:
  - `app/agent/report.py`：`LLMReportGenerator`（符合 graph.ReportGenerator 协议，`build_report_generator(settings, llm)` 装配 Markdown 落 `data/reports/`）；`_build_messages` 输入全部问答记录（含研判）+ 按 chunk_id 去重的知识来源 + evaluation_summary；`_parse_report` 注入 session_id/generated_at（不信任模型）+ Pydantic 校验 + 五维度精确匹配检查；`render_report_markdown` 渲染维度表格/评分依据/亮点/不足/建议/推荐知识点（表格单元转义 `|` 与换行）；失败重试 ≤2 次（共 3 次），LLM 调用异常（401 等）直接抛领域异常不占重试额度。
  - `app/agent/graph.py` 增量：出题/追问节点的 question dict 增加 `evidence` 快照（chunk_id/source/heading/score/text[:200]），供报告引用知识来源，state 保持纯 JSON 可快照。
  - `default_report_generator` 骨架保留为缺省，Task 9 装配时注入 LLM 生成器。
- **Priority**: high
- **Depends On**: Task 7
- **Description**:
  - 评分维度：岗位匹配度、基础知识、项目经验深度、表达与逻辑、问题解决能力（各 1-5 分 + 评语）；综合总分与结论。
  - 每个维度评分必须引用具体问答对/证据；输出亮点、不足、改进建议（可执行）、推荐复习知识点（回链题库 chunk 来源）。
  - 使用结构化输出（function calling / JSON mode + Pydantic 解析 + 失败重试 ≤2 次）；生成 Markdown 与结构化 JSON 两版报告并持久化。
- **Acceptance Criteria Addressed**: AC-6
- **Test Requirements**:
  - `rule` TR-8.1: 真实完成一场面试后报告通过 Report schema 校验：五维度齐全、分数在 1-5、每维度有证据引用、改进建议非空。
  - `rule` TR-8.2: mock LLM 首次返回残缺 JSON、第二次合法，断言重试后成功；连续非法时返回明确错误而非脏数据。

## Task 9: FastAPI 接口与 SSE 流式
- **Status**: `completed`
- **Completion Evidence**:
  - TR-9.1：`test_full_interview_chain_over_sse` 通过——multipart 创建会话（201，SessionRecord 字段与 schemas 一致）→ 空 content 触发开场（SSE 逐行解析：stage(opening) → token 流 → stage(question, round=1)，Q1 token 拼接完整）→ 两轮问答（末轮 SSE：stage(answer) → stage(scoring) → done(completed)）→ GET /sessions/{id} 状态 finished、消息角色序正确、出题消息带 ev-1 证据引用 → GET /report 通过 InterviewReport 校验（五维度齐全、分数 1-5、证据非空）且 Markdown 落盘。`test_finish_endpoint_early_and_idempotent` 覆盖提前结束（未作答出报告）与幂等返回。
  - TR-9.2：5 个错误分支用例通过——.exe 上传 400 INVALID_FILE_TYPE、超上限（测试置 8 字节）413 FILE_TOO_LARGE、缺 track 422 VALIDATION_ERROR、不存在会话四个端点（GET / GET report / finish / messages）全部 404 SESSION_NOT_FOUND、已创建未生成 404 REPORT_NOT_FOUND；响应体均为统一结构 `{"error": {code, message, details}}`。
  - TR-9.3：`test_upstream_401_degrades_then_chinese_error_event` 通过——SSE 首事件为 stage(opening)（首字节即阶段事件）；上游全 401 时 greeting/出题图内降级（兜底开场白 + 证据模板题"考点1"）正常下发，报告节点 AuthenticationError 转为 error 事件（code=UPSTREAM_AUTH_FAILED，message 含"API Key 无效或已过期（HTTP 401）…请检查 .env"中文排查建议）+ done(error)，脏报告不落库。
  - 真实全链路（2026-10-16 integration，真实 Qdrant 73 chunks + SiliconFlow + DeepSeek，`with TestClient` 持久循环）：创建 → SSE 开场 → 空泛作答连续触发真实追问 → done → 报告校验通过；两题均带真实证据来源（sample_resume_zh.md / ai_backend_jd_zh.md / ai_backend_jd_en.md / ai_backend_qa_bank.md）；自适应检索真实生效——首轮 top_score=0.6571 被判不足，改写 2 轮提升至 0.9506。
  - 联调发现并修复一个真实问题：TestClient 非上下文管理器模式每请求新建事件循环，跨请求复用的 AsyncOpenAI/httpx 连接池报 "Event loop is closed"（曾致研判默认放行、第二题证据丢失）；集成测试改用 `with TestClient(app)` 持久 portal（生产 uvicorn 单事件循环无此问题）。
  - 基线：`pytest -q -m "not integration"` 86 passed；`ruff check app tests` All checks passed。
- **Progress Notes**（实现要点存档）:
  - `app/api/deps.py`：ServiceContainer（settings/repo/store/embedder/reranker/retriever/adaptive/llm/report_generator）+ build_services 生产装配 + get_services 惰性装配（首请求缓存 app.state.services；测试直接赋值注入 mock 容器；健康检查不触发 Qdrant 文件锁）。
  - `app/api/sessions.py`：POST /sessions（multipart：track/num_questions/candidate_name/resume?/jd?；先校验后建会话避免孤儿；上传存 data/uploads/{session_id}/ 后经 ingest_loaded_document 摄入，source 含 session_id，摄入后 retriever.refresh_bm25()）；POST /sessions/{id}/messages（asyncio.Queue 桥接 event_sink → StreamingResponse：content 空且未开始 → start、否则 submit_answer、已开始空 content 安全无操作；异常 → error 事件 + done(error)；自记 first_event_ms）；POST finish（JSON，幂等）；GET 会话详情 / 报告。
  - `app/api/errors.py`：ApiError + 全局处理器（ApiError / HTTPException / RequestValidationError 422 / ModelProviderError 502 / ReportGenerationError 502 / Exception 500），统一 ErrorResponse 结构。
  - `app/agent/graph.py` 增量：`_say` token 事件仅 interviewer 发（候选人回显由前端本地渲染）；final_evaluation 节点发 stage(scoring) 事件（无消息体，供前端阶段徽标）。
  - `app/main.py`：挂 sessions 路由 + register_exception_handlers + 访问日志中间件（method/path/status/total_ms）；路由签名统一 Annotated 风格（规避 ruff B008）。
  - SSE 错误码约定：UPSTREAM_AUTH_FAILED / UPSTREAM_ERROR / REPORT_GENERATION_FAILED / INVALID_STATE / INTERNAL_ERROR。
- **Priority**: high
- **Depends On**: Task 7, Task 8
- **Description**:
  - 路由：`POST /sessions`（上传/选择简历+JD、岗位方向、题数）、`POST /sessions/{id}/messages`（SSE：阶段事件 + token 流 + done/error）、`POST /sessions/{id}/finish`（提前结束并出报告）、`GET /sessions/{id}`（状态与历史）、`GET /sessions/{id}/report`、`GET /health`。
  - 统一错误响应（code/message/details）与全局异常处理；上传校验（PDF/MD/TXT，≤10MB）；接入结构化日志中间件（分段计时：检索/首 token/总耗时）。
  - 上传文档自动摄入临时会话知识库（source 关联 session）。
- **Acceptance Criteria Addressed**: AC-7, AC-5, AC-3
- **Test Requirements**:
  - `rule` TR-9.1: TestClient 集成测试覆盖创建→两轮问答（SSE 事件可逐行解析出 stage/token/done）→结束→报告全链路；所有响应字段与 schemas 一致。
  - `rule` TR-9.2: 错误分支测试：未上传合法文档、发送超限文件、对不存在会话发消息，均返回统一错误结构与合理状态码。
  - `rule` TR-9.3: SSE 首字节包含阶段事件；模拟上游 401 时 error 事件携带可读中文说明。

## Task 10: Streamlit 三页式前端
- **Status**: `completed`
- **Completion Evidence**:
  - TR-10.1（字段逐一核对 checklist，代码评审）：前端读取的全部字段均来自既有契约，无自造/魔法字段——`/health`：status、version、config.{deepseek_configured,deepseek_model,siliconflow_configured,embedding_model,reranker_model,qdrant_mode,qdrant_url,langfuse_enabled}（与 `settings.config_status()` 逐键对齐）、components.qdrant.{status,mode,detail}；`POST /sessions`→SessionRecord：session_id/track/num_questions/candidate_name/status/created_at/updated_at；`GET /sessions/{id}`→SessionDetailResponse：session（同上）+ messages[].{role,content,stage,evidence[].{chunk_id,source,heading,score}}；SSE data：StageEventData.{stage,round}、TokenEventData.content、ErrorEventData.{code,message}、DoneEventData.reason；InterviewReport：session_id/dimensions[].{dimension,score,evidence,comment}/highlights/weaknesses/suggestions/recommended_topics[].{topic,source,reason}/generated_at。`tests/test_frontend.py` 的桩数据严格按上述契约构造，三页渲染均通过 → 字段名不符即断言失败。
  - TR-10.2（浏览器走查，示例数据 + 真实后端 + 真实 LLM）：开始页三指标实测 服务状态 healthy / DeepSeek 已配置 / SiliconFlow 已配置，导航、知识库来源、岗位方向、计划题目数、开始面试按钮齐全；创建会话（计划题目数=1）→ 面试页自动开场，问候语与第一题流式渲染，助手消息下方出现「证据来源：」chips，状态框标签显示「当前阶段：出题（第 1 题）」；提交约 142 字 RAG 相关回答后收到**真实追问**，共完成 3 轮问答（含 2 次追问）；报告页渲染 plotly 雷达图（五维度）、维度得分进度条、评分依据展开组、亮点/不足/改进建议清单、推荐复习知识点表格，「下载报告（Markdown）」按钮经浏览器确认存在于页底。控制台无 Streamlit 异常堆栈。
  - 错误提示场景：`tests/test_frontend.py::test_sse_error_event_shows_banner_with_code` 通过——SSE error 事件经 st.error 透出后端原文 message（含「API Key 无效或已过期」「401」）与「错误码：UPSTREAM_ERROR」，且页面不崩溃（无 exception）。后端 401→中文 error 事件由 TR-9.3 覆盖，前端渲染由此用例闭环。（说明：计划中的「浏览器内填写错误 key」实时走查因该浏览器环境卡死未能完成，故错误场景以确定性 AppTest 渲染测试 + 后端 SSE 测试为证据，未作臆测。）
  - 走查发现并修复 2 个真实问题：①「开始面试」抛 `StreamlitWidgetAlreadyInstantiatedError`——同轮内在导航组件（key="page"）实例化后改写其 session_state，改为「请求页（requested_page）→ 下一轮开头生效」模式；②`st.plotly_chart/dataframe` 使用 `use_container_width` 触发弃用告警（Streamlit 1.66），改用 `width="stretch"`，修复后服务端日志零告警。
  - AC-8 自评：4/5（阈值 ≥4 达标）。扣分项：浏览器视口的整页截图工具在该环境不可用（走查证据以元素树快照 + 断言为准），且实时错误场景未能用浏览器走查（已用确定性测试替代）。
  - 基线：`pytest -q -m "not integration"` 89 passed（新增 3 个前端用例）；`ruff check app tests frontend` All checks passed。
- **Progress Notes**（实现要点存档）:
  - `frontend/app.py`：入口与导航（`st.radio(key="page")`）；页面跳转统一走 `requested_page`，在主函数开头（组件实例化前）应用；侧边栏含 API 地址与可选密钥（session_state 内存保存、密码框、不落盘）。
  - `frontend/api_client.py`：``BackendClient`` 严格按 schemas 解析——multipart 创建、SSE 逐事件 yield（event/data 行解析）、统一错误解析 `{"error":{code,message,details}}` → ``ApiClientError``（透出中文原文），可选密钥经 `X-DeepSeek-Key`/`X-SiliconFlow-Key` 请求头传递。
  - 后端新增（支撑 FR-9 页面填 key）：`app/api/deps.py` 识别两个请求头，按密钥组合装配覆盖容器（复用基础容器的 QdrantStore/SessionRepository，避免重复打开文件锁），按密钥组合缓存于 `app.state.override_cache`；密钥仅存内存、不入日志。
  - `frontend/views_start.py`：健康指标 + 内置示例/上传简历·JD（pdf/md/txt，超限与类型错误由后端统一错误透出）+ 方向/题数/称呼 → 创建会话。
  - `frontend/views_interview.py`：`run_turn` 手动消费 SSE——st.status 随 stage 事件更新（开场/出题/追问/评分），token 累积渲染（带光标），error 事件写 session_state 供顶部横幅透出，done 事件更新状态；流结束 `st.rerun()` 以服务端历史重绘（证据 chips 完整）；首次进入自动开场（`auto_started` 防重跑）；`st.chat_input` 提交 + 提前结束按钮（页面级 spinner，防重复提交）。
  - `frontend/views_report.py`：plotly 雷达图（0-5 分径向轴）+ 维度分条与进度条 + 评分依据 expander + 亮点/不足/建议 + 推荐知识点表格（含来源列）+ 本地生成 Markdown 下载（字段同 Report，文件名 interview_report_{sid8}.md）。
  - `frontend/ui_common.py`：阶段文案映射（与 InterviewStage 对应）、消息渲染（角色气泡 + 阶段徽标 + 证据来源 chips）、错误横幅（`错误码：` 后缀）。
  - `tests/test_frontend.py`：`streamlit.testing.v1.AppTest` 真实执行 `frontend/app.py`，仅替换 `BackendClient` 类方法为桩；覆盖开始页渲染与开场、错误事件横幅、报告页图表/证据/表格/下载按钮。
  - **上线后视觉与交互升级（2026-10-17，部署到公网并在真实页面走查后重构）**：
    开始页由「运维三指标占据首屏」改为 Hero 优先——一句定位 + 报告 5 个评分维度预告，服务诊断与密钥配置收进折叠区；新增 `.streamlit/config.toml` 主题与全局 CSS（内容宽度收窄至 1080px、Hero 版式、控件样式），并在 Dockerfile 增补 `COPY .streamlit/` 保证容器内生效；岗位方向改中文标签 + 「其他（自定义）」文本补充；滑块/输入框/按钮文案去术语化（题目数量 / 你的称呼（可选）/「开始这场模拟面试 →」）。
    **会话可续接**：会话 ID 与页面写入 URL 查询参数（`?session=…&view=…`），刷新、换标签页或重开链接都能回到原面试（此前只存 session_state，刷新即丢整场会话）；并修复「首轮自动开场失败后卡在零消息空白页」——该状态下给出「重新生成第一题」出口。破坏性操作（提前结束 / 重新开始）改为二次确认弹窗；面试页新增进度条「已出题 N/总数」、面试结束后一键跳报告、失败可重试。
    验收（真实公网走查）：开始页 7 项新元素逐项确认存在（Hero 文案、5 个维度标签、默认折叠的「服务诊断」、来源两项新文案与句式统一、下拉 5 个中文选项、题目数量/你的称呼、按钮新文案）；主色为靛蓝（非 Streamlit 默认红）、内容宽度明显收窄居中；创建会话后地址栏出现 `?session=f67d5416…&view=interview`，**刷新后仍停留在面试页并保留第一题**；控制台无致命报错、无 ≥400 失败请求。
    回归：`pytest -q -m "not integration"` 128 passed（前端用例同步更新标签断言）；`ruff check frontend tests` All checks passed。
    排障记录：验证时一次浏览器检查报告了 3 个「旧标签」，与源码矛盾；改用容器内 `grep` 核对部署源文件（确为新版）并在**强制刷新后重新走查**，确认系浏览器缓存导致的误读，7 项实际全部生效。
    另注：`docker compose exec -T` 会吞掉通过 stdin 管道传入的后续脚本行（表现为多个检查「无输出」），逐条检查需加 `< /dev/null`。
  - **Hero 文案与配色二次打磨（2026-10-17）**：主标题由功能描述「让 AI 面试官基于你的简历与 JD 出题」改为结果导向「**把你的简历，变成一场模拟面试**」；副文案删除「约 10 分钟」（与可选题目数不匹配，属臆造）与「原文证据回链」等术语，改为「五维评分 / 亮点与改进建议 / 每处评价都能回溯到知识库原文」；徽标由技术栈（Agentic RAG · LangGraph）改为价值标签「简历驱动 · 评价可溯源」（技术栈移至文档）；维度 chips 补「报告维度」小标题以提供上下文。主色由通用 SaaS 靛蓝 `#4F46E5` 换为深青绿 `#0F766E`，同步 `.streamlit/config.toml`、`app.py` 的 `_STYLES` 与报告页雷达图（`line_color`/`fillcolor`，避免图表与主题撞色）。
    验收（强制刷新后真实浏览器实测）：标题、徽标、副文案、5 个维度标签逐字一致，旧标题已不存在；`getComputedStyle` 读取主按钮 `background-color` = **rgb(15, 118, 110)**（即 #0F766E，确认为青绿而非靛蓝/红）；chips 背景 rgb(243, 246, 246) 与文字 rgb(51, 71, 77) 与新 CSS 逐值一致（旧版为 #F3F4F6/#374151，可据此判定新样式表确已生效）。回归：128 passed、ruff All checks passed、部署后三容器 healthy。
- **Priority**: high
- **Depends On**: Task 9
- **Description**:
  - 开始页：API 连通状态（使用服务端配置或页面填写 key，仅本次会话保存于 session_state，不落盘）；上传简历/JD 或勾选内置示例；岗位方向与题目数；开始面试。
  - 面试页：对话式 UI，st.write_stream/手动消费 SSE 实现流式；展示阶段徽标（检索中/出题/追问/评分）与证据来源 chips；可提前结束。
  - 报告页：plotly 雷达图 + 维度分条 + 评语 + 证据问答展开 + 改进建议清单 + 推荐复习链接；提供报告下载（Markdown）。
  - 所有 API 错误通过 st.error/st.toast 透出原文 message；页面级 loading 与禁用防重复提交。
- **Acceptance Criteria Addressed**: AC-8
- **Test Requirements**:
  - `rule` TR-10.1: 前端 HTTP 客户端字段与 Task 6 schemas 逐一核对一致（代码评审 checklist），无硬编码魔法字段。
  - `rubric` TR-10.2: 浏览器走查完整流程（示例数据，真实后端），按 AC-8 anchors 评分 >= 4；覆盖一次错误提示场景（如错误 key）；evidence=走查截图清单与控制台日志。

## Task 11: RAGAS 评测集与回归脚本
- **Status**: `completed`
- **Completion Evidence**:
  - TR-11.1（一键运行 + 评测集 ≥40 且四类齐全）：`uv run python scripts/run_eval.py` 一键完成「校验评测集 → 43 条全量检索+受控生成 → RAGAS 评判 → 写 Markdown+JSON 报告 → 与基线对比 Δ」全链路，43/43 执行成功、0 执行错误。评测集 `data/eval/eval_dataset.json` 共 **43 条**：简历题 10、JD 匹配题 10（中文 6 + 英文 4，验证跨语言检索）、八股题 16（Python/后端/LLM/RAG-Agent 均匀抽自 Q01-Q40）、题库外诱导题 7（CEO 星座、量子物理、明日天气、刷好评脚本、简历薪资、JD 设备型号、背诗）；每条含 question / 原子化 ground_truth 要点 / expected_sources / expected_kb_types；加载器 `app/eval/dataset.py` 强制校验总数 ≥40、四类最低条数（8/8/16/6）、id 唯一、source 文件真实存在、oos 题不得期望命中来源（离线测试覆盖全部拒绝分支）。
  - TR-11.2（真实全量评测，2026-10-16 15:19，adaptive=True/top_k=6/评判=deepseek-chat/embedding=BAAI/bge-m3）：**faithfulness=0.985（阈值 .85）、context_recall=1.000（阈值 .80）、answer_relevancy=0.872（阈值 .85），三项全部达标**；context_precision=0.912（观察项，未设硬门槛）。分类别：resume 1.000/0.950/1.000/0.814，jd 1.000/0.820/1.000/0.894，qa 0.967/0.946/1.000/0.895。检索侧：期望来源命中率 **100%**（36/36），平均上下文 6.0 条、自适应平均 1.65 轮；诱导题拒答率 **7/7=100%**（目标 ≥80%），非诱导题误拒答率 **0%**；144 个 RAGAS 评判单元格零缺失。evidence 文件：`data/eval/reports/eval_20261016_151909.{json,md}` 与首版基线 `data/eval/baseline.json`（同内容定稿）。另用 `--limit 5` 实测基线对比：报告正确输出各项 Δ（如 faithfulness Δ=-0.025、answer_relevancy Δ=-0.053），门槛不过时进程退出码=2，可作回归信号。
  - 首版基线即达标，**无需回 Task 4/5 调参**；报告仍如实列出 2 个低分样例（resume-01：答案漏掉"计算机科学与技术"专业致 answer_relevancy=0.533、相关证据排第 2 位致 context_precision=0.5；jd-en-03：answer_relevancy=0.622），作为后续 prompt/精排优化的观察项。
  - AC-9 自评：5/5（锚点：三项门槛指标均超阈值且产出基线报告 → 记 5）。
  - 基线：`pytest -q -m "not integration"` **113 passed**（新增 `tests/test_eval.py` 24 例：评测集校验/拒答规则/受控生成/管线组装容错/报告聚合与基线 Δ），`ruff check app tests scripts` All checks passed。
- **Progress Notes**（实现要点存档）:
  - `app/eval/dataset.py`：pydantic 模型 `EvalQuestion/EvalDataset` + `load_dataset` + `validate_dataset`（错误聚合）；类别→kb_type 过滤、oos 期望拒答等派生属性。
  - `app/eval/generator.py`：严格接地的受控 RAG 问答 prompt（只依据资料、无依据必须说"知识库未提及/无法回答"、违规请求礼貌拒绝、空上下文给显式提示）——这是诱导题拒答与 faithfulness 评判的前提；生成走 `DeepSeekChatClient.chat(purpose="eval", temperature=0.1)`，不复用面试状态机的出题 prompt。
  - `app/eval/refusal.py`：规则法拒答判定（确定性、可单测）。两级判据：强直接短语（无法回答/不能协助/不在服务范围/违反平台规则…）前文 12 字内无第三人称或描述性标记（"模型拿到只言片语无法回答""让模型表达'知识库未提及'"不算拒答）；通用线索仅在 ≤160 字短答案生效。首轮全量跑发现 2 个误报（Q23/Q32 答案在正文中描述拒答策略），据此修正后非诱导题误拒答率归零，真实 7 条拒答全部仍命中。
  - `app/eval/runner.py`：`AdaptiveRetriever`（默认，生产路径）/`Retriever`（`--no-adaptive` 消融）双兼容的批跑管线，信号量并发（默认 4），chunk_id 去重组装 contexts、保留来源/轮数/rerank 分；单条异常捕获进 `error` 字段不中断整批。
  - `app/eval/metrics_ragas.py`：ragas 0.4.3 适配踩坑——①collections 指标只接受"现代"封装，LLM 用 `ragas.llms.llm_factory`（instructor，注入指向 DeepSeek 的 `openai.AsyncOpenAI`，function calling 可用），embedding 用 `ragas.embeddings.embedding_factory("openai", model=bge-m3, client=<SiliconFlow AsyncOpenAI>, interface="modern")`，旧 `LangchainLLMWrapper`/`aevaluate` 均被拒（v1/v2 体系不兼容）；②v2 指标直接并发调 `Faithfulness/ContextPrecision/ContextRecall/AnswerRelevancy.ascore(**kwargs)`，逐单元格容错为 None；③instructor 默认 max_tokens=1024 导致 2 个长答案 faithfulness 评判截断缺失，显式提到 4096 后 144/144 单元格齐全。依赖固定：`ragas>=0.4.3,<0.5` + `langchain-community==0.4.1`（0.4.2 移除了 ragas 仍导入的 `langchain_community.chat_models.vertexai`）。
  - `app/eval/report.py`：纯函数聚合（总体/分类别四项均值、来源命中率、拒答率、Δ）+ Markdown 渲染（门槛表/分类表/检索统计/诱导题明细/低分样例/失败样例），JSON 全量逐条落盘便于复盘。
  - `scripts/run_eval.py`：argparse（--dataset/--output-dir/--baseline/--top-k/--limit/--category/--no-adaptive/--concurrency/--workers/--save-baseline/--quiet）；报告输出 `data/eval/reports/eval_YYYYMMDD_HHMMSS.{json,md}`，存在 `data/eval/baseline.json` 时自动载入对比；退出码 0=门槛通过、2=未通过、1=环境/数据集错误。注意 Qdrant local 模式文件锁与运行中的 uvicorn 互斥，评测期间需停后端。
- **Priority**: medium
- **Depends On**: Task 8
- **Description**:
  - 在 `data/eval` 构建 ≥40 条评测集：问题、ground_truth 要点、期望命中的 source/kb_type；覆盖简历题、JD 匹配题、八股题、题库外诱导题（验证拒答/重检索）。
  - `scripts/run_eval.py`：批量跑检索 + 生成，调用 RAGAS（LLM/embedding 走 DeepSeek/SiliconFlow 适配器）计算 faithfulness、context_precision、context_recall、answer_relevancy，输出 Markdown + JSON 报告并与 `data/eval/baseline.json` 对比。
  - 记录首版基线；若指标不达 AC-9 阈值，回到 Task 4/5 调参（改写策略、K 值、rerank）并复评。
- **Acceptance Criteria Addressed**: AC-9
- **Test Requirements**:
  - `rule` TR-11.1: 评测脚本可一键运行并产出含四项指标的报告文件；评测集条数 ≥40 且四类覆盖齐全。
  - `rubric` TR-11.2: faithfulness ≥ 0.85、context_recall ≥ 0.8、answer_relevancy ≥ 0.85；按 AC-9 映射评分 >= 4；evidence=报告与基线文件。

## Task 12: 可观测性（Langfuse 可选 + 本地降级）
- **Status**: `completed`
- **Completion Evidence**:
  - TR-12.1（无 key 真实完整链路 → JSONL 关键 span 齐全）：以真实服务（真实 Qdrant + SiliconFlow + DeepSeek）经 HTTP 跑完一次面试（opening→追问→finish），仓库 `logs/traces.jsonl` 落盘证据：单会话 40 条记录、单 `trace_id`（=`session_id`）、`status=error` 计数 0，覆盖 node.greeting/ask_question/receive_answer/assess_answer/followup/final_evaluation/report、`llm.chat`（purpose/model/json_mode/prompt_tokens/completion_tokens/finish_reason/duration_ms）、`rewrite.multi_query`+`rewrite.hyde`、`retrieval.rerank`、`retrieval.summary`（vector_hit_count/bm25_hit_count/fused_count/rerank_candidate_count/returned/top_score）、`adaptive.verdict`+`adaptive.summary`。真实样本：`llm.chat` 1242ms/prompt_tokens=46/completion_tokens=19；`node.ask_question` 10050ms；`retrieval.summary` vector=26、bm25=20、fused=35、rerank 候选 24→返回 6、top_score=0.652；`adaptive.verdict` sufficient=false 并给出 improved_query。离线用例 `test_full_interview_writes_all_key_spans`（真实图 + 真实检索管线 + mock LLM）对同一批 span/层级/耗时字段做断言。
  - TR-12.2（有 key 上报、关闭后业务无差异）：
    - 业务无差异：`test_langfuse_enabled_keeps_business_behavior_identical` 分别以"关闭上报"与"开启上报（假 Langfuse 后端）"各跑一次完整面试，断言对话消息、会话状态、最终报告逐字一致，且开启时按 `session_id` 聚合出 `name=interview:{session_id}` 的 trace 并含图节点/检索 span。
    - 上报协议：`test_langfuse_sink_posts_trace_and_spans_to_ingestion_api` 用真实 langfuse SDK（2.60.10）上报到本地假 `POST /api/public/ingestion` 端点，断言 `trace-create`（id/name=interview:{sid}/sessionId）、`span-create`（名称、metadata=统一字段）、`span-update`（`level=ERROR`/`statusMessage`）按 Langfuse 协议送达——同名 trace/span 因而可在 UI 按 trace 名与 session 检索。
    - 降级：未安装 langfuse 或 key 缺失时仅写本地 JSONL（`test_build_tracer_degrades_when_langfuse_missing` / `test_build_tracer_without_keys_uses_jsonl_only`），观测 sink 异常仅告警不影响业务。
  - 质量基线：`pytest -q -m "not integration"` **122 passed**（Task 12 新增 `tests/test_observability.py` 9 例）；`ruff check app tests scripts` All checks passed；`pytest -m integration tests/test_api.py::test_real_full_chain_via_api` 1 passed（验证 SSE 与 finish 入口的埋点绑定未破坏既有链路）。
  - AC-10 按映射自评：**5/5**（无 key 结构化日志完整、mock 面覆盖 all key spans；有 key 同名 trace/span 协议级可达；两种模式业务逐字一致）。
- **Progress Notes**（实现要点存档）:
  - `app/observability/tracer.py`：`SpanRecord` 统一字段（name/trace_id/trace_name/session_id/span_id/parent_id/kind/started_at/duration_ms/status/error/fields）；`Span` 同时支持 `with`/`async with` 并提供 `set(**fields)` 补充运行期信息；contextvar 记录当前 span 以推断 `parent_id`（JSONL 可重建调用树）；`JsonlSink` 追加写且线程安全；`get_tracer()` 未设置时返回空实现（零副作用、不抛异常）；`traced` 装饰器对同步/异步函数通用；任一 sink 失败仅 warning，绝不中断业务（观测为旁路）。
  - `app/observability/langfuse_sink.py`：延迟导入 langfuse；`SpanRecord → trace().span().end(level/status_message)` 映射，metadata 携带 span_id/parent_id/kind/session_id/duration_ms 与业务字段；`close()` 执行 flush+shutdown 释放后台消费线程。
  - `build_tracer`：本地 JSONL 恒开；Langfuse 三件套齐备且依赖可用时并行上报；初始化失败自动降级（不阻断业务）。
  - 埋点位置（全部以上下文管理器/装饰器实现，不改函数签名与业务逻辑）：`llm.chat`/`llm.stream_chat`（purpose/model/json_mode/tokens/finish_reason/chunks/chars）；`retriever` 的 `rewrite.multi_query`、`rewrite.hyde`、`retrieval.rerank`、`retrieval.summary`(event)；`adaptive.verdict`/`adaptive.summary`(event)；图节点 `@traced`（node.greeting/ask_question/receive_answer/assess_answer/followup/final_evaluation/report）。
  - `app/api/sessions.py`：SSE 流与 `finish` 入口绑定 tracer（`trace_id=session_id`），`finally` 中 `reset_tracer` + `tracer.close()`；未配 key 时全程只写本地 JSONL。
  - `pyproject.toml`：`obs` 可选组固定 `langfuse>=2.50.0,<3`（`LangfuseSink` 基于 v2 的 `trace().span()` API，v3 改为 OTel 接口）。
  - `tests/conftest.py`：autouse 夹具把 `LOG_DIR` 重定向到用例临时目录，避免离线测试把观测 JSONL 写进仓库 `logs/`。
  - 注意：一次会话聚合为一个 trace（`trace_id=session_id`）；`logs/*.jsonl` 已在 `.gitignore` 忽略，不入库。
- **Priority**: medium
- **Depends On**: Task 9
- **Description**:
  - `app/observability/`：统一 tracer 接口；配置 Langfuse key 时上报 trace/span（session 维度：改写、双路召回计数、rerank、自评结论、图节点、LLM 调用与耗时）；未配置时写本地 JSONL 结构化日志（同字段结构）。
  - 在 RAG 管线与图节点埋点（不侵入业务逻辑，上下文管理器/装饰器实现）。
- **Acceptance Criteria Addressed**: AC-10
- **Test Requirements**:
  - `rule` TR-12.1: 无 key 运行完整测试链路后，JSONL 日志含一次面试的全部关键 span 字段。
  - `rule` TR-12.2: 配置有效 Langfuse key（集成测试/手工）时同名 trace 可在 Langfuse UI 查到；关闭后业务无任何行为差异（测试全绿）。

## Task 13: Docker 化与云服务器部署
- **Status**: `completed`（部署产物交付 + 离线校验 + **TR-13.1/TR-13.2 已在腾讯云服务器上真实跑通**，全项通过）
- **Completion Evidence**:
  - **部署产物**：`Dockerfile`（多阶段：builder 用 uv 0.13.0 执行 `uv sync --frozen --no-dev --no-install-project --extra obs`；runtime 复制 `.venv`，非 root 用户 `appuser` UID/GID 1000；一个镜像由 compose 以不同 command 分别启动 api 与 streamlit）；`docker-compose.yml`（qdrant `v1.19.2` + 命名卷 `qdrant_storage`、api、streamlit 三服务，均带 healthcheck，`depends_on: condition: service_healthy` 顺序 qdrant→api→streamlit，密钥仅经 `env_file: .env` 注入）；`.dockerignore`（排除 `.env*`、`.venv`、`__pycache__`、运行时产物 `data/db`、`logs` 等）；`deploy/deploy.sh`（含 `--help/--no-build/--skip-ingest`，检查 Docker→引导 `.env`→构建启动→轮询三容器 healthy→`check_models.py` 验活→`ingest_kb.py` 摄入→打印公网 `IP:8501`）；README 部署章节（规格/安全组/安装/一键部署/手动等价命令/验证达标/Windows 用法/更新日志备份）。
  - **离线等价校验（本机无 Docker，以结构断言 + 解析 + 静态检查替代运行时验证）**：compose 结构断言 12 项通过（三服务、共用镜像、`QDRANT_MODE=server`、内网 `API_BASE_URL=http://api:8000`、依赖顺序、三服务 healthcheck、命名卷、qdrant 无 `ports`、api 仅绑 `127.0.0.1:8000`、8501 发布、密钥仅 env_file）；Dockerfile 断言 3 项通过（多阶段、非 root、关键 flag）且全部 COPY 宿主源存在；`.dockerignore`/`.gitignore` 断言 5 项通过；`shellcheck -S style deploy/deploy.sh` 零告警。
  - **依赖同步命令有效性**：`uv sync --frozen --no-dev --no-install-project --extra obs --dry-run` 解析成功（"Would download 1 package / Would install 86 packages"，含 `langfuse==2.60.10`、`streamlit==1.66.0`，且项目自身未被安装），证明锁定文件与 obs extra 解析无误；此前一次完整下载复现因 **`pyarrow==26.0.0` 下载网络超时**（`client error (Connect)` / `operation timed out`）退出码 2，属环境网络问题而非命令问题。
  - **密钥安全**：仓库可见文件（按 ignore 规则排除 `.env`/`.venv` 等后）执行 `sk-[A-Za-z0-9]{16,}` 模式扫描零命中；`.env.example` 中原先误入的真实 Key 已改为 `sk-your-deepseek-key-here` / `sk-your-siliconflow-key-here` 占位符。
  - **TR-13.1 已实测（2026-10-17，腾讯云 Lighthouse 2C4G，Ubuntu 22.04，公网 42.194.242.186）**：`bash deploy/deploy.sh` 全流程通过 —— 镜像多阶段构建成功；`docker compose ps` 三容器全部 **healthy**（`ai-interviewer-qdrant-1`、`ai-interviewer-api-1`、`ai-interviewer-streamlit-1`）；容器内 `curl http://127.0.0.1:8000/health` → `{"status":"healthy",...,"qdrant":{"status":"ok","mode":"server"}}`（DeepSeek/SiliconFlow 均已配置、Langfuse 关闭态正常降级）；`check_models.py` 三接口 **chat/embedding/rerank 全部"可用"**（HTTP 200；embedding 维度 1024；rerank 降序 #0=0.9960/#1=0.0000）；`ingest_kb.py --rebuild` → resume=13 / jd=19 / qa=41，集合总点数 **73**（Qdrant server 模式）。
  - **TR-13.2 已实测（2026-10-17，浏览器走查公网入口）**：`http://42.194.242.186:8501` 从外部机器 `curl` 返回 **HTTP 200**（`/_stcore/health` 与 `/` 均为 200，证明防火墙 8501 已放行）；随后用浏览器对该公网地址完成**一次完整模拟面试**：填写候选人称呼 → 开始面试 → 首题 RAG 检索链路设计题流式输出完整 → 作答 → 面试官追问流式输出完整 → 再答一轮（多轮对话正常）→ 点击「提前结束并生成报告」→ 报告页展示岗位匹配度/基础知识/项目深度/表达逻辑/问题解决等维度评分与改进建议。**出题→作答→反馈→报告全链路在生产环境跑通。**
  - **构建期镜像源修复（本轮关键排障，已提交 `007dd74`）**：`uv.lock` 中每个包写死了官方 CDN 绝对直链（`https://files.pythonhosted.org/...`），`--frozen` 模式按该 URL 直连下载、**忽略 `UV_INDEX_URL`** → 国内反复卡在 `uv sync`（实测 `nsenter` 进构建命名空间 `ss` 看到 ESTAB 到 Fastly `151.101.0.223:443`，即 files.pythonhosted.org）。修复：Dockerfile 构建期 `sed -i "s#https://files.pythonhosted.org/#https://${UV_MIRROR_HOST}/#g" uv.lock`（默认 TUNA；镜像与官方文件字节一致、路径结构相同，sha256 校验不受影响）。修复后 `uv sync` **32s 完成**（此前 7+ 分钟无进展）。
  - **TR-13.3 已完整验证（2026-10-16，本机 `git init` 后实测）**：`git ls-files` 共 **100 个受跟踪文件，其中无 `.env`**（仅 `.env.example`）；`git check-ignore -v .env` 显示被 `.gitignore:2:.env` 规则忽略；对被跟踪文件逐文件执行 `sk-[A-Za-z0-9]{16,}` 模式扫描 → **0 命中**，无真实密钥入库。`.env` 为本地私有（含 2 个真实 Key，已确认被 git 忽略、不入库）。
- **Progress Notes**（实现要点存档）:
  - 命名卷首次创建时继承镜像内目录属主 → 非 root `appuser`(UID 1000) 无需启动期 `chown` 即可写卷。
  - `--no-install-project`：若把项目装进 site-packages，`settings.py` 的 `BASE_DIR` 会指向 site-packages，导致 `.env` 与 `data/` 相对路径解析错位，故项目以 `/app` 源码直接运行、不安装自身。
  - Qdrant 默认无鉴权 → 不发布 6333 端口；api 端口仅绑 `127.0.0.1:8000`（前端走容器内网访问），公网唯一入口 8501。
  - qdrant healthcheck 依据其镜像 entrypoint 为 `#!/usr/bin/env bash`（base `debian:12-slim`，经官方 Dockerfile 佐证）→ 用 bash 内建 `/dev/tcp` 探 6333，不依赖镜像内是否有 curl/wget；api/streamlit 用镜像自带 python 标准库探 HTTP 健康端点。
  - 镜像只装 `obs` extra、不含 eval(ragas) 依赖 → README 与 `deploy.sh` 中评测命令统一改为"本地开发环境 `uv sync --extra eval` 后运行"，避免给出必然失败的命令。
  - 国内服务器部署两大网络坑（已在 Dockerfile 内置规避）：① **PyPI**：直连 `files.pythonhosted.org` 仅约 69 KB/s，改走清华 TUNA 后 28 MB/s；但 `uv.lock` 内的**绝对直链**使 `--frozen` 绕过索引镜像，故必须同时改写直链主机（见 Completion Evidence）。② **Docker Hub**：`registry-1.docker.io` 直连不通，需在 `/etc/docker/daemon.json` 配 `registry-mirrors: ["https://mirror.ccs.tencentyun.com"]`（实测可拉取 `python:3.11-slim`、`qdrant/qdrant:v1.19.2`）。
  - 排障手法存档：用 `nsenter -t <pid> -n ss -tanp` 进构建命名空间观察网络，绕开"宿主机 `ss` 看不到构建容器连接"的盲区；`pgrep -af 'docker compo[s]e build'` 用字符类避免 `pkill -f` 匹配到自身 ssh 命令行。
- **Priority**: high
- **Depends On**: Task 10, Task 12
- **Description**:
  - 多阶段 `Dockerfile`（uv 安装依赖、非 root 运行，后端与 Streamlit 同镜像不同启动命令）；`docker-compose.yml`：qdrant（带命名卷）、api（8000）、streamlit（8501），仅 `.env` 注入密钥，健康检查与依赖顺序（depends_on + healthcheck）。
  - `deploy/deploy.sh`（Linux）：检查 Docker、拉取/构建、初始化 `.env` 引导提示、启动、运行 `check_models.py` 验活、打印公网访问地址；附 Windows 本地 `docker compose` 使用说明。
  - README 部署章节：服务器规格/安全组端口/Docker 安装/部署/更新/日志查看/数据备份（Qdrant 卷 + SQLite 卷）。
- **Acceptance Criteria Addressed**: AC-11, AC-1
- **Test Requirements**:
  - `rule` TR-13.1: 本地 `docker compose up -d` 三容器全部 healthy；容器内 `/health` 200，Streamlit 页面可打开。
  - `rule` TR-13.2: 在用户云服务器执行部署脚本后，公网 `IP:8501` 可完成一次完整模拟面试（走查证据）。
  - `rule` TR-13.3: `git ls-files` 不包含 `.env`；对镜像与仓库执行 key 模式扫描（sk- 等）无真实密钥命中。

## Task 14: 测试收口、README 与面试讲解材料
- **Status**: `completed`（文档与测试收口全部完成；TR-14.1 全项通过，TR-14.2 的"独立评审 ≥4"为人工评分项，待评审）
- **Completion Evidence**:
  - **TR-14.1 覆盖率**：`uv run pytest -q -m "not integration" --cov=app --cov-report=term-missing` → **128 passed, 4 deselected**（集成用例经 `-m integration` 分离）；总覆盖率 **89%**（branch 模式）。核心工具模块：`rag/chunker.py` 95%、`rag/fusion.py`(RRF) 100%、`agent/state.py` 100%、`api/schemas.py` 100%、`storage/db.py` 100% + `storage/qdrant_store.py` 81%、`rag/adaptive.py` 93%，另 `core/reranker.py` 补测后 100% —— 全部 ≥70%。
  - **TR-14.1 静态检查**：`uv run ruff check app tests frontend` → **All checks passed**。
  - **本轮新增测试**：`tests/test_model_clients.py` 新增 reranker 状态码映射参数化用例（404/400/429/500/302 → 统一领域异常）与空文档短路用例，共 +6 例，将 `core/reranker.py` 覆盖率由 67% 提升至 100%。
  - **覆盖率配置**：`pyproject.toml` 新增 `[tool.coverage.run]`（`source=["app"]`、`branch=true`）与 `[tool.coverage.report]`（排除 `TYPE_CHECKING`/`pragma: no cover` 等），使 `--cov=app` 可复现。
  - **README 重写**：补齐全部分区 —— 架构图（mermaid 组件与数据流 + LangGraph 状态机节点/条件边）、技术栈与选型表、快速开始（Windows 本地，默认 `QDRANT_MODE=local` 无需 Docker）、目录结构、接口表（6 个端点 + SSE 事件类型）、评测方法与当前基线指标、关键设计取舍（混合检索/RRF、两阶段 Rerank、Agentic RAG、LangGraph vs 裸 ReAct、Qdrant 双模式、可观测双后端）、测试与代码质量（含一键命令）、可观测性；并保留原部署章节（服务器规格/安全组/一键部署/验证达标/备份）。
  - **TR-14.2 待评审**：README 步骤可被无背景者照做一项，已按"复制 .env.example → 填 key → check_models → ingest_kb → 启动后端与前端"给出单一路径；"由独立评审打分 ≥4"需第三方评审，未自评。
  - **`docs/interview_talk.md`**：按「项目背景 → 核心难点 → 方案（数据/检索/自适应/编排/服务/可观测/评测分层）→ 量化效果 → 复盘」组织，并含 **10 个高频追问**应答要点（RRF vs 加权、改写噪声、重检索防死循环、中文分词、LangGraph 选型、RAGAS 坑、faithfulness 高而 relevancy 低、拒答与误拒答、状态持久化、性能与成本）。
- **Progress Notes**（实现要点存档）:
  - 覆盖率已在本任务前即达标，故**未为凑数而加测试**；仅对唯一低于 70% 的核心模块（reranker 67%）做针对性补测。
  - `eval/metrics_ragas.py` 覆盖率 18%：该模块为 RAGAS 集成路径（需 `--extra eval` + 真实评委模型），由集成评测覆盖，非核心工具模块，故未纳入离线单测目标。
  - README 内容存在重叠维护点：Task 13 部署章节被保留为 §10，新章节插在其前，避免重复书写。
- **Priority**: medium
- **Depends On**: Task 11, Task 13
- **Description**:
  - 补齐单元测试使核心工具模块（chunker、rrf、state 路由、schemas、storage、adaptive 循环）覆盖率 ≥70%；分离集成测试标记（`-m integration`）；CI 可选（至少提供 pytest 一键命令）。
  - README：架构图（组件与数据流）、快速开始（Windows 本地）、部署上线、接口表、评测方法与当前指标、关键设计取舍（为何混合检索、为何 Agentic RAG、为何 LangGraph 而非裸 ReAct）。
  - 整理 `docs/interview_talk.md`：按"项目背景→难点→方案→量化效果→复盘"的面试自述稿与 10 个高频追问应答要点。
- **Acceptance Criteria Addressed**: AC-12, AC-9
- **Test Requirements**:
  - `rule` TR-14.1: `pytest -q` 全绿；覆盖率报告显示核心工具模块 ≥70%；`ruff`/基础静态检查无错误。
  - `rubric` TR-14.2: 按 AC-12 anchors 由独立评审打分 >= 4；README 步骤可被无背景者照做成功。

## Task 15: 检索消融对照与真实会话成本/延迟实测（作品集增强 P0）
- **Status**: `completed`（两组实验均已真实跑通，数字全部来自实际运行；文档与证据已回写）
- **Completion Evidence**:
  - **检索层可切换消融档位**：`retriever.py` 新增 `RetrievalStages(rewrite/bm25/rrf/rerank)`（默认全开＝生产行为，零回归）；`fusion.py` 新增
    `concat_fuse` 朴素拼接融合基线用于对照。离线单测 **127 passed**、`ruff check app scripts` **All checks passed**。
  - **TR-15.1 消融对照已实测（2026-10-17，同一 43 条评测集，5 档）**：报告 `data/eval/reports/ablation_20261017_223939.{md,json}`。
    `context_precision`：dense **0.819** → +bm25 **0.824** → **+RRF 0.925** → +rerank **0.935** → +agentic **0.937**；
    每问耗时：0.31 / 0.36 / 0.45 / 0.38 / **2.28** s；总 token：41.6k / 41.7k / 42.9k / 44.7k / **87.6k**。
    **结论**：本场景精度收益主要来自 **RRF 融合排序**（+0.101，相对 +12%），远大于单加 BM25（+0.005）与 rerank（+0.010）；
    **agentic 档边际收益为负**（+0.002 却 6× 耗时、2.2× 成本、`context_recall` 1.000→0.993），其定位是**鲁棒性冗余**而非精度提升。
  - **TR-15.2 真实会话成本与延迟已实测（2026-10-17，6 场 × 5 题，真实模型）**：报告 `data/eval/reports/session_cost_20261017_230955.{md,json}`。
    单场 token **12,760–16,545**；单场成本 **均值 ¥0.052**（中位 ¥0.051、区间 ¥0.046–¥0.060）；单场端到端耗时 **均值 48.7s**（P50 48.9s / 最大 51.3s）；
    `ask_question` 均值 **10.7s**（P95 11.8s，首字延迟大头）、`report` 7.1s、`assess_answer` 1.4s；单次 LLM 调用 1.42s（P95 1.84s）、单次 rerank 0.46s。
    token 按用途：eval 96 次 / 64.9k、question 24 次 / 10.6k、rewrite 54 次 / 10.3k。
  - **文档回写**：README 新增 §6.1（消融对照 + 局限性说明）与 §6.2（会话成本/延迟），并在 §7 增加「为何实测 agentic 边际收益为负仍保留」的取​舍条目；
    `docs/interview_talk.md` 新增 §5.1/§5.2 量化小节，§4.7 补充两个实验脚本入口；§6 复盘扩写为 4 篇故障复盘。
  - **发现并修正的自身错误**：消融脚本表格列头初版把该列（`oos_refusal_rate`）标成「误拒答率」，与项目既有的「诱导题拒答率」口径相反，
    已修正脚本渲染逻辑并同步订正已生成报告，避免作品集自相矛盾。
- **Progress Notes**（实现要点存档）:
  - 消融实验**不新增任何生产功能**，只把已上线管线拆成可切换阶段，因此风险与工作量都极低、而证据价值高。
  - 实验运行期间**独占 Qdrant 文件锁**：两脚本均在本机以 local 模式直连 `data/qdrant_storage`（需先停后端/容器），符合项目既有约束。
  - 会话实测的成本口径写死在脚本常量（deepseek-chat ¥2/¥8 每百万 token，缓存未命中），并在报告头部声明；token 全部取自项目自身埋点的
    `llm.chat` 字段，**不依赖任何模型自报或外部账单**，因此可复现。
  - 会话实测的答题内容为**固定通用回答脚本**（非真实候选人），故报告中「均分」不代表产品评分水平——已在报告与脚本渲染中显式加注说明。
  - 消融档位独立重新生成答案（`temperature=0.1`），故 `faithfulness`/`answer_relevancy` 的档间微差含生成随机性，README 已标注「不宜过度解读」。
- **Priority**: high
- **Depends On**: Task 11, Task 12, Task 14
- **Description**:
  - 把检索管线拆成可切换阶段（dense / +bm25 / +rrf / +rerank / +agentic），用同一评测集跑出「质量–延迟–成本」对照表，量化每个组件的边际贡献。
  - 用真实模型跑真实多轮面试，从自身埋点聚合单场 token、成本、各阶段耗时（P50/P95），给出可复现的成本与延迟基线。
  - 结论回写 README 与面试讲解材料，并诚实标注实验局限（评测集饱和、生成随机性、答题脚本非真人）。
- **Acceptance Criteria Addressed**: AC-9, AC-12
- **Test Requirements**:
  - `rule` TR-15.1: 消融脚本可在同一评测集上跑出全部档位对照，且各档指标/延迟/token 有据可查（报告文件可复核）。
  - `rule` TR-15.2: 会话实测脚本可完成多场完整面试并输出单场成本、端到端耗时与节点级耗时，数字可由埋点文件逐条复算。

## Task 16: Judge 自评偏置治理——可切换独立 judge + 异源 judge 交叉验证（作品集增强 P1）
- **Status**: `completed`（配置、脚本、交叉验证实验、文档与证据全部完成；数字全部来自实际运行）
- **Completion Evidence**:
  - **可切换独立 judge 配置**：`settings.py` 新增 `EVAL_JUDGE_MODEL / EVAL_JUDGE_BASE_URL / EVAL_JUDGE_API_KEY` 与派生属性
    `judge_model / judge_base_url / judge_api_key`（**全部留空则依次回落** `LLM_MODEL_EVAL`→`deepseek_model`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_API_KEY`，
    即默认与既有行为**逐字一致、生产零变化**）。`app/eval/metrics_ragas.py` 新增 `JudgeSpec` + `resolve_judge(settings)`，
    `build_judges`/`run_ragas_metrics` 支持注入 `judge`（仅离线评测路径，不触碰线上评分链路）。`.env.example` 同步补三项说明。
  - **异源 judge 可用性实测（2026-10-18）**：探测 SiliconFlow 模型清单（45 个 Qwen 系）；
    对候选逐一实测 **function calling + 吞吐**：`Qwen/Qwen2.5-72B-Instruct` tool_call=True 1.7s、5/5 串行请求平均 2.6s；
    `Qwen3-30B-A3B-Instruct-2507` tool_call=True 4.2s；`Qwen2.5-32B-Instruct` tool_call=True 1.9s；
    **`Qwen2.5-7B-Instruct` tool_call=False**（不能作 ragas instructor judge）。最终选 `Qwen/Qwen2.5-72B-Instruct`。
  - **TR-16.1 交叉验证已实测（2026-10-18，同一 36 条生成结果 / 144 个指标单元，0 缺失）**：
    报告 `data/eval/reports/judge_cross_check_20261018_014148.{md,json}`；脚本 `scripts/run_judge_cross_check.py`。
    自评 `deepseek-chat` vs 异源 `Qwen/Qwen2.5-72B-Instruct`（Δ=自评−异源，>0=自评偏高）：

    | 指标 | 自评均值 | 异源均值 | Δ | 平均绝对差 | ≤0.10 一致 | Pearson |
    | --- | --- | --- | --- | --- | --- | --- |
    | faithfulness | 0.973 | 1.000 | -0.027 | 0.027 | 88.9% | N/A（异源无方差） |
    | context_precision | 0.928 | 0.914 | +0.015 | 0.024 | 91.7% | 0.866 |
    | context_recall | 1.000 | 1.000 | 0.000 | 0.000 | 100.0% | N/A（双方恒定） |
    | answer_relevancy | 0.899 | 0.790 | **+0.108** | 0.112 | **55.6%** | 0.638 |

    全体 144 格：|Δ|≤0.05 **76.4%**、≤0.10 **84.0%**、≤0.20 **93.8%**。
    **结论（如实，不美化）**：`faithfulness`/`context_precision`/`context_recall` 对换 judge 稳健 → **未见自评抬高**；
    仅 `answer_relevancy` 自评显著偏高（+0.108、一致率 55.6%、Pearson 0.638），该项绝对值**降级为趋势信号**、不作硬门槛。
  - **限流治理**：跨厂商 judge 触发 SiliconFlow `429 TPM limit reached`；`run_ragas_metrics` 新增**仅对限流失败**的指数退避重试
    （`max_attempts`/`retry_backoff`，默认 1/不重试＝不改既有行为），脚本对异源 judge **默认串行**（`--cross-workers=1`）。
    串行后重跑：**0 次 429、0 个单元格丢失、144/144 全满**。
  - **文档回写**：README 新增 §6.3（交叉验证表 + 结论 + 配套改动）与 §7 新增「为何做异源 judge 交叉验证」取舍条目；
    `docs/interview_talk.md` 新增 §4.7 脚本入口、§5.3 量化小节、§5 基线补 `answer_relevancy` 克制解读注、
    §6.1 新增第 ⑤ 篇故障复盘（TPM 限流「越重试越糟」）、Q6/Q7 补充「自评偏置」应答口径。
- **Progress Notes**（实现要点存档）:
  - 交叉验证**不重新生成答案**：同一批 `RowOutcome` 交两个 judge 打分，故所有 Δ **只归因于 judge 本身**，方法上最干净。
  - `answer_relevancy` 自评偏高**不作删除处理**：它是真实存在且可解释的现象（依赖「人造问题 + embedding 相似度」，对 judge 切题判准敏感），
    正确做法是**披露 + 限定用法**，而非隐藏或换掉指标。
  - Pearson 在 `faithfulness`/`context_recall` 上为 N/A 属**预期**：异源 judge 在 36 条上给出**恒定值**（1.000）→ 方差为 0，相关系数无定义；
    `compare_metric` 已对「样本不足 / 方差为 0」返回 None 而非抛错或报 NaN。
  - 串行 72B 判分耗时较长（`context_precision`/`context_recall` 需按 context/reference 多次 LLM 调用），
    故脚本默认单并发；这是**限流下的正确取舍**（加并发会 429、重试还会重发整段 prompt 放大配额消耗）。
- **Priority**: high
- **Depends On**: Task 11, Task 15
- **Description**:
  - 新增可切换的独立 judge 配置（默认回落 `eval_model`，生产零变化），使 RAGAS 评判模型可与线上评分模型解耦。
  - 引入**另一厂商、另一模型族**的 judge，对**同一批生成结果**再打一遍分，量化 per-metric 均值差 Δ 与逐条一致率，
    把「自评偏置」从口头辩解变成**可复核的数字**。
  - 结论与局限回写 README 与面试讲解材料；对受 judge 影响大的指标（`answer_relevancy`）明确限定解读方式。
- **Acceptance Criteria Addressed**: AC-9, AC-12
- **Test Requirements**:
  - `rule` TR-16.1: 同一批生成结果可被两套 judge 分别打分，输出 per-metric Δ、平均/最大绝对差、多档一致率与 Pearson（报告可复核）。
  - `rule` TR-16.2: `EVAL_JUDGE_*` 全部留空时评测行为与改动前一致（离线单测覆盖默认回落链路）。

## Task 17: 引用质量度量升级——来源命中率 → P/R/F1（作品集增强 P1）
- **Status**: `completed`（指标实现、真实复跑、文档与证据完成；数字全部来自实际运行）
- **Completion Evidence**:
  - **指标实现**：`RowOutcome` 新增 `source_precision` / `source_recall` 与派生属性 `source_f1`（`app/eval/runner.py`）；
    在计算 `source_hit` 的同处按来源粒度算出 P/R（precision = 召回中期望来源占比；recall = 期望来源被覆盖占比）。
    `app/eval/report.py` 的检索侧统计新增 overall + by_category 的 P/R/F1，逐条 `rows[]` 亦输出三字段，
    Markdown 报告新增 P/R/F1 行与「| 类别 | n | 来源命中率 | P | R | F1 | 平均上下文数 |」表。
  - **新增诊断项**：`answer_citation_rate`（答案显式引用标记率，正则 `_CITATION_RE` = `【资料N】`/`[N]`/`根据资料N`/`来源:`），
    用于判断「引用质量可否在答案侧度量」。
  - **真实复跑（2026-10-18，43 条）**：报告 `data/eval/reports/eval_20261018_021445.{md,json}`。
    TR-11.2 门槛**全部通过**：faithfulness 0.989、context_recall 0.993、answer_relevancy 0.895、
    context_precision 0.940（观察项）；对比基线 Δ = +0.004 / -0.007 / +0.023 / +0.028。
    检索侧：来源命中率 100.0%（二值，饱和）、**来源精确率 P 0.861**、**来源召回率 R 1.000**、**F1 0.907**、
    **答案显式引用标记率 22.2%（8/36）**。
  - **分类别**：`resume` P=1.000、`qa` P=1.000、`jd` P=**0.500**（R 均为 1.000）。
    JD 类每问同时召回 `ai_backend_jd_zh.md` + `ai_backend_jd_en.md`，而期望集只列中文 JD。
  - **结论（如实，含局限）**：二值命中率恒 100% 无区分度的问题被 P/R/F1 **缓解**（能暴露 JD 类 P=0.5）；
    但 **JD 的 P=0.5 是「单期望来源」口径偏严**，把跨文件（同 JD 英文版）的合法召回判为非期望，
    属**新指标的副作用**而非检索错误。改进方向（已明确、未实施）：`expected_sources` 支持同题多文件或改用品级相关度。
    答案仅 22.2% 自带引用标记 → 引用质量**只能在检索侧度量**，故不命名为「citation accuracy」（避免过度声称）。
  - **文档回写**：README 新增 §6.4（含局限披露），§6.1 局限处补交叉引用；`docs/interview_talk.md`
    §5 表更新为最新复跑值并保留首轮值对照、§5.1 补 §5.4 交叉引用、新增 §5.4、Q7 数值同步、新增 Q11/Q12。
- **Progress Notes**（实现要点存档）:
  - 升级前先用**既有 2026-10-16 报告逐条复算**做过口径验证（mean P=0.861 / R=1.000），确认新指标有区分度后再落代码，
    避免"先写指标、后编解释"。
  - `source_f1` 是 `RowOutcome` 的 property 而非存储字段，避免同一事实两处维护；聚合时对每行算 F1 再取均值。
  - P/R/F1 只对 `expected_sources` 非空的非诱导题计算；oos 行为 None，`mean()` 自动忽略，不进分母。
- **Priority**: medium
- **Depends On**: Task 11, Task 15
- **Description**:
  - 把饱和的二值「来源命中率」升级为来源粒度 precision/recall/F1，使检索侧指标具备区分度。
  - 增加「答案显式引用标记率」诊断，据此明确引用质量的度量口径（检索侧 vs 答案侧）。
  - 如实披露单期望来源口径带来的偏差，并给出改进方向。
- **Acceptance Criteria Addressed**: AC-9, AC-12
- **Test Requirements**:
  - `rule` TR-17.1: `RowOutcome` 的 P/R/F1 计算与边界（P+R=0、缺字段）正确；报告聚合与 Markdown 渲染包含新字段。
  - `rule` TR-17.2: 二值命中率恒为 1.0 时，P/F1 仍能区分「召回中混入非期望来源」的样例。

## Task 18: 评测回归门禁——阈值跌出即非零退出 + 接入 CI（作品集增强 P1）
- **Status**: `completed`（实现、真实运行取证、文档与 CI 工作流完成）
- **Completion Evidence**:
  - **门禁逻辑（可离线单测）**：新建 `app/eval/gate.py`——`latest_report(reports_dir)` 取 `eval_*.json` 最新一份；
    `check_report(report, *, baseline=None, max_drop=0.05)` 返回违规列表，5 类 `kind`：
    `missing_metric` / `below_threshold` / `oos_refusal` / `exec_errors` / `baseline_drop`。
    阈值来源复用 `app/eval/metrics_ragas.THRESHOLDS` 与 `app/eval/report.GATED_METRICS / OOS_REFUSAL_TARGET`，不新造一套常量。
  - **可执行入口**：新建 `scripts/check_eval_gate.py`，退出码 **0=通过 / 2=存在违规 / 1=报告缺失或不可解析**；
    参数 `--report / --reports-dir / --baseline / --max-drop / --no-baseline`；用 `get_settings()`（**不** `validate_runtime`，故无需任何密钥）。
  - **真实运行取证（2026-10-18）**：
    - `uv run python scripts/check_eval_gate.py` → 报告 `eval_20261018_021445.json`（faithfulness 0.989 / context_precision 0.940 /
      context_recall 0.993 / answer_relevancy 0.895），输出「通过 ✅」，**退出码 0**。
    - `uv run python scripts/check_eval_gate.py --max-drop 0.001` → 抓到
      `[baseline_drop] context_recall 相对基线下降 0.0069（> 允许 0.0010）：1.0000 → 0.9931`，输出「未通过 ❌（1 项）」，**退出码 2**。
    - `uv run ruff check app scripts tests` → All checks passed；`uv run pytest -q` → **146 passed, 1 skipped**（跳过为缺 `langfuse`）。
  - **接入 CI**：新建 `.github/workflows/ci.yml`（push / PR / 手动触发，**全程无密钥**）：
    `astral-sh/setup-uv` + `uv sync --frozen` → `ruff check` → `pytest -q -m "not integration"` → `check_eval_gate.py`。
  - **关键取舍**：**不在 CI 里重跑 RAGAS**——重跑需密钥、有真实费用、约 7 分钟且受 TPM 限速；
    故把评测定为**发布前手动步骤**（`scripts/run_eval.py` 产报告并入库），CI 只对**已入库报告**做秒级、无密钥、可复现的确定性校验。
  - **文档回写**：README 新增 §6.5（门禁做法/取舍/实测退出码/CI 工作流）与 §7 取舍条目、§8 命令块补门禁命令；
    `docs/interview_talk.md` 新增 §5.5、Q13，§5 单测数改 146、§6.1「下一步 ①」标记完成。
- **Progress Notes**（实现要点存档）:
  - 门禁刻意**不重跑评测**：把「评测（重、要密钥、有费用）」与「校验（轻、确定性）」解耦，是本次的核心设计决定；
    CI 里「什么都跑」会带来额度消耗与不可复现的外部抖动。
  - 违规以**结构化 dict 列表**返回（含 `kind/metric/value/threshold/drop`）而非直接抛异常，便于单测断言与后续接 PR 评论。
  - `latest_report` 只匹配 `eval_*.json`，天然排除 `ablation_*` / `judge_cross_check_*` / `session_cost_*` 报告，避免误校验。
- **Priority**: medium
- **Depends On**: Task 11, Task 17
- **Description**:
  - 把评测阈值与拒答率目标固化为可执行门禁：跌出阈值 / 基线跌幅超限 / 有执行失败即非零退出。
  - 以「校验已入库报告」的形态接入 CI，避免在 CI 中重跑 RAGAS（密钥/费用/耗时/TPM 限速）。
- **Acceptance Criteria Addressed**: AC-9, AC-12
- **Test Requirements**:
  - `rule` TR-18.1: 健康报告通过；缺指标 / 低于阈值 / 拒答率不足 / 存在执行失败 / 基线跌幅超限五类违规均能被 `check_report` 检出。
  - `rule` TR-18.2: 门禁退出码契约（0 通过 / 2 违规 / 1 报告缺失）稳定，且 `latest_report` 取最新一份报告。

## Task 19: 语义缓存——复用 Qdrant 跳过重复出题的检索管线（作品集增强 P2-①）

- **Status**: `completed`（实现、单测、真实 A/B 实测取证、文档回写完成）
- **Completion Evidence**:
  - **实现**：新建 `app/rag/cache.py`——`SemanticCache(settings, store, embedder)`，`get(query, scope)` /
    `put(query, scope, result)` / `clear()` / `stats()`；新增配置 `SEMANTIC_CACHE_ENABLED`（默认 true）/
    `SEMANTIC_CACHE_THRESHOLD`（0.95）/ `SEMANTIC_CACHE_TTL_SECONDS`（1800）。
  - **复用 Qdrant**：`QdrantStore` 新增独立集合 `{collection}__semantic_cache`（与知识库集合隔离，**共用同一客户端**，
    规避 local 模式多客户端争抢文件锁）；新增 `upsert_semantic_cache` / `search_semantic_cache` / `delete_semantic_cache` /
    `clear_semantic_cache`（server 模式对 `scope` 建 payload 索引）。
  - **接入检索**：`AdaptiveRetriever.retrieve` 先查缓存，命中直接返回；未命中则走原 `_retrieve_uncached` 并回填。
    `scope = kb_types|top_k` 避免跨检索范围污染。`RetrievedChunk.from_dict` / `RetrievalResult.from_dict` /
    `AdaptiveResult.from_dict` 提供与 `dataclasses.asdict` 的**无损往返**。
  - **失效防线**：TTL 超期即删点并视为未命中；上传摄入（`app/api/sessions.py`）与知识库摄入（`scripts/ingest_kb.py`）
    完成后调用 `clear()` 清空缓存。
  - **装配**：`app/api/deps.py` 的 `build_services` 构造 `cache` 并注入 `AdaptiveRetriever`；`ServiceContainer` 新增 `cache` 字段。
  - **单测**：新建 `tests/test_semantic_cache.py`（5 例，真实 Qdrant local 模式 + 假 embedding/检索）：
    序列化无损往返、相似度命中/未命中与 scope 隔离、关闭开关为空操作、`clear()` 失效、
    `AdaptiveRetriever` 二次查询命中缓存（底层 Retriever 只被调用 1 次）。`uv run pytest -q -m "not integration"`
    → **147 passed, 1 skipped**（4 例集成 deselected；全量 151 passed）；`ruff check app tests scripts` → All checks passed。
  - **真实 A/B 实测（2026-10-18，同一批真实面试、仅缓存开关不同）**：新建 `scripts/measure_semantic_cache.py`
    （复用 `measure_session_cost._run_sessions/_aggregate`，新增 `cache_enabled` / `reset_cache` 参数与 `set_enabled` 开关）。
    `uv run python scripts/measure_semantic_cache.py --sessions 6 --questions 5` →
    报告 `data/eval/reports/semantic_cache_20261018_093958.{json,md}`：
    - `ask_question` 平均 **9194.1ms → 3182.2ms（-65.4%）**，P50 **9110.8ms → 1720.7ms**；
    - 单场总耗时 **43.65s → 25.84s（-40.8%）**；单场成本（估）**¥0.052887 → ¥0.037707（-28.7%）**；
    - LLM 调用 **174 → 90（-48.3%）**；总 token **86847 → 59654（-31.3%）**；
    - rewrite 调用/token **54/10295 → 12/2281（-77.8%）**；eval 调用 **96 → 54（-43.8%）**；rerank **54 → 12（-77.8%）**；
    - 缓存自身：查询 18、命中 14、未命中 4、写入 4，**命中率 0.7778**（阈值 0.95，TTL 1800s）；
    - 两遍作答均分区间一致（1.0–1.4），**出题质量未观测到退化**。
  - **文档回写**：README 新增 §6.6 与 §7 两条取舍、§8 命令块；`docs/interview_talk.md` 新增 §5.6、Q14，
    Q10 改为「已落地语义缓存」、§6「下一步 」标记完成、单测数改 147；`.env.example` 补 3 项缓存配置。
- **Progress Notes**（实现要点存档）:
  - 选型**复用 Qdrant 而非引入 Redis**：检索结果本就以向量形态存于 Qdrant，命中判定天然是一次相似度查询；
    单独拉 Redis 属过度工程（多组件、多运维），且共用同一客户端规避了 local 模式的文件锁冲突。
  - 缓存**整包检索结果**而非只缓存 embedding：真正的开销是改写与自评两个 LLM（§6.1 显示自适应管线几乎每次跑满轮数），
    只缓存 embedding 省不到大头；整包重放用无损序列化往返保证「命中 == 重跑」。
  - 命中阈值取 **0.95** 偏保守：意图是「只对几乎完全相同的查询命中」，宁可不命中也不给近似答案；
    实测 0.778 命中率来自「同 track 重复出题」，4 次未命中恰为 4 个不同 track 的首次出现。
  - 加 `set_enabled` 运行时开关是为了**同环境 A/B**（同一实例即时切换），避免「两台机器/两次部署」引入的可比性噪声。
- **Priority**: high
- **Depends On**: Task 15, Task 16, Task 18
- **Description**:
  - 用查询级语义缓存跳过「查询改写 + 向量化 + 双路召回 + 精排 + 充足性自评」这条在重复出题上被反复执行的昂贵管线。
  - 复用已有 Qdrant 承载缓存，提供 TTL 与知识库变更失效两道防线，并给出真实 A/B 的延迟/成本/token 降幅。
- **Acceptance Criteria Addressed**: AC-8, AC-12
- **Test Requirements**:
  - `rule` TR-19.1: `AdaptiveResult` 经 `asdict` → `from_dict` 往返后与原对象相等（缓存重放等价性）。
  - `rule` TR-19.2: 相似度 ≥ 阈值且 scope 相同才命中；低于阈值或不同 scope 不命中；`enabled=False` 时 get/put 为空操作。
  - `rule` TR-19.3: `clear()` 后同查询不再命中；`AdaptiveRetriever` 二次相同查询由缓存服务、底层 Retriever 仅调用一次。

## Task 20: 上下文预算——压缩喂进模型的上下文体量（作品集增强 P2-②）

- **Status**: `completed`（实现、单测、真实 A/B 实测取证、文档回写完成）
- **Completion Evidence**:
  - **实现**：新建 `app/agent/context.py`——`ContextBudget`（不可变）含 `enabled` / `evidence_snippet_chars` /
    `asked_history_max` / `report_source_max` / `report_source_snippet_chars`，以及 `clip_evidence` / `recent_asked` /
    `limit_sources` 三个裁剪入口；`from_settings(settings, enabled=...)` 支持运行时覆盖；`legacy()` **逐值复刻改造前行为**
    （证据片段 150 字 / 已问问题不限 / 报告来源不限、片段 120 字）。新增配置 `CTX_BUDGET_ENABLED`（默认 true）/
    `CTX_EVIDENCE_SNIPPET_CHARS`（100）/ `CTX_ASKED_HISTORY_MAX`（6）/ `CTX_REPORT_SOURCE_MAX`（8）/
    `CTX_REPORT_SOURCE_SNIPPET_CHARS`（60）。
  - **接入节点**：`app/agent/graph.py` 的 `InterviewEngine.__init__` 新增 `ctx_budget` 参数（默认按配置构造），
    `_node_ask_question` 用 `clip_evidence` + `recent_asked`、`_node_assess_answer` 用 `clip_evidence`；
    `app/agent/report.py` 的 `_build_messages(state, budget)` 用 `limit_sources` + `report_source_snippet_chars`，
    `LLMReportGenerator` 新增 `budget` 参数（默认 `legacy()` 保证既有测试行为不变），`build_report_generator` 按配置开启。
  - **单测**：新建 `tests/test_context_budget.py`（6 例）：`legacy()` 逐值复刻、`from_settings` 开关切换、
    `recent_asked` 只保留最近 N、`limit_sources` 保序裁剪、报告 prompt 开预算时来源裁剪到 8 条且片段截断到 60 字、
    关预算时保留全部 12 条且片段截断到 120 字。`uv run pytest -q -m "not integration"` → **153 passed / 1 skipped**；
    `ruff check app tests scripts` → All checks passed。
  - **真实 A/B 实测（2026-10-18，同一批真实面试、两遍均开语义缓存以隔离单一变量）**：新建
    `scripts/measure_context_budget.py`（复用 `measure_session_cost._run_sessions`，新增 `budget_enabled` 参数；
    并为 `_aggregate` 增加**按父节点归属 LLM token** 的 `node_tokens` 聚合）。
    `uv run python scripts/measure_context_budget.py --sessions 6 --questions 6` →
    报告 `data/eval/reports/context_budget_20261018_100453.{json,md}`：
    - 全链路 LLM token **60749 → 54614（-10.1%）**；LLM 调用次数 **90 → 90**（只压缩单次上下文，不减调用）；
    - 单场成本（估）**¥0.038388 → ¥0.034582（-9.9%）**；单场总耗时 **29.02s → 26.3s（-9.4%）**；
    - 按节点：`node.report` **21228 → 18458（-13.0%，3538→3076/次）**、`node.assess_answer` **21902 → 19354（-11.6%，608→538/次）**、
      `node.ask_question` 14952 → 14150（-5.4%，498→472/次）；`node.followup` 调用 0（脚本化作答未触发追问分支）；
    - 两遍作答均分区间一致（1.0–1.6），**未观测到质量退化**。
  - **文档回写**：README 新增 §6.7 与 §7 一条取舍、§8 命令块、CI 单测数改 153；
    `docs/interview_talk.md` 新增 §5.7 与 Q15、Q7/Q10 更新、§6「下一步」标记完成、单测数改 153；
    `.env.example` 补 5 项预算配置。
- **Progress Notes**（实现要点存档）:
  - 定位依据来自**按父节点归属 token**（新增 `node_tokens` 聚合）：先量出 token 大头在 report/assess/ask 三处的上下文，
    再决定压缩对象，避免「盲目砍召回条数」。砍召回会伤 §6.1 已验证的 RRF 融合收益，压上下文则不动召回。
  - `legacy()` 的存在是为了**可控 A/B**：关闭开关即逐值复刻旧行为，保证两次测量只差一个变量；
    这也是本项目「所有优化都要有真实对照数字」原则的延续。
  - 权衡显式记录：片段截短会削弱「追问结合简历细节」的粒度与报告依据丰富度；实测成本/耗时下降约 10%，
    质量（均分区间）未观测到退化，但这批是脚本化作答，**不构成对真实候选人场景的强结论**，已在报告中如实标注。
- **Priority**: medium
- **Depends On**: Task 15, Task 19
- **Description**:
  - 把「喂进模型的上下文」显式预算化（证据片段长度、已问问题条数、报告来源条数与片段长度），在不减少召回的前提下压 token。
  - 提供关闭即复刻旧行为的开关，用同环境 A/B 给出真实 token/成本/耗时降幅，并覆盖单测。
- **Acceptance Criteria Addressed**: AC-8, AC-12
- **Test Requirements**:
  - `rule` TR-20.1: `ContextBudget.legacy()` 逐值等于改造前行为（150 字片段 / 不限条数 / 120 字来源片段）。
  - `rule` TR-20.2: 开启预算时证据片段按配置截断、`recent_asked` 只保留最近 N 条、`limit_sources` 按上限保序裁剪。
  - `rule` TR-20.3: 报告 prompt 在开/关预算两态下，知识来源条数与片段长度分别被裁剪 / 保留全量。

## Task 21: 公网限速与成本熔断——防额度滥用（作品集增强 P2-③）

- **Status**: `completed`（实现、单测、真实起服务 curl 取证、文档回写完成）
- **Completion Evidence**:
  - **实现**：新建 `app/api/quota.py`——`RateLimiter`（**固定窗口**，`window_seconds=60`，`check(key)`
    超限返回 `429 RATE_LIMITED` + `retry_after`）、`DailyBudget`（累计 `prompt/completion` token，按
    `LLM_INPUT_PRICE_PER_M` / `LLM_OUTPUT_PRICE_PER_M` 折算预估成本，任一超限返回 `503 BUDGET_EXHAUSTED`，
    跨日自动重置）、`QuotaManager`（`check_request` **先熔断后限速**、`record_usage`、`snapshot`）与模块级单例
    `get_quota()` / `reset_quota()`。新增配置 `RATE_LIMIT_ENABLED`（默认 true）/ `RATE_LIMIT_PER_MINUTE`（20）/
    `COST_BUDGET_ENABLED`（true）/ `DAILY_TOKEN_BUDGET`（1_000_000）/ `DAILY_COST_BUDGET_CNY`（20.0）/
    `LLM_INPUT_PRICE_PER_M`（2.0）/ `LLM_OUTPUT_PRICE_PER_M`（8.0）。
  - **接入接口**：`app/api/deps.py` 新增 `_client_key(request)`（优先取 `X-Forwarded-For` 首个 IP，否则
    `request.client.host`）与 `enforce_quota(request)`（不允许时抛 `ApiError`，带 `details` 与 `Retry-After` 头）；
    三个**会花钱**的接口（`POST /sessions`、`POST /sessions/{id}/messages`、`POST /sessions/{id}/finish`）挂
    `dependencies=[Depends(enforce_quota)]`；只读 GET 不限速。`app/api/errors.py` 的 `ApiError` 增 `headers` 字段并由
    处理器透传；`app/api/health.py` 新增只读 `GET /quota` 返回 `get_quota().snapshot()`。
  - **用量回灌**：`app/core/llm.py` 的 `DeepSeekChatClient` 新增 `usage_sink: Callable[[int,int],None]`，`chat()`
    成功后回灌 `(prompt_tokens, completion_tokens)`；`build_services` / `build_override_services` 注入
    `usage_sink=get_quota().record_usage`（`stream_chat` 不计数，实际面试路径只用 `chat`）。
  - **单测**：新建 `tests/test_quota.py`（11 例，FakeClock 控时钟）：限速超限 429 + 窗口恢复 + key 隔离、
    token 熔断、成本熔断、跨日重置、开关关闭恒放行、熔断优先于限速，以及 HTTP 集成
    （`POST /sessions` 429 + `Retry-After`；熔断 503；只读 `GET /quota` 不受限速；快照 shape）。
    `tests/conftest.py` 新增 autouse `_neutralize_quota`（默认关限速/熔断，避免串扰既有用例）。
    `uv run pytest -q -m "not integration"` → **164 passed / 1 skipped**（4 例集成 deselected；含集成共 168）；
    `ruff check app tests scripts` → All checks passed；`uv run python scripts/check_eval_gate.py` → 退出码 0。
  - **真实起服务 curl 取证（2026-10-18 本地 uvicorn，端口 8011/8012/8013，用完即停）**：
    - `RATE_LIMIT_PER_MINUTE=3` → 前 3 次 `201`，第 4/5 次 **`429` + 响应头 `Retry-After: 60`**；
    - `RATE_LIMIT_PER_MINUTE=1` → 第 2 次 `429`，body
      `{"error":{"code":"RATE_LIMITED","message":"请求过于频繁：每个客户端每分钟最多 1 次，请 56 秒后重试。","details":{"used_in_window":3,"limit":1}}}`；
    - `DAILY_TOKEN_BUDGET=0` → `POST /sessions` **`503 BUDGET_EXHAUSTED`**（message 含 "token 已达 0/0"）；
    - 只读 `GET /quota` 分别返回 `{"rate_limit":{"enabled":true,"per_minute":3},"budget":{"enabled":false}}`
      与 `{"rate_limit":{"enabled":false,"per_minute":null},"budget":{"enabled":true,"day":"2026-10-18","calls":0,"tokens":0,"cost_cny":0.0,"max_tokens":0,"max_cost_cny":20.0,...}}`。
  - **文档回写**：README 新增 §6.8 与 §7 一条取舍、§8 CI 单测数改 164；`docs/interview_talk.md` 新增 §5.8、
    §6「下一步」标记第 ⑤ 项完成、单测数改 164；`.env.example` 补 7 项配额配置。
  - **服务器侧验证（2026-10-18，腾讯云轻量 42.194.242.186，提交 `3db7863` 经 git bundle + reset 同步）**：
    `docker compose up -d --build` 重建后三容器（qdrant / api / streamlit）均 `healthy`；容器内
    `GET /health` → **200**；`GET /quota` → `{"rate_limit":{"enabled":true,"per_minute":20},"budget":{"enabled":true,
    "day":"2026-10-18","calls":0,"tokens":0,"cost_cny":0.0,"max_tokens":1000000,"max_cost_cny":20.0,...}}`（服务器 `.env`
    未显式配置时由新增默认值兜底，**防护在线上默认生效**）；公网前端 `:8501` → **200**。
- **Progress Notes**（实现要点存档）:
  - 选型**进程内存而非 Redis**：单实例部署下固定窗口与当日累计都不需要跨进程共享，拉 Redis 属过度工程
    （多一组件、多一份运维），与 §5.6「复用 Qdrant 而非另起缓存」同一取舍逻辑。
  - **熔断优先于限速**：先判断当日额度是否耗尽，再判单 IP 频率——额度没了限速放行也没意义，顺序保证
    「配额保护」成为最后一道闸门。
  - 限速**只挂会花钱的三个接口**、熔断做成**全局跨客户端**：前者避免误伤只读浏览（`GET /sessions`、`GET /quota`
    不限速）；后者保护的是「账号额度」而非「单用户公平性」——单用户能被限速挡住，聚合流量仍需总量闸门。
  - `usage_sink` 回调注入而非在配额模块内包 LLM：让 LLM 客户端保持对配额**零依赖**，配额只做「记账 + 判定」，
    依赖方向单向、可单测。
- **Priority**: high
- **Depends On**: Task 15, Task 19
- **Description**:
  - 给以公网为唯一入口的面试全链路加两层闸门：每 IP 每分钟请求数限速 + 每日 token/成本全局熔断，防额度滥用。
  - 提供只读用量快照接口与开关（关闭即行为不变），并以真实起服务 curl 取证状态码与响应头。
- **Acceptance Criteria Addressed**: AC-8, AC-12
- **Test Requirements**:
  - `rule` TR-21.1: 固定窗口内超限即 `429 RATE_LIMITED` 且带 `Retry-After`；窗口滚动后恢复；不同 key 互不影响。
  - `rule` TR-21.2: 当日 token 或成本达上限即 `503 BUDGET_EXHAUSTED`；跨日自动重置；熔断判定优先于限速。
  - `rule` TR-21.3: 开关关闭时恒放行；HTTP 集成下「花钱接口」受限、「只读 `GET /quota`」不受限，快照 shape 稳定。

## Task 22: 训练模式——单账号刷题练习与进度追踪（作品集增强）

- **Status**: completed
- **Completion Evidence**:
  - **实现**：
    - `app/agent/assess.py`（新建）：从 `graph.py` 抽出研判 prompt 与解析逻辑
      （`ASSESS_SYSTEM` / `ASSESS_USER` / `parse_assess_verdict` / `QUALITY_SCORE`），作为面试与训练
      **唯一研判事实源**，防两处 prompt 漂移；`graph.py` 改为导入（5 处引用替换）。
    - `app/train/auth.py`（新建）：`hash_password` / `verify_password`（scrypt，N=2^14/r=8/p=1，随机盐，
      格式 `scrypt$N$r$p$salt_hex$hash_hex`，`hmac.compare_digest` 防时序侧信道）；
      `issue_token` / `verify_token`（HMAC-SHA256 无状态令牌 `trn1.<b64url(payload)>.<b64url(sig)>`，
      payload 含 `scope="train"` + `exp`，与面试 session_id 零命名空间重叠）。
    - `app/train/questions.py`（新建）：解析题库 Markdown（`## Q01: 题干` + `- tags:` + `- 答案要点：`），
      `question_id = <文件名>#<Q编号>`；`draw(exclude_ids)` 优先排除已答、全答完回退全量随机
      （`random.SystemRandom`）；模块级缓存单例。
    - `app/api/train.py`（新建）：`POST /train/login`（专用登录限速防爆破；未配置三要素 → 503
      `TRAIN_NOT_CONFIGURED`；错凭据 → 401 `TRAIN_AUTH_FAILED`）、`POST /train/next`
      （鉴权 + `check_rate_only` 防抓题、不查熔断，响应**不含答案要点**）、`POST /train/answer`
      （鉴权 + `enforce_quota`；研判用共享 ASSESS prompt、evidence=答案要点；落库 `train_attempts`）、
      `POST /train/reveal`（鉴权 + `check_rate_only`；**不调 LLM、不落库**——看答案 ≠ 练过，不计入进度）、
      `GET /train/history`（最近作答 + 已练/总数 + 质量分布）。
    - `app/api/quota.py`：`QuotaManager` 新增 `check_rate_only(client_key)`（只查每 IP 限速不查成本熔断）。
    - `app/storage/db.py`：新增 `train_attempts` 表（attempt_id/username/question_id/source/question/
      answer/quality/feedback/created_at）+ `add_train_attempt` / `list_train_attempts` /
      `list_train_done_question_ids`。
    - `app/config/settings.py`：新增 `train_username` / `train_password_hash` / `train_token_secret` /
      `train_token_ttl_seconds=3600` / `train_login_rate_per_minute=10` / `train_qa_dir`；
      `.env.example` 补配置块与生成命令、明文 HTTP 警告。
    - 前端：`frontend/api_client.py` 加 `train_token` 与五个 train 方法（Bearer 头）；
      `frontend/views_train.py`（新建）登录表单 → 抽题 → 作答（**含「显示答案」按钮**：reveal 下发要点，
      看完可继续作答或直接下一题）→ 评分徽标（🟢/🟡/🔴）+ st.info 点评 +
      折叠参考答案要点 + 进度 metric/覆盖率 + 历史列表；401 自动清令牌回登录态；
      `frontend/app.py` 导航加「刷题训练」页（无 session 也可进入）。
  - **单测**：`tests/test_train.py`（15 例）：哈希往返/错密码/非法格式、令牌签发/过期/篡改/错密钥/
    面试 ID 不可冒充、题库解析与抽题排除回退、API 全链路（login→next→answer→history）、next 不含要点、
    未配置 503、错口令/无令牌 401、未知题 404、预算熔断 answer 503、登录限速 429、
    **reveal 返回答案要点且零 LLM 调用（handler 触发即失败的 MockTransport）、不落库进度为 0**、仓储层 CRUD；
    `tests/test_frontend.py` 新增 3 例（登录→作答全链路；显示答案按钮流程；401 回登录态）。
    `uv run pytest -q -m "not integration"` → **182 passed / 1 skipped**（含集成共 186）；
    `ruff check app tests scripts frontend` → All checks passed；
    `uv run python scripts/check_eval_gate.py` → 退出码 0。
  - **文档回写**：README 新增 §6.9、§4 目录补 train/ 与 views_train、§5 接口表加 5 行、§7 加两条取舍、
    单测数 164→182；`docs/interview_talk.md` 新增 §5.9、§6 下一步标记完成、单测数同步；
    `.env.example` 补训练模式配置块。
  - **服务器侧验证（2026-10-18，腾讯云轻量 42.194.242.186，提交 `b13f9db` 经 git bundle + reset 同步）**：
    `docker compose up -d --build` 重建后三容器（qdrant / api / streamlit）均 `healthy`；宿主机
    `GET /health` → **200**；真实凭据走通全链路——`POST /train/login` → 200 签发 `trn1.` 令牌；
    `POST /train/next` → 200 抽中 `ai_backend_qa_bank.md#Q11`（MySQL 索引/B+ 树，响应不含答案要点）；
    `POST /train/answer` 提交真实作答 → 200，真实 LLM 研判 `quality=good` + 中文点评 + 4 条参考答案要点，
    `attempt_id=1` 落库；`GET /train/history` → `done_count=1 / total_count=40 / quality_counts={"good":1}`；
    无令牌访问 `/train/next` → **401**；公网前端 `:8501` → **200**。
  - **踩坑记录（docker compose .env 插值）**：scrypt 哈希格式 `scrypt$N$r$p$salt$hash` 含 `$`，
    docker compose 读 `.env` 时把 `$1`/`$8` 当变量展开为空，导致容器内哈希被截断为 `scrypt6384`、
    登录恒 `TRAIN_AUTH_FAILED`；修复为 `.env` 中写 `$$`（compose 转义为 `$`）后 `printenv` 验证正确。
    远端 shell 对 heredoc 内 `$$` 会展开为 PID，故含 `$` 的行一律 base64 编码写入。
- **Progress Notes**（实现要点存档）:
  - **单账号 + 无状态令牌而非注册体系**：训练模式是个人刷题工具，多用户注册/邮箱验证属过度工程；
    凭据走 `.env`（scrypt 哈希，不明文），令牌 HMAC 签名免服务端会话存储，重启不丢登录态。
  - **抽共享研判 prompt 而非各写一份**：训练与面试「评价口径必须一致」是硬约束，抽 `assess.py`
    作唯一事实源，未来调 prompt 只改一处；训练侧研判失败**不沿用面试的默认 good 兜底**而直接抛错——
    训练场景答错没有后续追问补救，错评比报错更糟。
  - **答案要点只在作答后下发**：`/train/next` 响应绝不含要点（`TrainQuestionView` 模型层面就没有该字段），
    防泄题也防有人把接口当免费 LLM 代理套答案。
  - **分层限速**：login 专用限速防爆破 → next 用 `check_rate_only` 防抓题但不查熔断（抽题不花 LLM 钱，
    不该被全局熔断误伤）→ answer 挂 `enforce_quota` 完整防护（研判是真金白银的 LLM 调用）。
  - **题库直取不依赖 Qdrant**：`question_bank` 直接解析 Markdown 文件，无向量库环境（CI、轻量部署）也能跑。
- **Priority**: high
- **Depends On**: Task 21
- **Description**:
  - 与模拟面试分离的单账号刷题练习：scrypt 校验 + HMAC 短期令牌登录（凭据走 .env），从题库抽取
    自带答案要点的八股题，作答后复用面试同口径 DeepSeek 研判给 good/partial/poor 与点评，
    `train_attempts` 表落库支撑进度追踪；接口鉴权 + 分层限速防公网额度滥用。
- **Acceptance Criteria Addressed**: AC-8, AC-12
- **Test Requirements**:
  - `rule` TR-22.1: 凭据校验（scrypt 往返/错密码/非法格式）与令牌（签发/过期/篡改/错密钥/面试 ID 不可冒充）
    全部通过；未配置三要素时 login 返回 503 `TRAIN_NOT_CONFIGURED`。
  - `rule` TR-22.2: `/train/next` 排除该用户已答题（全答完回退全量随机）且响应不含答案要点；
    `/train/answer` 对未知题返回 404，研判失败不兜底放行；作答落库 `train_attempts` 且 history 统计正确。
  - `rule` TR-22.3: 无令牌/错口令 401；登录专用限速超限 429；`/train/answer` 受全局预算熔断 503；
    前端登录→作答全链路与 401 自动回登录态可复现。
  - `rule` TR-22.4: `/train/reveal` 返回答案要点且全程零 LLM 调用、不落库（history 进度不变）；
    未知题 404、无令牌 401；前端「显示答案」后可继续作答或直接下一题，进度不前进。

## 任务依赖与建议排期

| 日 | 任务 |
|---|---|
| D1 | Task 1、Task 3（可并行起步，先跑通两个 API 验活） |
| D2 | Task 2（示例数据 + 摄入） |
| D3 | Task 4（混合检索核心） |
| D4 | Task 5（Agentic RAG）、Task 6（持久化，部分并行） |
| D5-D6 | Task 7（状态机） |
| D7 | Task 8（报告）、Task 9（API/SSE） |
| D8 | Task 10（前端） |
| D9 | Task 11（评测）+ Task 12（可观测） |
| D10 | Task 13（部署上线）、Task 14（收口） |
| D11 | Task 15（作品集增强：检索消融对照 + 真实会话成本/延迟实测） |
| D12 | Task 16（作品集增强：judge 自评偏置治理 + 异源 judge 交叉验证） |
| D13 | Task 17（作品集增强：引用质量度量升级为来源 P/R/F1） |
| D14 | Task 18（作品集增强：评测回归门禁接入 CI） |
| D15 | Task 19（作品集增强 P2-①：语义缓存降低重复出题延迟/成本/ token） |
| D16 | Task 20（作品集增强 P2-②：上下文预算压缩 token 与成本） |
| D17 | Task 21（作品集增强 P2-③：公网限速与成本熔断防额度滥用） |
| D18 | Task 22（训练模式：单账号刷题练习与进度追踪） |
