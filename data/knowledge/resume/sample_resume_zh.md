# 张明 - 后端开发工程师 / AI 应用开发方向

## 基本信息

- 姓名：张明
- 学历：硕士，计算机科学与技术，某 985 高校，2023 年毕业
- 工作年限：3 年
- 求职意向：AI 应用开发工程师 / 大模型后端工程师
- 联系方式：zhangming.example@email.com

## 专业技能

- 编程语言：Python（精通）、Go（熟悉）、SQL、Shell
- Web 框架：FastAPI、Django、Flask，理解 ASGI 与异步编程
- 大模型技术：LangChain、LlamaIndex、LangGraph，熟悉 Prompt Engineering、RAG、Function Calling、Agent 编排
- 向量与检索：Qdrant、Milvus、Elasticsearch，理解向量索引 HNSW、混合检索、RRF 融合、Rerank
- 数据存储：MySQL（索引优化、事务隔离级别）、Redis（缓存、分布式锁、限流）、PostgreSQL
- 中间件与运维：Kafka、RabbitMQ、Docker、Docker Compose、Kubernetes 基础、Nginx、Prometheus + Grafana
- 工程能力：pytest 单元/集成测试、Git Flow、CI/CD（GitHub Actions）、结构化日志与链路追踪

## 工作经历

### 杭州云启科技有限公司 | 后端开发工程师 | 2023.07 - 至今

参与公司客服智能化与业务中台建设，从 0 到 1 负责两个核心系统的后端与大模型能力落地。

## 项目经历

### 项目一：基于 RAG 的企业智能客服系统（核心项目，2024.03 - 2025.06）

- 项目背景：公司客服团队每月处理约 10 万次咨询，知识库有 3000+ 篇文档，传统关键词检索命中率低、答案无溯源。
- 技术栈：Python 3.11、FastAPI、LangChain、Qdrant、bge-m3、bge-reranker-v2-m3、DeepSeek、Redis、MySQL、Docker Compose。
- 个人职责与产出：
  - 设计"查询改写（Multi-Query/HyDE）+ 向量与 BM25 混合检索 + RRF 融合 + Rerank 精排"的 RAG 管线，问答采纳率从 52% 提升到 78%。
  - 实现 Agentic RAG：LLM 自评检索质量，上下文不足时自动改写问题重检索，将"知识库有答案但没召回"的比例从 18% 降到 6%。
  - 用 LangGraph 编排"意图识别 → 工具选择（知识库/工单查询/转人工）→ 生成 → 安全审核"流程，替代原先难以维护的超长 if-else 链。
  - 回答附带 chunk 级证据引用，支持点击跳转到原文档段落，客诉误答率下降 40%。
  - 构建 200 条 RAGAS 评测集并接入 CI，用 faithfulness、context recall 等指标做回归，防止 prompt 与检索参数劣化。
  - 通过 Redis 缓存高频问题的检索结果、对 embedding 调用做批处理，P95 响应时间从 6.2s 降到 2.1s。

### 项目二：电商订单中台高并发改造（2023.09 - 2024.02）

- 项目背景：大促期间下单接口峰值 QPS 从日常 2000 冲到 15000，出现超卖与接口超时。
- 技术栈：Go、Python（数据同步服务）、MySQL、Redis、Kafka、Nginx。
- 个人职责与产出：
  - 用 Redis Lua 脚本实现库存预扣的原子操作，解决"查库存-扣库存"竞态导致的超卖问题，大促零超卖。
  - 引入 Kafka 异步下单削峰，配合幂等键与本地消息表保证最终一致性，下单接口 P99 从 800ms 降到 150ms。
  - 基于 Prometheus + Grafana 搭建业务大盘，定位到慢 SQL 后通过联合索引与覆盖索引优化，核心查询耗时下降 90%。

### 在校项目：分布式爬虫与舆情分析平台（2022.06 - 2022.12）

- 基于 Scrapy-Redis 的分布式爬虫，使用布隆过滤器做 URL 去重，日均采集 500 万网页；
- 用 TextRank 与早期 LLM API 做热点摘要，相关成果获校级竞赛二等奖。

## 教育背景

### 某 985 高校 | 计算机科学与技术 | 硕士 | 2020.09 - 2023.06

- 主修课程：高级算法、分布式系统、机器学习、自然语言处理；GPA 3.7/4.0。
- 研究方向：信息检索与文本挖掘，发表中文核心论文一篇（短文本聚类方向）。

## 自我评价

- 工程与算法结合型后端：既能处理高并发、缓存、数据库等传统后端问题，也能独立完成 RAG/Agent 应用的设计、评测与优化。
- 重视量化验证：习惯用评测集、指标和日志驱动优化决策，而非凭感觉调 prompt。
