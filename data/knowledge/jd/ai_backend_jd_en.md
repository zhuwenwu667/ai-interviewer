# AI Application Developer (Backend / LLM Platform) - Job Description

## Position Overview

- Title: AI Application Developer, LLM Backend
- Team: Intelligent Platform Engineering
- Location: Hangzhou, China (hybrid/remote-friendly)
- Experience: 3+ years in backend engineering, 1+ year building LLM-powered applications
- Education: Bachelor's degree or above, Computer Science or related field preferred

## Responsibilities

1. Design and build the company's LLM application platform, including a RAG knowledge base, agent workflows, and a unified model gateway.
2. Own the retrieval-augmented generation pipeline: document parsing and chunking, embeddings, hybrid retrieval, reranking, and context management; continuously improve answer faithfulness and relevance.
3. Orchestrate multi-step agents with frameworks such as LangGraph, including tool calling, human-in-the-loop, state persistence, and failure recovery.
4. Build offline and online evaluation systems for RAG and LLM quality, and use data to drive decisions on prompts, retrieval strategies, and model selection.
5. Ensure performance and reliability under high concurrency through caching, asynchronous processing, rate limiting, graceful degradation, capacity planning, and end-to-end observability.
6. Collaborate with ML, frontend, and product teams to ship LLM features for customer support, marketing, and developer productivity.

## Required Qualifications

1. Strong Python engineering skills with FastAPI or an equivalent async web framework; solid understanding of the event loop, coroutines, and concurrency control.
2. Deep, hands-on understanding of the RAG lifecycle: hybrid retrieval (dense and sparse), reranking, query rewriting, and chunking strategies.
3. Experience with at least one mainstream agent framework (LangGraph, LlamaIndex, or AutoGen) and awareness of trade-offs among ReAct, plan-and-execute, and state-machine patterns.
4. Working knowledge of a production vector database (Qdrant, Milvus, or Weaviate) and its indexing fundamentals (HNSW parameters, distance metrics).
5. Solid backend fundamentals: MySQL indexing and transactions, Redis patterns, message queues, and Docker-based deployment.
6. Strong engineering habits: type hints, unit testing, structured error handling, logging, and code review.

## Preferred Qualifications

- Experience shipping RAGAS or custom LLM-as-a-judge evaluation pipelines, including awareness of metric definitions and evaluation pitfalls.
- Production experience with function calling, multi-agent collaboration, and agent memory.
- Familiarity with DeepSeek, Qwen, and other Chinese LLM APIs exposing OpenAI-compatible endpoints, and with sampling parameters such as temperature and top_p.
- Experience optimizing LLM application cost and latency via caching, model tiering, prompt compression, and streaming.
- Experience with Kubernetes and observability tools such as OpenTelemetry or Langfuse.

## Interview Focus

- Diagnosing RAG failure modes: missed recall, poor ranking, lost context, and hallucination, and the mitigation for each.
- System design: a multi-collection enterprise question-answering system with citation-level traceability and evaluation.
- Coding: asynchronous concurrency, idempotency, retry with backoff, and edge-case handling.
- Project depth: quantified outcomes, trade-offs behind technology choices, and concrete lessons learned rather than simply "we called an API".
