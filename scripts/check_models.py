"""三接口真实可用性验活：DeepSeek Chat / SiliconFlow Embedding / Rerank。

用法（项目根目录）：
    uv run python scripts/check_models.py            # 文本表格输出
    uv run python scripts/check_models.py --json     # JSON 结构化输出

任一接口不可用时退出码为 1。所有结论均来自真实 HTTP 调用，不做任何臆测。
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, dataclass

# 允许直接以脚本方式运行（scripts/ 不在包内）
sys.path.insert(0, ".")

from app.config import validate_runtime  # noqa: E402
from app.core.embedding import SiliconFlowEmbedder  # noqa: E402
from app.core.llm import DeepSeekChatClient  # noqa: E402
from app.core.reranker import SiliconFlowReranker  # noqa: E402


@dataclass
class CheckResult:
    interface: str
    provider: str
    model: str
    ok: bool
    detail: str


async def _check_chat(settings) -> CheckResult:
    client = DeepSeekChatClient(settings, timeout=30.0)
    try:
        result = await client.chat(
            [{"role": "user", "content": "ping，请回复 pong"}],
            purpose="rewrite",
            max_tokens=8,
        )
        return CheckResult(
            "chat", "DeepSeek", result.model, True,
            f"HTTP 200，返回 {len(result.content)} 字符：{result.content[:30]!r}",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("chat", "DeepSeek", settings.deepseek_model, False, str(exc))
    finally:
        await client.close()


async def _check_embedding(settings) -> CheckResult:
    embedder = SiliconFlowEmbedder(settings, timeout=30.0)
    try:
        vectors = await embedder.embed(["接口验活"])
        dim_ok = len(vectors) == 1 and len(vectors[0]) == settings.embedding_dim
        return CheckResult(
            "embedding", "SiliconFlow", settings.embedding_model, dim_ok,
            f"HTTP 200，返回 {len(vectors)} 个向量，维度={len(vectors[0]) if vectors else 0}"
            + ("" if dim_ok else f"（期望维度 {settings.embedding_dim}）"),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("embedding", "SiliconFlow", settings.embedding_model, False, str(exc))
    finally:
        await embedder.close()


async def _check_rerank(settings) -> CheckResult:
    reranker = SiliconFlowReranker(settings, timeout=30.0)
    try:
        items = await reranker.rerank(
            "接口验活", ["接口验活测试", "一篇无关的旅游日记"], top_n=2
        )
        ordered = all(items[i].score >= items[i + 1].score for i in range(len(items) - 1))
        return CheckResult(
            "rerank", "SiliconFlow", settings.reranker_model, ordered,
            f"HTTP 200，返回 {len(items)} 条结果且按分数降序："
            + ", ".join(f"#{it.index}={it.score:.4f}" for it in items),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("rerank", "SiliconFlow", settings.reranker_model, False, str(exc))
    finally:
        await reranker.close()


async def main(as_json: bool) -> int:
    try:
        settings = validate_runtime(require=("chat", "embedding"))
    except Exception as exc:  # noqa: BLE001 - 配置缺失需友好退出而非 traceback
        if as_json:
            print(json.dumps({"ok": False, "stage": "config", "error": str(exc)},
                             ensure_ascii=False, indent=2))
        else:
            print(f"[配置检查失败] {exc}")
        return 2
    results = await asyncio.gather(
        _check_chat(settings),
        _check_embedding(settings),
        _check_rerank(settings),
    )
    if as_json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        print("=" * 78)
        print(f"{'接口':<10}{'供应商':<13}{'模型':<28}{'结论':<6}详情")
        print("-" * 78)
        for r in results:
            print(f"{r.interface:<10}{r.provider:<13}{r.model:<28}{'可用' if r.ok else '不可用':<6}{r.detail}")
        print("=" * 78)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(as_json="--json" in sys.argv[1:])))
