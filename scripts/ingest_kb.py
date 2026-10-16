"""知识库一键摄入脚本。

用法（项目根目录）：
    uv run python scripts/ingest_kb.py              # 增量/幂等摄入 data/knowledge
    uv run python scripts/ingest_kb.py --rebuild    # 删除集合并全量重建

目录约定（data/knowledge 下三类）：
    resume/  -> 简历知识库
    jd/      -> 岗位 JD 知识库
    qa/      -> 八股题库
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, ".")

from app.config import validate_runtime  # noqa: E402
from app.core.embedding import SiliconFlowEmbedder  # noqa: E402
from app.rag.ingestion import ingest_path  # noqa: E402
from app.rag.loader import SUPPORTED_SUFFIXES  # noqa: E402
from app.rag.models import KbType  # noqa: E402
from app.storage.qdrant_store import QdrantStore  # noqa: E402

KB_DIRS = (KbType.RESUME, KbType.JD, KbType.QA)


def _discover_files(root: Path) -> list[tuple[Path, KbType]]:
    found: list[tuple[Path, KbType]] = []
    for kb_type in KB_DIRS:
        directory = root / kb_type.value
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                found.append((path, kb_type))
    return found


async def main(rebuild: bool) -> int:
    settings = validate_runtime(require=("embedding",))
    data_root = settings.resolve("data/knowledge")
    files = _discover_files(data_root)
    if not files:
        print(f"[警告] {data_root} 下未发现任何可摄入文档（resume/jd/qa）")
        return 1

    store = QdrantStore(settings)
    embedder = SiliconFlowEmbedder(settings)
    try:
        if rebuild:
            store.reset_collection()
            print("[重建] 已清空旧集合")

        totals: dict[str, int] = {kb.value: 0 for kb in KB_DIRS}
        for path, kb_type in files:
            result = await ingest_path(
                store,
                embedder,
                path,
                kb_type,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            totals[kb_type.value] += result["chunks"]
            flag = f"替换旧点 {result['deleted']} 个" if result["deleted"] else "新增"
            print(
                f"[{kb_type.value:<6}] {path.name:<32} "
                f"{result['chunks']:>3} chunks（{flag}）"
            )

        total = store.count()
        print("-" * 60)
        print(
            f"摄入完成：resume={totals['resume']} jd={totals['jd']} "
            f"qa={totals['qa']}；集合总点数={total}，Qdrant({settings.qdrant_mode})"
        )
        return 0
    finally:
        await embedder.close()
        store.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(rebuild="--rebuild" in sys.argv[1:])))
