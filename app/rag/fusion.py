"""RRF（Reciprocal Rank Fusion）多路召回融合排序。

score(doc) = Σ 1 / (k + rank)，rank 从 1 开始；
k 为平滑常数，越大各名次分数差距越平缓。
"""

from __future__ import annotations


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """融合多路排名列表。

    参数:
        rankings: 每一路按相关度降序的文档 ID 列表；同一文档出现在多路时分数累加。
        k: RRF 平滑常数（非负）。

    返回:
        (文档 ID, 融合分数) 列表，按分数降序；同分按 ID 字典序，保证结果确定性。
    """
    if k < 0:
        raise ValueError(f"RRF 平滑常数 k 必须为非负数，当前为 {k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
