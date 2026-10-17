"""Task 11 离线测试：评测集校验、拒答检测、受控生成、批跑管线、报告渲染。

RAGAS 真实指标评判依赖外部模型服务，属于 integration 范畴，不在此文件。
"""

from __future__ import annotations

import pytest

from app.eval.cross_judge import build_comparison, compare_metric, pearson, render_markdown
from app.eval.dataset import (
    CATEGORIES,
    DatasetValidationError,
    EvalDataset,
    EvalQuestion,
    load_dataset,
    validate_dataset,
)
from app.eval.gate import check_report, latest_report
from app.eval.generator import EMPTY_CONTEXT_HINT, build_generation_messages, generate_answer
from app.eval.metrics_ragas import METRIC_KEYS, JudgeSpec, _is_rate_limited
from app.eval.refusal import is_refusal
from app.eval.report import build_report, mean
from app.eval.report import render_markdown as render_eval_markdown
from app.eval.runner import RowOutcome, run_pipeline

DATASET_PATH = "data/eval/eval_dataset.json"
KNOWN_SOURCES = {
    "sample_resume_zh.md",
    "ai_backend_jd_zh.md",
    "ai_backend_jd_en.md",
    "ai_backend_qa_bank.md",
}


def _q(qid: str, category: str = "qa", **overrides) -> EvalQuestion:
    payload = {
        "id": qid,
        "category": category,
        "question": f"问题 {qid}",
        "ground_truth": ["要点一", "要点二"],
        "expected_sources": [] if category == "oos" else ["ai_backend_qa_bank.md"],
        "expected_kb_types": [] if category == "oos" else ["qa"],
    }
    payload.update(overrides)
    return EvalQuestion(**payload)


# ---------- 评测集结构 ----------


def test_real_dataset_file_validates():
    dataset = load_dataset(DATASET_PATH)
    assert len(dataset.questions) >= 40
    validate_dataset(dataset, known_sources=KNOWN_SOURCES)
    for category in CATEGORIES:
        assert dataset.by_category(category), f"缺少类别 {category}"


def test_dataset_too_small_rejected():
    dataset = EvalDataset(questions=[_q("x1"), _q("x2", category="oos")])
    with pytest.raises(DatasetValidationError, match="不少于 40"):
        validate_dataset(dataset)


def test_dataset_duplicate_ids_and_unknown_sources_rejected():
    questions = [
        _q("dup", category="resume", expected_sources=["ghost.md"]),
        _q("dup", category="jd"),
    ]
    dataset = EvalDataset(questions=questions)
    with pytest.raises(DatasetValidationError) as exc_info:
        validate_dataset(dataset, known_sources=KNOWN_SOURCES)
    msg = str(exc_info.value)
    assert "id 重复" in msg
    assert "ghost.md" in msg


def test_oos_questions_must_not_expect_sources():
    dataset = EvalDataset(
        questions=[_q("bad-oos", category="oos", expected_sources=["a.md"])]
    )
    with pytest.raises(DatasetValidationError, match="oos 诱导题 expected_sources"):
        validate_dataset(dataset)


def test_dataset_filter_and_ground_truth_text():
    dataset = EvalDataset(
        questions=[_q("r1", category="resume"), _q("q1"), _q("o1", category="oos")]
    )
    assert [q.id for q in dataset.filter(categories=["resume", "oos"])] == ["r1", "o1"]
    assert dataset.filter(limit=2)[0].id == "r1"
    assert _q("x").ground_truth_text == "要点一；要点二"
    assert _q("x", category="oos").kb_type_filter is None


# ---------- 拒答检测 ----------


@pytest.mark.parametrize(
    "answer",
    [
        "知识库中未提及相关信息，无法回答。",
        "抱歉，简历中并未提到候选人的薪资期望，我无法确定。",
        "该问题不在我的资料范围内，不能为您提供编造的型号。",
        "刷好评违反平台规则，我不能协助完成此类请求。",
    ],
)
def test_refusal_detected(answer):
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "候选人张明有 3 年后端开发经验，曾主导 RAG 客服项目。",
        "RRF 通过对每路排名取倒数求和来融合多路召回结果。",
        # 长答案正文"描述拒答策略"不算拒答（Q23 类题目）
        "缓解幻觉的手段包括：检索不到时让模型显式说明知识库未提及，而不是强行作答；"
        "此外还要降低 temperature 并要求引用来源，保证答案有据可依。" * 2,
        "块太小时语义破碎，模型拿到只言片语无法回答；因此应采用父子文档策略，"
        "小块检索、大块喂给模型，并保留标题层级等元数据。" * 2,
        "",
    ],
)
def test_normal_answer_not_flagged_as_refusal(answer):
    assert not is_refusal(answer)


def test_long_answer_with_refusal_cue_at_head_still_refusal():
    # 较长但开门见山拒答（oos-04 类违规请求解释）；前文无第三人称标记
    answer = (
        "抱歉，这个请求我无法协助。批量刷好评并规避平台风控属于伪造评价、"
        "操纵平台数据的违规行为，可能违反平台规则甚至相关法律法规，也违背职业道德，"
        "因此我不能提供此类脚本或规避检测的具体方法。如果你关注的是电商高并发"
        "场景下的正当技术实践，我可以介绍订单风控、限流或幂等设计方面的公开方案，"
        "建议把精力放在正当的系统优化与合规运营上，谢谢理解。"
    )
    assert len(answer) > 160
    assert is_refusal(answer)


# ---------- 受控生成 ----------


class _FakeChatResult:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, content: str = " 答案内容 ") -> None:
        self.content = content
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _FakeChatResult(self.content)


def test_build_messages_includes_numbered_evidence():
    messages = build_generation_messages("问题", ["上下文甲", "上下文乙"])
    assert messages[0]["role"] == "system"
    assert "只能依据" in messages[0]["content"]
    assert "【资料 1】" in messages[1]["content"]
    assert "上下文甲" in messages[1]["content"]
    assert "【资料 2】" in messages[1]["content"]


def test_build_messages_empty_context_hint():
    messages = build_generation_messages("问题", [])
    assert EMPTY_CONTEXT_HINT in messages[1]["content"]


async def test_generate_answer_strips_and_uses_eval_purpose():
    llm = _FakeLLM("  答案  ")
    answer = await generate_answer(llm, "q", ["c"])
    assert answer == "答案"
    assert llm.calls[0]["kwargs"]["purpose"] == "eval"


# ---------- 批跑管线 ----------


class _FakeChunk:
    def __init__(self, chunk_id: str, source: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.source = source
        self.text = text


class _FakeItem:
    def __init__(self, chunk, score: float = 0.9) -> None:
        self.chunk = chunk
        self.rerank_score = score


class _FakeRetrieval:
    def __init__(self, chunks, rounds: int = 1, stopped: str = "sufficient") -> None:
        self.chunks = chunks
        self.rounds = rounds
        self.stopped_reason = stopped
        self.retrieval = self  # 模拟 AdaptiveResult.retrieval 指向自身


class _FakeRetriever:
    def __init__(self, result: _FakeRetrieval, raise_for: str | None = None) -> None:
        self.result = result
        self.raise_for = raise_for
        self.calls: list[tuple] = []

    async def retrieve(self, query, kb_types, top_k):
        self.calls.append((query, kb_types, top_k))
        if self.raise_for and query == self.raise_for:
            raise RuntimeError("boom")
        return self.result


async def test_run_pipeline_assembles_contexts_and_sources():
    chunks = [
        _FakeItem(_FakeChunk("c1", "ai_backend_qa_bank.md", "要点内容")),
        _FakeItem(_FakeChunk("c1", "ai_backend_qa_bank.md", "重复块"), 0.8),
        _FakeItem(_FakeChunk("c2", "ai_backend_qa_bank.md", "第二块"), 0.7),
    ]
    retriever = _FakeRetriever(_FakeRetrieval(chunks, rounds=2))
    rows = await run_pipeline([_q("qa-01")], retriever, _FakeLLM("正常答案"))
    row = rows[0]
    assert row.contexts == ["要点内容", "第二块"]  # chunk_id 去重
    assert row.retrieved_sources == ["ai_backend_qa_bank.md"]
    assert row.source_hit is True
    assert row.rounds == 2
    assert row.refusal_detected is False
    # 评测题 kb 过滤透传
    assert retriever.calls[0][1] == ["qa"]


async def test_run_pipeline_oos_empty_context_refusal():
    retriever = _FakeRetriever(_FakeRetrieval([], rounds=1, stopped="max_rounds_reached"))
    llm = _FakeLLM("知识库中未提及相关信息，无法回答。")
    rows = await run_pipeline([_q("oos-01", category="oos")], retriever, llm)
    row = rows[0]
    assert row.contexts == []
    assert row.source_hit is None
    assert row.refusal_detected is True


async def test_run_pipeline_records_error_without_aborting():
    good = _FakeRetrieval([_FakeItem(_FakeChunk("c1", "a.md", "t"))])
    retriever = _FakeRetriever(good, raise_for="问题 boom")
    rows = await run_pipeline(
        [_q("boom"), _q("qa-02")], retriever, _FakeLLM("答案")
    )
    assert len(rows) == 2
    errored = next(r for r in rows if r.id == "boom")
    assert errored.error and "boom" in errored.error
    assert next(r for r in rows if r.id == "qa-02").error is None


# ---------- 报告聚合与渲染 ----------


def _row(
    qid: str,
    category: str,
    *,
    source_hit: bool | None = True,
    refusal: bool = False,
    answer: str = "答案",
    contexts: int = 2,
    rounds: int = 1,
) -> RowOutcome:
    return RowOutcome(
        id=qid,
        category=category,
        question=f"问题 {qid}",
        ground_truth="要点",
        expected_sources=[] if category == "oos" else ["a.md"],
        contexts=[f"ctx{i}" for i in range(contexts)],
        retrieved_sources=[] if category == "oos" else ["a.md"],
        answer=answer,
        refusal_detected=refusal,
        source_hit=source_hit,
        rounds=rounds,
    )


def _dataset() -> EvalDataset:
    return EvalDataset(
        version="1.0",
        questions=[
            _q("resume-01", category="resume"),
            _q("qa-01", category="qa"),
            _q("oos-01", category="oos"),
        ],
    )


def test_mean_ignores_none():
    assert mean([1.0, None, 3.0]) == 2.0
    assert mean([None, None]) is None


def test_build_report_pass_case_and_markdown():
    rows = [
        _row("resume-01", "resume", rounds=2),
        _row("qa-01", "qa"),
        _row("oos-01", "oos", source_hit=None, refusal=True, contexts=0),
    ]
    scores = {
        "resume-01": {
            "faithfulness": 0.95,
            "context_precision": 0.9,
            "context_recall": 0.92,
            "answer_relevancy": 0.96,
        },
        "qa-01": {
            "faithfulness": 0.9,
            "context_precision": 0.85,
            "context_recall": 0.88,
            "answer_relevancy": 0.9,
        },
    }
    report = build_report(
        _dataset(), rows, scores, config={"use_adaptive": True, "top_k": 6}
    )
    assert report["passed"] is True
    overall = report["metrics"]["overall"]
    assert overall["faithfulness"] == pytest.approx(0.925, abs=1e-6)
    assert report["retrieval"]["oos_refusal_rate"] == 1.0
    assert report["retrieval"]["in_scope_refusal_rate"] == 0.0
    assert report["retrieval"]["source_hit_rate"] == 1.0
    assert report["baseline"]["present"] is False

    md = render_eval_markdown(report)
    assert "# RAGAS 离线评测报告" in md
    assert "faithfulness" in md and "题库外诱导题明细" in md
    assert "全部通过" in md


def test_source_f1_property_edges():
    ok = RowOutcome(
        id="x", category="qa", question="q", ground_truth="g",
        expected_sources=["a.md"], contexts=[], retrieved_sources=[],
        source_precision=0.5, source_recall=0.5,
    )
    assert ok.source_f1 == pytest.approx(0.5)
    both_zero = RowOutcome(
        id="y", category="qa", question="q", ground_truth="g",
        expected_sources=["a.md"], contexts=[], retrieved_sources=[],
        source_precision=0.0, source_recall=0.0,
    )
    assert both_zero.source_f1 is None
    missing = RowOutcome(
        id="z", category="qa", question="q", ground_truth="g",
        expected_sources=["a.md"], contexts=[], retrieved_sources=[],
        source_precision=1.0,
    )
    assert missing.source_f1 is None


def test_report_citation_precision_recall_and_citation_rate():
    rows = [
        RowOutcome(
            id="qa-01", category="qa", question="q1", ground_truth="g1",
            expected_sources=["a.md", "b.md"], contexts=["c1", "c2"],
            retrieved_sources=["a.md", "c.md"], answer="依据【资料1】作答。",
            source_hit=True, source_precision=0.5, source_recall=0.5,
        ),
        RowOutcome(
            id="qa-02", category="qa", question="q2", ground_truth="g2",
            expected_sources=["a.md", "b.md"], contexts=["c1"],
            retrieved_sources=["a.md", "b.md"], answer="直接作答，无引用标记。",
            source_hit=True, source_precision=1.0, source_recall=1.0,
        ),
    ]
    report = build_report(_dataset(), rows, {}, config={})
    ret = report["retrieval"]
    assert ret["source_hit_rate"] == 1.0
    assert ret["source_precision"] == pytest.approx(0.75)
    assert ret["source_recall"] == pytest.approx(0.75)
    assert ret["source_f1"] == pytest.approx(0.75)  # (0.5 + 1.0) / 2
    assert ret["answer_citation_rate"] == pytest.approx(0.5)
    # 二值命中率恒为 1.0，而 P/F1 能区分「召回里混入非期望来源」
    assert ret["by_category"]["qa"]["source_precision"] == pytest.approx(0.75)
    payload = {r["id"]: r for r in report["rows"]}["qa-01"]
    assert payload["source_f1"] == pytest.approx(0.5)
    assert payload["source_recall"] == pytest.approx(0.5)

    md = render_eval_markdown(report)
    assert "来源精确率 P" in md and "答案显式引用标记率" in md


def test_build_report_fail_and_baseline_delta():
    rows = [_row("qa-01", "qa"), _row("oos-01", "oos", source_hit=None, refusal=False)]
    scores = {
        "qa-01": {
            "faithfulness": 0.5,  # 低于 0.85
            "context_precision": 0.4,
            "context_recall": 0.55,  # 低于 0.8
            "answer_relevancy": 0.9,
        }
    }
    baseline = {
        "generated_at": "2026-01-01T00:00:00+08:00",
        "metrics": {
            "overall": {
                "faithfulness": 0.6,
                "context_precision": 0.5,
                "context_recall": 0.7,
                "answer_relevancy": 0.85,
            }
        },
    }
    report = build_report(
        _dataset(), rows, scores, config={"use_adaptive": False, "top_k": 6},
        baseline=baseline,
    )
    assert report["passed"] is False
    assert report["thresholds_pass"]["faithfulness"] is False
    assert report["oos_refusal_pass"] is False
    deltas = report["baseline"]["deltas"]
    assert deltas["faithfulness"] == pytest.approx(-0.1, abs=1e-6)
    assert deltas["answer_relevancy"] == pytest.approx(0.05, abs=1e-6)
    md = render_eval_markdown(report)
    assert "未全部通过" in md and "↓0.100" in md


def test_is_rate_limited_detects_throttling():
    assert _is_rate_limited(RuntimeError("Error code: 429 - rate limit exceeded"))
    assert _is_rate_limited(RuntimeError("TPM limit reached, please retry"))
    assert not _is_rate_limited(RuntimeError("Connection reset by peer"))
    assert not _is_rate_limited(ValueError("invalid schema"))


# ---------- 评测回归门禁 ----------


def _healthy_report(**overrides) -> dict:
    report = {
        "metrics": {
            "overall": {
                "faithfulness": 0.99,
                "context_precision": 0.94,
                "context_recall": 0.99,
                "answer_relevancy": 0.90,
            }
        },
        "retrieval": {"oos_refusal_rate": 1.0},
        "errors": [],
    }
    report.update(overrides)
    return report


def test_gate_passes_on_healthy_report():
    assert check_report(_healthy_report()) == []


def test_gate_flags_below_threshold_and_missing_metric():
    report = _healthy_report()
    report["metrics"]["overall"]["faithfulness"] = 0.80  # 门槛 0.85
    del report["metrics"]["overall"]["context_recall"]
    kinds = {(v["kind"], v.get("metric")) for v in check_report(report)}
    assert ("below_threshold", "faithfulness") in kinds
    assert ("missing_metric", "context_recall") in kinds


def test_gate_flags_oos_refusal_and_exec_errors():
    report = _healthy_report(
        retrieval={"oos_refusal_rate": 0.5},
        errors=[{"id": "qa-01", "error": "TimeoutError"}],
    )
    kinds = {v["kind"] for v in check_report(report)}
    assert "oos_refusal" in kinds
    assert "exec_errors" in kinds


def test_gate_baseline_drop_respects_max_drop():
    baseline = {"metrics": {"overall": {"context_recall": 0.99, "answer_relevancy": 0.90}}}
    # 无跌幅 → 不违规
    assert check_report(_healthy_report(), baseline=baseline, max_drop=0.05) == []
    # 跌幅 0.01：宽松容忍（0.05）下合规，收紧到 0.005 后被拦下
    slight = _healthy_report()
    slight["metrics"]["overall"]["context_recall"] = 0.98
    assert check_report(slight, baseline=baseline, max_drop=0.05) == []
    strict = check_report(slight, baseline=baseline, max_drop=0.005)
    assert any(v["kind"] == "baseline_drop" for v in strict)
    # 跌幅 0.09 > 0.05 → 违规
    dropped = _healthy_report()
    dropped["metrics"]["overall"]["context_recall"] = 0.90
    bad = check_report(dropped, baseline=baseline, max_drop=0.05)
    assert any(v["kind"] == "baseline_drop" and v["metric"] == "context_recall" for v in bad)


def test_gate_baseline_none_is_not_a_failure():
    assert check_report(_healthy_report(), baseline=None) == []


def test_latest_report_picks_newest(tmp_path):
    assert latest_report(tmp_path) is None
    (tmp_path / "eval_20260101_000000.json").write_text("{}", encoding="utf-8")
    (tmp_path / "eval_20260202_000000.json").write_text("{}", encoding="utf-8")
    (tmp_path / "archived.txt").write_text("x", encoding="utf-8")
    newest = latest_report(tmp_path)
    assert newest is not None and newest.name == "eval_20260202_000000.json"


# ---------- 异源 judge 交叉验证 ----------


def test_pearson_basics():
    assert pearson([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert pearson([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None  # 方差为 0
    assert pearson([1.0, 2.0], [1.0, 2.0]) is None  # 样本不足


def test_compare_metric_pairs_and_missing():
    primary = [0.9, 0.8, None, 0.7]
    cross = [0.8, 0.8, 0.5, 0.7]
    out = compare_metric(primary, cross)
    assert out["n_pairs"] == 3
    assert out["n_missing"] == 1
    assert out["primary_mean"] == pytest.approx(0.8)  # None 不计入
    assert out["cross_mean"] == pytest.approx(0.7)
    assert out["delta"] == pytest.approx(0.1, abs=1e-6)
    assert out["mean_abs_diff"] == pytest.approx(0.0333, abs=1e-4)
    assert out["max_abs_diff"] == pytest.approx(0.1, abs=1e-6)
    assert out["agree_0.1"] == pytest.approx(1.0)
    assert out["agree_0.05"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["exact"] == pytest.approx(2 / 3, abs=1e-4)
    assert out["pearson"] == pytest.approx(0.866, abs=1e-3)


def test_compare_metric_all_missing():
    out = compare_metric([None], [None])
    assert out["n_pairs"] == 0
    assert out["mean_abs_diff"] is None
    assert out["agree_0.1"] is None
    assert out["pearson"] is None


def test_build_comparison_and_render_flags_self_inflation():
    rows = [_row("qa-01", "qa"), _row("qa-02", "qa")]
    primary = {
        "qa-01": dict.fromkeys(METRIC_KEYS, 0.9),
        "qa-02": dict.fromkeys(METRIC_KEYS, 0.9),
    }
    cross = {
        "qa-01": dict.fromkeys(METRIC_KEYS, 0.8),
        "qa-02": dict.fromkeys(METRIC_KEYS, 0.8),
    }
    rep = build_comparison(
        rows,
        primary,
        cross,
        primary=JudgeSpec("deepseek-chat", "https://api.deepseek.com", "sk"),
        cross=JudgeSpec("Qwen/Qwen2.5-72B-Instruct", "https://api.siliconflow.cn/v1", "sk"),
        config={"dataset": {"executed": 2, "counts": {"qa": 2}}},
    )
    assert rep["n_scored"] == 2
    assert rep["n_cells"] == 2 * len(METRIC_KEYS)
    for key in METRIC_KEYS:
        assert rep["self_inflation"][key] == pytest.approx(0.1, abs=1e-6)
    assert rep["overall_cell_agreement"]["agree_0.2"] == pytest.approx(1.0)
    assert rep["overall_cell_agreement"]["agree_0.05"] == pytest.approx(0.0)

    md = render_markdown(rep)
    assert "# RAGAS 异源 judge 交叉验证报告" in md
    assert "Qwen/Qwen2.5-72B-Instruct" in md
    assert "自评偏高" in md
