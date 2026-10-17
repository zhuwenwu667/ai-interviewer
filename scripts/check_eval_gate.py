"""评测回归门禁脚本：校验最新评测报告，跌出阈值即非零退出。

用法（项目根目录）：
    uv run python scripts/check_eval_gate.py                 # 校验最新报告
    uv run python scripts/check_eval_gate.py --report <路径>
    uv run python scripts/check_eval_gate.py --max-drop 0.03 # 收紧基线跌幅容忍
    uv run python scripts/check_eval_gate.py --no-baseline   # 跳过基线对比

退出码：0=通过；2=存在违规（可在 CI 中直接作为失败信号）；1=报告缺失/无法解析。

为何不在此重跑 RAGAS：重跑需密钥与真实费用、耗时数分钟，不适合 CI。
本门禁只做**对已入库报告的确定性校验**，把评测本身当作发布前的手动步骤。
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from app.config import get_settings  # noqa: E402
from app.eval.gate import check_report, latest_report  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测回归门禁：跌出阈值即非零退出")
    parser.add_argument("--report", default="", help="指定报告路径；默认取 reports 目录最新一份")
    parser.add_argument("--reports-dir", default="data/eval/reports", help="报告目录")
    parser.add_argument("--baseline", default="data/eval/baseline.json", help="基线报告路径")
    parser.add_argument("--max-drop", type=float, default=0.05, help="允许相对基线的最大跌幅")
    parser.add_argument("--no-baseline", action="store_true", help="跳过基线对比")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()

    report_path = (
        settings.resolve(args.report)
        if args.report
        else latest_report(settings.resolve(args.reports_dir))
    )
    if report_path is None:
        print(f"[门禁] 未在 {args.reports_dir} 找到评测报告（{args.reports_dir}/eval_*.json）")
        return 1

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[门禁] 报告无法解析：{report_path}（{type(exc).__name__}: {exc}）")
        return 1

    baseline = None
    baseline_path = settings.resolve(args.baseline)
    if not args.no_baseline and baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[门禁] 基线无法解析：{baseline_path}（{exc}）")

    print(f"[门禁] 报告：{report_path}")
    print(f"[门禁] 生成时间：{report.get('generated_at')}")
    overall = (report.get("metrics") or {}).get("overall", {}) or {}
    for key, value in overall.items():
        print(f"[门禁]   {key:<18} {value}")
    print(
        "[门禁] 基线对比："
        + (f"启用（允许跌幅 {args.max_drop:.4f}）" if baseline else "未启用")
    )
    print(f"[门禁] 允许最大跌幅：{args.max_drop:.4f}")

    violations = check_report(report, baseline=baseline, max_drop=args.max_drop)
    if not violations:
        print("[门禁] 通过 ✅")
        return 0

    print(f"[门禁] 未通过 ❌（{len(violations)} 项）")
    for item in violations:
        print(f"[门禁]   - [{item['kind']}] {item['detail']}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
