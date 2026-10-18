"""回答研判（assess）的共享 prompt 与解析：面试模式与训练模式唯一事实源。

两种模式必须对「同一份回答」给出一致评价，因此 system prompt、user 模板、
quality 取值集合与解析兜底逻辑全部集中在该模块，禁止各自另写一份。

- ``ASSESS_SYSTEM`` / ``ASSESS_USER``：研判 prompt（严格 JSON 输出）；
- ``parse_assess_verdict``：解析 LLM 输出；失败时默认 ``good`` 放行（面试不中断），
  训练模式自行决定是否沿用该兜底；
- ``QUALITY_SCORE``：quality → 1-5 分的映射（报告聚合用）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

ASSESS_SYSTEM = (
    "你是技术面试官，研判候选人的回答质量。"
    '严格输出 JSON：{{"quality": "good|partial|poor", "feedback": "一句话点评", '
    '"followup_question": "追问问题或空字符串"}}。'
    "quality=good 时 followup_question 为空；partial/poor 时给出一个精确针对回答漏洞、"
    "并结合证据或简历细节的追问问题。"
)
ASSESS_USER = "问题：{question}\n考察点：{focus}\n参考依据：\n{evidence}\n\n候选人回答：{answer}\n\n请研判。"

QUALITY_SCORE = {"good": 4, "partial": 3, "poor": 2}


def parse_assess_verdict(content: str) -> dict[str, Any]:
    """研判解析；失败时默认 quality=good（放行进入下一题，不中断面试）。"""
    try:
        data = json.loads(content)
        quality = str(data.get("quality") or "").strip().lower()
        if quality in QUALITY_SCORE:
            return {
                "quality": quality,
                "feedback": str(data.get("feedback") or "").strip(),
                "followup_question": str(data.get("followup_question") or "").strip(),
            }
    except json.JSONDecodeError:
        pass
    logger.warning("研判输出无法解析，默认按 good 放行：%r", content[:120])
    return {"quality": "good", "feedback": "研判解析失败，默认通过", "followup_question": ""}
