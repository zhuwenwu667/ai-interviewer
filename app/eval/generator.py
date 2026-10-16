"""受控的 RAG 答案生成：只依据检索上下文作答，无依据时显式拒答。

评测中的"生成"环节不走面试状态机（其 prompt 面向出题/追问），而使用严格接地
（grounded）的问答 prompt——这也是 RAGAS faithfulness / context_recall 的评判前提：
答案必须由 contexts 支撑，资料未覆盖时必须明说"知识库未提及"。
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.llm import ChatMessage, DeepSeekChatClient

SYSTEM_PROMPT = (
    "你是一名严谨的企业知识库问答助手。请严格遵守以下规则：\n"
    "1. 只能依据【参考资料】中的内容回答问题，不得使用资料之外的任何事实或推测；\n"
    "2. 资料未提及或信息不足以回答时，必须明确说明“知识库中未提及相关信息，无法回答”，"
    "不要编造数字、名称、时间或结论；\n"
    "3. 对于违法、违规或违背职业道德的请求，礼貌拒绝并说明原因；\n"
    "4. 直接给出答案，使用与问题相同的语言，简洁准确，可分点陈述；不要复述这些规则。"
)

EMPTY_CONTEXT_HINT = "（本次没有检索到任何参考资料）"


def build_generation_messages(
    question: str, contexts: Sequence[str]
) -> list[ChatMessage]:
    """组装严格接地问答的消息列表。"""
    if contexts:
        evidence = "\n\n".join(
            f"【资料 {i}】\n{text}" for i, text in enumerate(contexts, start=1)
        )
    else:
        evidence = EMPTY_CONTEXT_HINT
    user_content = f"参考资料：\n{evidence}\n\n问题：{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def generate_answer(
    llm: DeepSeekChatClient,
    question: str,
    contexts: Sequence[str],
    *,
    temperature: float = 0.1,
) -> str:
    """依据 contexts 生成答案；无 context 时模型仍被要求显式拒答。"""
    messages = build_generation_messages(question, contexts)
    result = await llm.chat(
        messages,
        purpose="eval",
        temperature=temperature,
    )
    return result.content.strip()
