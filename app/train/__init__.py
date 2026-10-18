"""训练模式（刷题练习）：与模拟面试隔离的独立轻量模块。

- 不复用 LangGraph 状态机：抽题 → 作答 → 单轮研判 → 展示答案要点的短闭环；
- 评价与面试模式同源（``app.agent.assess``），保证口径一致；
- 凭据/令牌见 ``app.train.auth``，题库解析见 ``app.train.questions``。
"""
