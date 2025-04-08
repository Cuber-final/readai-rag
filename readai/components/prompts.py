DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""

QA_GENERATE_PROMPT_TMPL_ZH_V1 = """\
以下是上下文信息：
---------------------
{context_str}
---------------------
基于上述上下文信息（不依赖先验知识），生成与以下查询相关的问题。
您的身份是一名教师/教授。您的任务是为即将到来的测验/考试制定 {num_questions_per_chunk} 个问题。问题应涵盖文档的不同方面，且仅限于提供的上下文信息。
"""

QA_GENERATE_PROMPT_TMPL_ZH = """\
##任务：根据以下书籍片段内容，生成具有多维度的讨论性和理解性问题，帮助用户深入理解该段内容。
##Examples:

input_case1:
上下文内容：
---------------------
在《论语》中，孔子提出了“仁”的概念，认为“仁”是人与人之间相互关爱、尊重和理解的基础。他强调，只有具备“仁”的品质，才能真正实现社会的和谐与稳定。
---------------------
请根据上下文生成2个问题
output:
孔子所倡导的“仁”在现代社会中具有怎样的现实意义？
如何理解“仁”是实现社会和谐与稳定的基础？

input_case2:
上下文内容：
---------------------
《物种起源》中，达尔文提出了自然选择学说，认为生物的进化是通过自然选择的过程实现的。他指出，适应环境的个体更容易生存和繁衍后代，从而推动了物种的进化。
---------------------
请根据上下文生成1个问题
output:
适应环境的个体更容易生存和繁衍后代，这一观点如何解释生物多样性的形成？

##下面是用户输入：
上下文内容：
---------------------
{context_str}
---------------------
请根据上下文生成{num_questions_per_chunk}个问题，要求问题至少涉及以下其中一个方面，问题字数长度在100字以内：
1. 片段的核心概念、观点、情节或事件。
2. 作者的意图或观点背后的逻辑推理。
3. 对该段内容的分析、批判或反思。
4. 如果有相关的背景知识或隐含信息，探讨它们的影响。
5. 生成至少一个假设性问题或假设情境，探索不同的可能性。
6. 比较该段内容与其他类似情节或观点的异同。

##格式要求：请参考上面的例子格式生成问题，尽量不携带数字编号并且只需要输出问题，多个问题分行输出即可
##输出：
"""

QA_GENERATE_EASY = """\
You are an assistant helping users read Chinese books.
Based on the context below, generate {num_questions_per_chunk} **easy** questions that test basic understanding or facts.

Context:
---------------------
{context_str}
---------------------

Rules:
- Questions must be written in **Chinese**.
- Do not include answers or question numbers.
- Each question must be concise (<= 20 characters).
- Only ask about **explicit content** from the context (e.g., names, definitions, facts).
- Avoid "这", "那", "根据" and similar vague words.

Example question:
某理论的核心概念是?
"""
