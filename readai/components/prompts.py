DEFAULT_QA_GENERATE_PROMPT_TMPL_ZH = """\
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

QA_GENERATE_PROMPT_TMPL_ZH = """\
以下是上下文信息：
---------------------
{context_str}
---------------------
基于上述上下文信息（不依赖先验知识），生成与以下查询相关的问题。
您的身份是一名教师/教授。您的任务是为即将到来的测验/考试制定 {num_questions_per_chunk} 个问题。问题应涵盖文档的不同方面，且仅限于提供的上下文信息。
"""
