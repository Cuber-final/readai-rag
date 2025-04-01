from typing import Dict, Any
from llama_index.core import PromptTemplate

class ReadingPrompts:
    """电子书阅读场景的提示词模板"""
    
    @staticmethod
    def get_summary_prompt() -> str:
        """获取章节总结的提示词"""
        return """请对以下文本内容进行总结，重点关注：
1. 主要观点和论据
2. 关键概念和术语
3. 重要事件或情节
4. 作者的主要意图

文本内容：
{text}

请以结构化的方式提供总结。"""

    @staticmethod
    def get_question_prompt() -> str:
        """获取问题生成的提示词"""
        return """基于以下文本内容，生成3-5个有助于理解的问题：
1. 确保问题覆盖文本的主要观点
2. 包含不同难度级别的问题
3. 问题应该促进深度思考

文本内容：
{text}

请以列表形式提供问题。"""

    @staticmethod
    def get_explanation_prompt() -> str:
        """获取概念解释的提示词"""
        return """请解释以下概念或术语，并提供：
1. 基本定义
2. 关键特征
3. 相关示例
4. 与其他概念的关系

概念/术语：
{concept}

上下文：
{context}

请以结构化的方式提供解释。"""

    @staticmethod
    def get_connection_prompt() -> str:
        """获取知识点关联的提示词"""
        return """分析以下文本内容，找出：
1. 与已学知识的联系
2. 可能的应用场景
3. 相关的延伸阅读建议
4. 需要进一步探索的问题

当前文本：
{text}

已学知识：
{previous_knowledge}

请以结构化的方式提供分析。"""

    @staticmethod
    def get_reading_guide_prompt() -> str:
        """获取阅读指导的提示词"""
        return """基于以下文本内容和阅读目标，提供阅读指导：
1. 重点内容提示
2. 阅读策略建议
3. 可能遇到的难点
4. 理解检查点

文本内容：
{text}

阅读目标：
{reading_goal}

请以结构化的方式提供指导。"""

    @staticmethod
    def get_quiz_prompt() -> str:
        """获取测验题生成的提示词"""
        return """基于以下文本内容，生成测验题：
1. 包含多种题型（选择题、填空题、简答题）
2. 覆盖不同难度级别
3. 包含答案和解析

文本内容：
{text}

请以结构化的方式提供测验题。"""

    @staticmethod
    def get_reading_analysis_prompt() -> str:
        """获取阅读分析的提示词"""
        return """分析以下文本的阅读特征：
1. 文本难度评估
2. 主要写作手法
3. 语言特点
4. 结构特征

文本内容：
{text}

请以结构化的方式提供分析。"""

    @staticmethod
    def get_reading_suggestion_prompt() -> str:
        """获取阅读建议的提示词"""
        return """基于以下阅读历史和兴趣，提供阅读建议：
1. 推荐阅读顺序
2. 相关主题推荐
3. 阅读策略建议
4. 学习目标设定

阅读历史：
{reading_history}

兴趣领域：
{interests}

请以结构化的方式提供建议。"""

SENTENCE_QA_TEMPLATE = PromptTemplate(
    """你是一个专注于帮助用户理解文档内容的AI助手。现在用户选择了一个句子并提出了问题。
    
    选中的句子是:
    {selected_text}
    
    相关的上下文是:
    {context_str}
    
    用户的问题是:
    {query_str}
    
    请基于选中的句子和上下文,详细回答用户的问题。回答时请特别关注用户选中的句子,
    同时结合上下文来确保理解准确。如果上下文中的信息不足以回答问题,请明确指出。
    
    回答:
    """
)

SENTENCE_CONTEXT_TEMPLATE = PromptTemplate(
    """基于用户选中的句子:
    {selected_text}
    
    请总结这句话的关键信息,并说明它在当前上下文中的作用。
    如果这句话涉及到特定概念、术语或引用,请一并解释。
    
    总结:
    """
) 