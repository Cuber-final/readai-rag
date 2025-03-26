from typing import Generator, Dict, Any, List, Optional
import logging
import json

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

from readai.core.config import settings
from readai.core.exceptions import LLMException

logger = logging.getLogger(__name__)


class LLMService:
    """LLM服务类"""
    
    def __init__(self):
        self.llm = self._get_llm()
        
    def _get_llm(self) -> Ollama:
        """获取LLM模型"""
        try:
            logger.info(f"初始化LLM模型: {settings.LLM_MODEL_NAME}")
            return Ollama(
                model=settings.LLM_MODEL_NAME,
                base_url=settings.OLLAMA_BASE_URL,
                request_timeout=120.0,
                temperature=0.7,
                top_p=0.95,
                context_window=8192,
            )
        except Exception as e:
            logger.error(f"初始化LLM模型失败: {str(e)}")
            raise LLMException(f"初始化LLM模型失败: {str(e)}")
    
    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        model_name: Optional[str] = None
    ) -> Generator[str, None, None]:
        """流式聊天完成"""
        try:
            # 如果指定了模型名称，则使用指定模型
            llm = self.llm
            if model_name and model_name != settings.LLM_MODEL_NAME:
                logger.info(f"使用指定模型: {model_name}")
                llm = Ollama(
                    model=model_name,
                    base_url=settings.OLLAMA_BASE_URL,
                    request_timeout=120.0,
                    temperature=temperature,
                    top_p=0.95,
                    context_window=8192,
                )
            
            # 转换消息格式
            chat_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    chat_messages.append(ChatMessage(role=MessageRole.USER, content=content))
                elif role == "assistant":
                    chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
                elif role == "system":
                    chat_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=content))
            
            # 流式生成响应
            response_iter = llm.stream_chat(chat_messages, temperature=temperature)
            
            for response in response_iter:
                yield response.delta
                
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            yield f"生成回答失败: {str(e)}"
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        model_name: Optional[str] = None
    ) -> str:
        """非流式聊天完成"""
        try:
            # 如果指定了模型名称，则使用指定模型
            llm = self.llm
            if model_name and model_name != settings.LLM_MODEL_NAME:
                logger.info(f"使用指定模型: {model_name}")
                llm = Ollama(
                    model=model_name,
                    base_url=settings.OLLAMA_BASE_URL,
                    request_timeout=120.0,
                    temperature=temperature,
                    top_p=0.95,
                    context_window=8192,
                )
            
            # 转换消息格式
            chat_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    chat_messages.append(ChatMessage(role=MessageRole.USER, content=content))
                elif role == "assistant":
                    chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))
                elif role == "system":
                    chat_messages.append(ChatMessage(role=MessageRole.SYSTEM, content=content))
            
            # 生成响应
            response = llm.chat(chat_messages, temperature=temperature)
            return response.message.content
            
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            raise LLMException(f"生成回答失败: {str(e)}")


# 创建全局LLM服务实例
llm_service = LLMService() 