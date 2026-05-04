"""
1. 按照模型功能的不同分类
- 非对话模型（LLMs，Text Model）
- 对话模型（Chat Models）
- 嵌入模型（Embedding Models）

1. 按照模型调用时，参数书写位置的不同（api-key，base_url 模型提供商的地址，model-name）
- 硬编码的方式：将参数写在代码中
- 使用环境变量的方式
- 使用配置文件的方式（推荐）

2. 具体API的调用
- 使用LangChain提供的API（推荐）
- 使用OpenAI官方的API
- 使用其他平台提供的API
"""

# -------------------------------------------------------------- #
# 1. 非对话模型。
# 输入接受文本字符串或 PromptValue 对象
# 输出总是文本字符串
# 不支持多轮对话上下文
# -------------------------------------------------------------- #

import os
os.environ["USER_AGENT"] = "my-langchain-app/1.0"

from langchain_openai import OpenAI

llm = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",  # required by the client but ignored by LM Studio
    model="qwen2.5-14b-instruct",  # must match the model name shown in LM Studio
)

message = llm.invoke("Write a poem about spring.")
print(message)

# -------------------------------------------------------------- #
# 2. Chat Models
# 输入接受消息列表 List[BaseMessage] 或 PromptValue，每条消息指定角色（如SystemMessage, HumanMessage, AIMessage）
# 输出带有角色的消息对象（BaseMessage 子类），通常是AIMessage
# 原生支持多轮对话，通过消息列表维护上下文，模型可基于完整对话历史生成回复
# -------------------------------------------------------------- #
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

chat_model = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",  # required by the client but ignored by LM Studio
    model="qwen2.5-14b-instruct",  # must match the model name shown in LM Studio
)

messages = [
    SystemMessage(content="I am a super intelligent assistant. My name is SuperStar."),
    HumanMessage(content="Hello, my name is Richard. Nice to meet you.")
]

response = chat_model.invoke(messages)

print(type(response))
print(response.content)

# -------------------------------------------------------------- #
# 2. Embedding Models
# -------------------------------------------------------------- #
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5",
    check_embedding_ctx_length=False
)

res = embeddings.embed_query("Please embedd this information.")
print(len(res))
