from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import SystemMessage, HumanMessage

os.environ['OPENAI_BASE_URL'] = os.getenv("LM_BASE_URL")
os.environ['OPENAI_API_KEY'] = os.getenv("LM_API_KEY")

chat_model = ChatOpenAI(
    model="qwen2.5-14b-instruct"
)

"""
invoke()
Input:
- string
- List[BaseMessage]

Output:
BaseMessage: AIMessage <class 'langchain_core. message.ai.AIMessage'>


Message type:
- SystemMessage: 用来设计AI行为规则或背景信息
- HumanMessage: 表示来自用户的输入
- AIMessage: 存储AI回复的内容
- ChatMessage: 可以自定义角色的通用消息类型
- FunctionMessage/ToolMessage: 函数调用/工具消息，用于函数调用结果的消息类型
"""
system_message = SystemMessage(content="You are an expert in English Education.")
human_message = HumanMessage(content="Please help me make a plan to study English used at work.")

messages = [system_message, human_message]

# response = chat_model.invoke(messages)

# print(response.content)

"""
Runnable类 定义的公共的调用方法：
- invoke：处理单条输入，等待LLM推理完成后再返回调用结果。当用户发出请求后，系统在后台等待模型生成完整响应，然后一次性将全部结果返回。
- stream：流式响应，逐字输出LLM的响应结果。更像是实时对话，更贴近人类交互习惯，适合构建强调实时反馈的应用。
- batch：处理批量输入
"""

# stream()
chat_model_stream = ChatOpenAI(
    model="qwen2.5-14b-instruct",
    streaming=True
)

messages = [HumanMessage(content="Hello, please introduce yourself.")]
# print("Start output:")
# for chunk in chat_model_stream.stream(messages):
#     print(chunk.content, end="", flush=True)

# print("\n End")

# batch
messages1 = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="请帮我介绍一下什么是机器学习")
]
messages2 = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="请帮我介绍一下什么是AIGC")
]
messages3 = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="请帮我介绍一下什么是大模型技术")
]
messages = [messages1, messages2, messages3]

# response = chat_model.batch(messages)
# print(response)

"""
Async
应该与 asyncio 的 await 语法一起使用以实现并发

- astream : 异步流式响应
- ainvoke : 异步处理单条输入
- abatch : 异步处理批量输入
- astream_log : 异步流式返回中间步骤，以及最终响应
- astream_events : （测试版）异步流式返回链中发生的事件（在 langchain-core 0.1.14 中引入）
"""
import asyncio
import time
async def async_call(llm):
    await asyncio.sleep(5) # 模拟异步操作
    print("异步调用完成")
async def perform_other_tasks():
    await asyncio.sleep(5) # 模拟异步操作
    print("其他任务完成")
async def run_async_tasks():
    start_time = time.time()
    await asyncio.gather(
        async_call(None), # 示例调用，使用None模拟LLM对象
        perform_other_tasks()
    )
    end_time = time.time()
    return f"总共耗时：{end_time - start_time}秒"

if __name__ == "__main__":
    result = asyncio.run(run_async_tasks())
    print(result)