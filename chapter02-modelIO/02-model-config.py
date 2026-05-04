"""
调用方法：
- OpenAI() / ChatOpenAI()：创建一个模型对象
- model.invoke()：执行调用，将用户输入发送给模型
- .content：提取模型返回的实际文本内容

必须设置的参数：
- base_url：大模型API服务的根地址
- api_key：模型秘钥
- model/model_name：指定要调用的具体大模型名称

其它参数：
- temperature
- max_tokens：限制生成文本的最大长度，防止输出过长。1 token ~ 3-4个英文字母，对话场景通常可设置为512-1024
"""

# -------------------------------------------------------------- #
# 1. 硬编码方式
# -------------------------------------------------------------- #
from langchain_openai import ChatOpenAI

# chat_model = ChatOpenAI(
#     base_url="http://127.0.0.1:1234/v1",
#     api_key="lm-studio",  # required by the client but ignored by LM Studio
#     model="qwen2.5-14b-instruct",  # must match the model name shown in LM Studio
# )

# -------------------------------------------------------------- #
# 2. 使用环境变量
# -------------------------------------------------------------- #
import os, dotenv

# # Method 1
# chat_model = ChatOpenAI(
#     model="qwen2.5-14b-instruct",  # must match the model name shown in LM Studio
#     base_url=os.environ["LM_BASE_URL"],
#     api_key=os.environ["LM_API_KEY"]
# )

# # Method 2
# dotenv.load_dotenv()
# chat_model = ChatOpenAI(
#     model="qwen2.5-14b-instruct",  # must match the model name shown in LM Studio
#     base_url=os.getenv("LM_BASE_URL"),
#     api_key=os.getenv("LM_API_KEY")
# )

# Method 3, 必须是OPENAI_，因为这是ChatOpenAI的默认值
os.environ['OPENAI_BASE_URL'] = os.getenv("LM_BASE_URL")
os.environ['OPENAI_API_KEY'] = os.getenv("LM_API_KEY")
chat_model = ChatOpenAI(
    model="qwen2.5-14b-instruct"
)

response = chat_model.invoke("Hello")
print(response)

# -------------------------------------------------------------- #
# 3. OPENAI
# -------------------------------------------------------------- #
from openai import OpenAI

# Use non-chat model
client = OpenAI(
    api_key="",
    base_url=""
)
response = client.completions.create(
    model="",
    prompt="",
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].text.strip())

# Use chat model
response = client.chat.completions.create(
    model="",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello."}
    ],
    temperature=0.5,
    max_tokens=150
)

print(response.choices[0].message)