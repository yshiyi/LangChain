import os
os.environ["USER_AGENT"] = "my-langchain-app/1.0"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",  # required by the client but ignored by LM Studio
    model="qwen2.5-14b-instruct",  # must match the model name shown in LM Studio
)

# Create a prompt
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical document writer."),
    ("user", "{input}")
])

# Create an output parser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

output_parser = JsonOutputParser()

chain = prompt | llm | output_parser
message = chain.invoke({"input": "What is LangChain? Please response with JSON format, using question for the query and answer for the response."})

# Create a web document loader
# from langchain_community.document_loaders import PlaywrightURLLoader

# loader = PlaywrightURLLoader(
#     urls=["https://openai.com/index/harness-engineering/"],
#     remove_selectors=["header", "footer", "nav"]
# )

# docs = loader.load()

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

loader = AsyncChromiumLoader(["https://openai.com/index/harness-engineering/"])
raw_docs = loader.load()

bs_transformer = BeautifulSoupTransformer()
docs = bs_transformer.transform_documents(
    raw_docs,
    tags_to_extract=["p", "h1", "h2", "h3", "article"]
)


# Create chunks using embedding models
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    model="text-embedding-nomic-embed-text-v1.5",
    check_embedding_ctx_length=False
)

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, embeddings)

# Create a retriever
from langchain_core.prompts import PromptTemplate

retriever = vector.as_retriever(search_kwargs={"k": 3})
# retriever.search_kwargs = {"k": 3}
docs = retriever.invoke("What is Harness Engineering?")

prompt_template = """
You are a technical assistant.
Your mission is to generate response to user's query based on the information provided below.
Make sure your response is grounded to the information provided below. Don't hallucinate when you are not sure.
If the information provided below is not enough to help you with generating trustable response, you may reply "I don't know."

Information given:
{info}

User's query:
{question}
"""

template = PromptTemplate.from_template(prompt_template)

prompt = template.format(info = docs, question = "What is Harness Engineering?")

response = llm.invoke(prompt)


# response = llm.invoke("Hello")
print(response)