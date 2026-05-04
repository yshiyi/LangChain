import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

dotenv.load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

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
from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_path="https://openai.com/index/harness-engineering/",
    bs_kwargs=dict(parse_only = bs4.SoupStrainer(id="UCAP-CONTENT"))
)

docs = loader.load()

# Create chunks using embedding models
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
documents = text_splitter.split_documents(docs)

vector = FAISS.from_documents(documents, embeddings)

# Create a retriever
from langchain_core.prompts import PromptTemplate

retriever = vector.as_retriever(search_kwarg={"k": 3})
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