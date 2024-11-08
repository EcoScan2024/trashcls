from typing import List, Tuple

from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_function_messages

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool

from langchain_core.utils.function_calling import convert_to_openai_function
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.pydantic_v1 import BaseModel

from langchain_community.vectorstores import Pinecone as PC
from pinecone import Pinecone

import os

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY") # Search Engine

search = TavilySearchAPIWrapper()

description = """"Mesin pencari yang dioptimalkan untuk hasil yang komprehensif, akurat, \
dan terpercaya. Berguna ketika Anda perlu menjawab pertanyaan \
tentang peristiwa terkini atau informasi terbaru. \
Inputnya harus berupa kueri pencarian. \
Jika pengguna menanyakan sesuatu yang Anda tidak ketahui, \
Anda sebaiknya menggunakan alat ini untuk melihat apakah alat ini dapat menyediakan informasi, \
Selalu jawab pertanyaan menggunakan bahasa indonesia, \
Selalu katakan "Mencari di Internet..." di awal jawaban dan berikan paragraf baru untuk jawaban pertama."""

tavily_tool = TavilySearchResults(api_wrapper=search, description=description)

vectorstore_tools_description = (
 "Alat untuk membantu dokumentasi tentang dukungan layanan informasi mengenai dunia Blockchain dan pemahaman lebih tentang NFT"
 "Gunakan alat ini untuk mencari informasi mengenai Cara Membuat e-Wallet, Transparansi dan Jejak Transaksi pada Blockchain, Keamanan dan Keaslian NFT serta Kemudahan Transfer NFT dan lain sebagainya."
 "Buatlah jawaban sedetail mungkin agar pengguna dapat memahami maksudnya"
 "'Selalu ucapkan 'Terima kasih telah bertanya!' dan menggunakan Bahasa Indonesia"
 "Jika anda tidak menggunakan tools docstore untuk menjawab maka jawab saja 'Maaf saya tidak mengetahui jawabanya, terimakasih!' "
 "Jika anda tidak mengetahui jawaban jangan membuat jawaban sendiri hanya gunakan data dari docstore dan jawab saja 'Maaf saya tidak mengetahui jawabanya, terimakasih!'"
)

vectordb = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
docsearch = PC.from_existing_index("chatbot", embeddings)

retriever = docsearch.as_retriever()

vectorstore_tool = create_retriever_tool(retriever, "docstore", vectorstore_tools_description)

tools = [vectorstore_tool, tavily_tool]

# Set up LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    # model="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=3048,
    cache=True,
)

assistant_system_message = """You are a helpful assistant. \
If the docstore tools don't provide relevant information then make an answer with tavily tools to search for information on the internet, then end with 'Mendapatkan Jawaban Dari Internet' """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])


def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


agent = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)


class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput
)