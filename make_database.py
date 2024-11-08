from langchain_community.document_loaders import WebBaseLoader

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import os

from dotenv import load_dotenv
load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

urls = [
    "https://www.techfor.id/cara-mudah-membuat-dompet-digital-wallet-mata-uang-kripto/", # 1
    "https://crypto-wallet.id/menghubungkan-coinbase-ke-metamask-di-indonesia/", # 2
    "https://crypto-wallet.id/cara-mentransfer-dari-coinbase-ke-metamask-di-indonesia//", # 3
    "https://id.beincrypto.com/belajar/cara-lengkap-mempersiapkan-wallet-metamask/", # 4
    "https://www.cropty.io/id/nft-wallet", # 5
    "https://www.cryptomedia.id/cryptopedia/apa-itu-nft/", # 6
    "https://ethereum.org/id/whitepaper/", # 7

    # Security
    "https://ethereum.org/id/roadmap/secret-leader-election/", # 8
    "https://ethereum.org/id/roadmap/pbs/", # 9
    "https://ethereum.org/id/staking/withdrawals/", # 10
    "https://ethereum.org/id/governance/", # 11

    "https://ethereum.org/id/roadmap/danksharding/", # 12

    # User Experience
    "https://ethereum.org/id/roadmap/account-abstraction/", # 13
    "https://ethereum.org/id/roadmap/verkle-trees/", # 14

    # History
    "https://ethereum.org/id/history/", # 15

    "https://www.nurhidayat.web.id/blog/teknologi-blockchain", # 16
    "https://academy.binance.com/id/articles/positives-and-negatives-of-blockchain", # 17
    "https://indodax.com/academy/fluktuasi-adalah/", # 18
    "https://crypto.com/university/id/what-are-crypto-derivatives-options-futures", # 19
    "https://coinvestasi.com/belajar/hedging-cara-aman-lindungi-kekayaan-crypto", # 20
    "https://teknologi.id/finance/mengenal-tiket-nft-solusi-inovatif-dalam-industri-tiket-acara", # 21
    "https://www.binance.com/id/blog/nft/apa-itu-tiket-nft-dan-bagaimana-cara-kerjanya-421499824684904022", # 22

    "https://www.binance.com/id/blog/nft/cara-membeli-nft-dalam-4-langkah-mudah-421499824684903165", # 23
    "https://dkid.media/2023/07/11/mengenal-tiket-nft-solusi-baru-untuk-tiket-acara/", # 24
    "https://indodax.com/academy/panduan-cara-jual-nft-di-opensea-agar-cepat-laku/", # 25
    
]

docs = [WebBaseLoader(url).load() for url in urls]
documents = [item for sublist in docs for item in sublist]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=2500,
    chunk_overlap=250,

)

chunked_docs = text_splitter.split_documents(documents)

vectordb = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

index_name = "chatbot"

if index_name not in vectordb.list_indexes().names():
    vectordb.create_index(
        name=index_name,
        dimension=3072,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )


docsearch = PineconeVectorStore.from_documents(chunked_docs, embeddings, index_name=index_name)

print(docsearch.similarity_search("what is agent memory?", k=3))