from langchain_community.vectorstores import InMemoryVectorStore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document, BaseDocumentCompressor
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from typing import Sequence
import os
from dotenv import load_dotenv
load_dotenv()

# 1. 封装你的排序算法为“文档压缩器”
class MyLangChainReranker(BaseDocumentCompressor):
    def compress_documents(self, documents: Sequence[Document], query: str, callbacks=None) -> Sequence[Document]:
        
        # 模拟：简单取前 2 篇
        return documents[:2]

# 2. 准备基础检索器
llm = ChatOpenAI(model=str(os.getenv("LLM_NAME")), api_key=str(os.getenv("LLM_API_KEY")), base_url=str(os.getenv("LLM_BASE_URL")))

embedding = HuggingFaceEmbeddings(model_name=str(os.getenv("EMBEDDING_PATH")), model_kwargs={"device": "cuda"})  # 使用 GPU)
prompt_template = """
Given these texts:
-----
{context}
-----
Please answer the following question:
{query}
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "query"],
)

texts = [
    "Basquetball is a great sport.",
    "Fly me to the moon is one of my favourite songs.",
    "The Celtics are my favourite team.",
    "This is a document about the Boston Celtics",
    "I simply love going to the movies",
    "The Boston Celtics won the game by 20 points",
    "This is just a random text.",
    "Elden Ring is one of the best games in the last 15 years.",
    "L. Kornet is one of the best Celtics players.",
    "Larry Bird was an iconic NBA player.",
]

# Create a retriever
retriever = InMemoryVectorStore.from_texts(texts, embedding).as_retriever(
    search_kwargs={"k": 10}
)
query = "What can you tell me about the Celtics? Answer in Chinese."

# Get relevant documents ordered by relevance score
docs = retriever.invoke(query)

# Create and invoke the chain:
with get_openai_callback() as cb:
    chain = create_stuff_documents_chain(llm, prompt)
    response = chain.invoke({"context": docs, "query": query})
    print(response)
    print(f"总 Token 数: {cb.total_tokens}")
    print(f"Prompt Token 数: {cb.prompt_tokens}")
    print(f"完成 Token 数: {cb.completion_tokens}")
    print(f"预估总费用 (USD): ${cb.total_cost}")

# ==========================================
# 【实验组 2】：包装上你的压缩/排序算法
# ==========================================
with get_openai_callback() as cb:
    my_compressor = MyLangChainReranker()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=my_compressor, 
        base_retriever=retriever
    )

    # # 最终送给 LLM 的只有经过你算法筛选后的文档
    docs_experiment = compression_retriever.invoke(query)
    chain = create_stuff_documents_chain(llm, prompt)
    response = chain.invoke({"context": docs_experiment, "query": query})
    print(response)
    print(f"总 Token 数: {cb.total_tokens}")
    print(f"Prompt Token 数: {cb.prompt_tokens}")
    print(f"完成 Token 数: {cb.completion_tokens}")
    print(f"预估总费用 (USD): ${cb.total_cost}")