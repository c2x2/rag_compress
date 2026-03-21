import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.hf import hf_embed
from lightrag.utils import setup_logger, EmbeddingFunc, TokenTracker
from lightrag.rerank import cohere_rerank
from typing import List, Optional, Dict, Any
from functools import partial
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
load_dotenv()
# Pre-load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(str(os.getenv("EMBEDDING_PATH")))
embed_model = AutoModel.from_pretrained(str(os.getenv("EMBEDDING_PATH")))

tokentracker = TokenTracker()
async def my_rerank(query: str, documents: List[str], top_n: Optional[int] = None)-> List[Dict[str, Any]]:

    compressed_docs = []

    def _compress(doc, query):
        return doc

    for doc in documents:
        compressed = _compress(doc, query)  # ⭐你的算法
        compressed_docs.append(compressed)

    return [{"content": c} for c in compressed_docs[:2]]

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

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # print("promt: ", prompt)
    # print("sys prompt", system_prompt)
    # print("history", history_messages)
    result =  await openai_complete_if_cache(
        "qwen-plus",
        prompt,
        system_prompt=system_prompt,
        token_tracker=tokentracker,
        history_messages=history_messages,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        **kwargs
    )
    return result


setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=2048,
        model_name="BAAI/bga-large-zh",
        func=partial(
            hf_embed.func,  # 使用 .func 访问底层未封装的函数
            tokenizer=tokenizer,
            embed_model=embed_model
        )),
        llm_model_func=llm_model_func,
        rerank_model_func=my_rerank
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    return rag

async def main():
    try:
        # 初始化RAG实例
        rag = await initialize_rag()
        await rag.ainsert(texts)

        # 执行混合检索
        mode = "hybrid"
        tokentracker.reset()
        print(await rag.aquery(
                "What can you tell me about the Celtics? Answer in Chinese.",
                param=QueryParam(mode=mode, only_need_prompt=True)
            ))
        print(tokentracker.get_usage())
        

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())