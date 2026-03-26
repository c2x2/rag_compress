import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import setup_logger, EmbeddingFunc 
from lightrag.rerank import cohere_rerank
from typing import List, Optional, Dict, Any
from functools import partial
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from dotenv import load_dotenv
import torch
load_dotenv()
# Pre-load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(str(os.getenv("EMBEDDING_PATH")))
embed_model = AutoModel.from_pretrained(str(os.getenv("EMBEDDING_PATH")))
model_path = str(os.getenv("LLM_PATH"))
llm_tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="cuda")


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

# async def llm_model_func(
#     prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
# ) -> str:
#     # print("promt: ", prompt)
#     # print("sys prompt", system_prompt)
#     # print("history", history_messages)
#     result =  await openai_complete_if_cache(
#         "qwen-plus",
#         prompt,
#         system_prompt=system_prompt,
#         token_tracker=tokentracker,
#         history_messages=history_messages,
#         api_key=os.getenv("LLM_API_KEY"),
#         base_url=os.getenv("LLM_BASE_URL"),
#         **kwargs
#     )
#     return result


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    **kwargs
) -> str:

    # 🌿 1. 构造 messages
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in history_messages:
        messages.append(msg)

    messages.append({"role": "user", "content": prompt})

    # 🌿 2. 构造输入文本（Qwen chat格式）
    input_text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 🌿 3. tokenize
    inputs = llm_tokenizer(
        input_text,
        return_tensors="pt"
    ).to(model.device)

    prompt_tokens = inputs["input_ids"].shape[-1]

    # 🌙 4. 推理
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # 🌿 5. 计算生成 tokens
    generated_tokens = outputs[0][prompt_tokens:]
    completion_tokens = generated_tokens.shape[-1]

    # 🌿 6. decode
    result = llm_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    total_tokens = prompt_tokens + completion_tokens

    # 🌟 7. 打印 or 记录
    print(f"📊 Token Usage:")
    print(f"  prompt_tokens     = {prompt_tokens}")
    print(f"  completion_tokens = {completion_tokens}")
    print(f"  total_tokens      = {total_tokens}")

    return result


# async def main():
#     result = await llm_model_func("你是谁?")

#     print(result)




# setup_logger("lightrag", level="INFO")

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
        print(await rag.aquery(
                "What can you tell me about the Celtics? Answer in Chinese.",
                param=QueryParam(mode=mode, only_need_prompt=False)
            ))


    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    asyncio.run(main())