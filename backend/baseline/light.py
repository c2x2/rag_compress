from backend.baseline.rag_base import BaseRag
from backend.data_progress.base import QAItem
from typing import List, TypedDict
import os
from tqdm import tqdm
from dotenv import load_dotenv
from functools import partial
import asyncio
import json
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
import torch
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc

load_dotenv()

class Ragas(TypedDict):
    question:List[str]
    answer:List[str]
    contexts:List[List[str]]
    ground_truth:List[str]

class LightRag(BaseRag):
    def __init__(self, use_compress:bool=False):
        super().__init__("LightRag")
        self.batch_size = 8
        self.top_k = 5
        self.max_context_len = 1200
        self.max_new_tokens = 256
        self.root_path = str(os.getenv("ROOT_PATH"))

        #定义模型
        model_path = str(os.getenv("LLM_PATH"))
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer = tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="cuda")
        self.model = model
        model.config.use_cache = False

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(str(os.getenv("EMBEDDING_PATH")))
        self.embedding = AutoModel.from_pretrained(str(os.getenv("EMBEDDING_PATH")))

        #评测用模型
        self.evalute_llm = HuggingFacePipeline(
            pipeline=pipeline(
                task="text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                max_new_tokens=self.max_new_tokens, 
                batch_size=self.batch_size, 
                max_length=None, 
                return_full_text=False,
                temperature=0,
                do_sample=False
            )
        )
        self.evalute_embedding = HuggingFaceEmbeddings(
            model_name=str(os.getenv("EMBEDDING_PATH")), 
            model_kwargs={"device": "cuda"}, 
            encode_kwargs={
                "batch_size": self.batch_size,   # 🔥 核心优化
                "normalize_embeddings": True
            }
        )
        self.use_compress = use_compress

    async def llm_model_func(
        self,
        prompt,
        system_prompt=None,
        history_messages=[],
        keyword_extraction=False,
        **kwargs
    ) -> str:
        
        # 🌿 1. 构造完整输入（模仿 chat 格式）
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in history_messages:
            messages.append(msg)

        messages.append({"role": "user", "content": prompt})

        # 🌿 2. 转成模型输入格式（Qwen推荐方式）
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt"
        ).to(self.model.device)

        # 🌙 3. 推理（生成）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        # 🌿 4. 解码（只取新生成部分）
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return result.strip()

    #rag系统
    def init_lightrag(self, work_dir):
        rag = LightRAG(
            working_dir=work_dir,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=2048,
                model_name="local",
                func=partial(
                    hf_embed.func,
                    tokenizer=self.embedding_tokenizer,
                    embed_model=self.embedding
                )
            ),
            llm_model_func=self.llm_model_func,
            rerank_model_func=None
        )
        return rag

    def __get_compress(self) -> str:
        return "use_compress" if self.use_compress else "no_compress"

    def load_data(self, datasets: List[QAItem], dataname:str):
        self.datasets = datasets
        self.dataname = dataname

    def measure(self, origin, compressed):
        return
    
    def compress(self, docs:List[str]) -> List[str]:
        compressed_docs = []
        for doc in docs:
            compressed_docs.append(doc[:50])
        
        return compressed_docs
    
    async def build_prompts_batch(self, queries: List[str]):
        tasks = [
            self.rag.aquery(
                q,
                param=QueryParam(mode="hybrid", only_need_prompt=True)
            )
            for q in queries
        ]
        prompts = await asyncio.gather(*tasks)
        return prompts

    def generate_batch(self, prompts):
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(text)

        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_context_len,
            return_tensors="pt"
        ).to("cuda")

        # 记录输入张量的宽度（这是所有 prompt 补齐后的统一长度）
        input_tensor_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                # temperature=0,
                # top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

        responses = []
        prompt_tokens_counts = []
        completion_tokens_counts = []
        total_tokens_counts = []

        for i in range(len(outputs)):
            # 1. 获取该行真正的 prompt 长度（不含 padding）
            actual_prompt_len = inputs["attention_mask"][i].sum().item()
            
            # 2. 正确切片：从 input_tensor_len 之后开始才是真正的生成内容
            # 无论前面有多少 PAD，生成的回答始终跟在整个输入张量后面
            generated_tokens = outputs[i][input_tensor_len:]
            
            # 3. 过滤掉生成结果末尾可能的填充 token (如果有的话)
            # 寻找第一个 EOS token 之后的位置并截断，或者直接用 skip_special_tokens
            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 4. 计算真实的 Token 数量
            # 生成的有效 token 数需要排除掉生成阶段产生的 [PAD]
            # 我们通过解码后再编码，或者直接统计非 pad 的 token
            actual_gen_tokens = (generated_tokens != self.tokenizer.pad_token_id).sum().item()

            responses.append(text)
            prompt_tokens_counts.append(actual_prompt_len)
            completion_tokens_counts.append(actual_gen_tokens)
            total_tokens_counts.append(actual_prompt_len + actual_gen_tokens)

        return responses, prompt_tokens_counts, completion_tokens_counts, total_tokens_counts

    def run(self):
        print("============开始运行rag系统============")

        target: Ragas = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        token_stats = {
            "prompt_tokens": [],
            "completion_tokens": [],
            "total_tokens": []
        }

        for i in tqdm(range(0, len(self.datasets), self.batch_size)):
            batch = self.datasets[i:i+self.batch_size]

            queries = [qa["query"] for qa in batch]
                # 🌿 1. 创建独立 working_dir
            working_dir = f"./data/batch_{i}"

            # 🌙 2. 删除旧数据（防止重复运行污染）
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir)

            self.rag = self.init_lightrag(working_dir)
            asyncio.run(self.rag.initialize_storages())

            texts = []
            for qa in batch:
                texts.extend(qa["contents"])

            asyncio.run(self.rag.ainsert(texts))

            # 🔥 1. 用 LightRAG 生成 prompt
            prompts = asyncio.run(self.build_prompts_batch(queries))
            # 🔥 2. batch 推理
            responses, ptks, ctks, ttks = self.generate_batch(prompts)

            # 🔥 3. 写入
            for qa, resp, pt, ct, tt in zip(batch, responses, ptks, ctks, ttks):
                target["question"].append(qa["query"])
                target["contexts"].append([])  # ⚠️ LightRAG 内部处理了
                target["ground_truth"].append(qa["answer"])
                target["answer"].append(resp)

                token_stats["prompt_tokens"].append(int(pt))
                token_stats["completion_tokens"].append(int(ct))
                token_stats["total_tokens"].append(int(tt))

            torch.cuda.empty_cache()

        final_output = {
            "data": target,
            "tokens": token_stats
        }

        filepath = f"{self.root_path}/results/{self.name}/{self.__get_compress()}_{self.dataname}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)

        print(f"结果已保存至{filepath}")
        return final_output


    async def run_demo(self, contexts, query):
        pass