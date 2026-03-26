from backend.baseline.rag_base import BaseRag
from backend.data_progress.base import QAItem
from tqdm import tqdm
from dotenv import load_dotenv
import os
import json
from typing import List, Optional, TypedDict
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from ragas import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from datetime import datetime
from vllm import LLM, SamplingParams

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall
)
from datasets import Dataset
from transformers import AutoTokenizer
load_dotenv()

class Ragas(TypedDict):
    question:List[str]
    answer:List[str]
    contexts:List[List[str]]
    ground_truth:List[str]

class Compress(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List["NodeWithScore"], query_bundle: Optional[QueryBundle] = None
    ) -> List["NodeWithScore"]:

        for node in nodes:
            node.node.text = node.node.text[:50]
        
        return nodes

class LlamaRag(BaseRag):
    llm:AutoModelForCausalLM
    embedding:HuggingFaceEmbedding
    use_compress:bool
    def __init__(self, use_compress:bool = False, tokenizer=None, model=None):
        super().__init__("llama")
        self.batch_size = 8
        self.top_k = 5
        self.max_context_len = 1200
        self.max_new_tokens = 256
        #定义模型
        model_path = str(os.getenv("LLM_PATH"))
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.tokenizer = tokenizer
        if not model:
            model = LLM(
                model=model_path,
                dtype="float16",
                gpu_memory_utilization=0.75,   # 🔥吃满显存
                max_model_len=2048
            )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens
        )        
        self.model = model
        #评测用模型
        # self.evalute_llm = HuggingFacePipeline(pipeline=pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, batch_size=self.batch_size, max_length=None))
        self.embedding = HuggingFaceEmbedding(
            model_name=str(os.getenv("EMBEDDING_PATH")), 
            device="cuda",
            embed_batch_size=self.batch_size,
            normalize=True
        )
        self.use_compress = use_compress  
        self.mycompress = Compress()

        #设置模型
        Settings.embed_model = self.embedding

    def __get_compress(self) -> str:
        return "use_compress" if self.use_compress else "no_compress"
    
    def load_data(self, datasets: List[QAItem], dataname:str):
        self.datasets = datasets
        print(len(self.datasets))
        self.dataname = dataname

    def measure(self, origin, compressed):
        return
        
    def compress(self, docs):
         return super().compress(docs)
    
    def build_prompt(self, context, query):
        return f"""
                Given these texts:
                -----
                {context}
                -----
                Please answer the following question:
                {query}
                """

    def generate_batch(self, prompts):
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(text)
        
        outputs = self.model.generate(
            formatted_prompts,
            self.sampling_params
        )

        responses = []
        prompt_tokens = []
        completion_tokens = []
        total_tokens = []

        for out in outputs:
            text = out.outputs[0].text
            responses.append(text)

            # 🔥 vLLM自带token统计
            pt = len(out.prompt_token_ids)
            ct = len(out.outputs[0].token_ids)
            tt = pt + ct

            prompt_tokens.append(pt)
            completion_tokens.append(ct)
            total_tokens.append(tt)

        return responses, prompt_tokens, completion_tokens, total_tokens
    
    def run(self):
        """
        进行测试
        """
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

        for i in tqdm(range(0, len(self.datasets), self.batch_size), desc="start query"):
            batch = self.datasets[i:i+self.batch_size]

            batch_prompts = []
            batch_docs = []
            batch_qas = []

            for qa in batch:
                documents = [Document(text=content) for content in qa["contents"]]
                #construct index vector
                index = VectorStoreIndex.from_documents(documents)
                if self.use_compress:
                    engine = index.as_retriever(similarity_top_k = self.top_k, node_postprocessors=[self.mycompress])
                else:
                    engine = index.as_retriever(similarity=self.top_k)
                
                docs = [data.text for data in engine.retrieve(QueryBundle(qa["query"]))]

                context_text = '\n'.join([d for d in docs])
                context_text = context_text[:self.max_context_len]

                prompt = self.build_prompt(context_text, qa["query"])

                batch_prompts.append(prompt)
                batch_docs.append(docs)
                batch_qas.append(qa)

            responses, ptks, ctks, ttks = self.generate_batch(batch_prompts)

            # 写入
            for qa, docs, resp, pt, ct, tt in zip(
                batch_qas, batch_docs, responses, ptks, ctks, ttks
            ):

                target["question"].append(qa["query"])
                target["contexts"].append([d for d in docs])
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
        root_path = str(os.getenv("ROOT_PATH"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{root_path}/backend/results/{self.name}/{self.__get_compress()}_{self.dataname}_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)
        print(f"结果已保存至{filepath}")
        return
    
    def llm_model_func(
        self,
        prompt,
        system_prompt=None,
        history_messages=[],
        keyword_extraction=False,
        **kwargs
    ):

        # 🌿 1. 构造 messages
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in history_messages:
            messages.append(msg)

        messages.append({"role": "user", "content": prompt})

        # 🌿 2. 构造输入文本（Qwen chat格式）
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 🌿 3. tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt"
        ).to(self.model.device)

        prompt_tokens = inputs["input_ids"].shape[-1]

        # 🌙 4. 推理
        with torch.no_grad():
            outputs = self.model.generate(
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
        result = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        total_tokens = prompt_tokens + completion_tokens

        return result, prompt_tokens, completion_tokens, total_tokens

    def run_demo(self, contexts:List[str], query:str, use_compress:bool):
        
        index = VectorStoreIndex.from_documents([Document(text=content) for content in contexts])

        if use_compress:
            engine = index.as_retriever(similarity_top_k = self.top_k, node_postprocessors=[self.mycompress])
        else:
            engine = index.as_retriever(similarity=self.top_k)
        
        docs = [data.text for data in engine.retrieve(QueryBundle(query))]

        prompt = self.build_prompt(docs, query)

        responses, ptks, ctks, ttks = self.llm_model_func(prompt)

        return {
            "answer":responses,
            "tokens":{
                "Prompt":ptks,
                "Completion":ctks,
                "Total":ttks
            }
        }


