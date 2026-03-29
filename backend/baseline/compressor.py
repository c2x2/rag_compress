import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class QueryAwareIterativeLLMLingua:
    def __init__(self, model_name="gpt2", device=None):
        """
        初始化迭代压缩器。
        为了快速跑通 demo，默认使用 gpt2。在实际 baseline 评测中，
        你可以替换为论文中提到的 LLaMA 或 Alpaca 等小模型。
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在加载小型语言模型: {model_name} 到 {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compress_context(self, query: str, context: str, keep_ratio: float = 0.5, segment_size: int = 100):
        """
        使用 ITPC 算法，结合 Query 对 Context 进行迭代压缩。
        
        :param query: 用户的查询问题
        :param context: RAG 检索到的长文本
        :param keep_ratio: 整体保留比例 (0 到 1 之间)
        :param segment_size: 每次迭代处理的 Token 块大小 (论文默认使用 100)
        :return: 压缩后的文本
        """
        # 1. 对 Query 和 Context 进行分词
        query_text = f"Question: {query}\nContext: "
        query_ids = self.tokenizer.encode(query_text, return_tensors="pt").to(self.device)
        context_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)[0]
        
        total_context_tokens = len(context_ids)
        if total_context_tokens == 0:
            return ""

        # 2. 将 Context 切分为多个 segment
        # 类似 S = {s_1, s_2, ..., s_m}
        segments = [context_ids[i : i + segment_size] for i in range(0, total_context_tokens, segment_size)]
        
        kept_context_ids = [] # 用于存放每一轮压缩后保留下来的 Token IDs
        
        # 3. 迭代处理每个 segment
        for segment in segments:
            # 核心 ITPC 逻辑：当前块的条件概率依赖于 Query + "前面所有已经压缩过的块"
            if len(kept_context_ids) > 0:
                history_tensor = torch.tensor(kept_context_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                prefix_ids = torch.cat([query_ids, history_tensor], dim=-1)
            else:
                prefix_ids = query_ids
                
            segment_tensor = segment.unsqueeze(0)
            
            # 将 prefix 和当前的 segment 拼接送入模型
            input_ids = torch.cat([prefix_ids, segment_tensor], dim=-1)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            
            # 计算当前 segment 的 Loss (交叉熵)
            # Logits 错位对齐：我们要用前一个 token 的预测来对齐当前 token 的真实标签
            segment_start_idx = prefix_ids.shape[1] - 1
            segment_logits = logits[0, segment_start_idx : -1, :]
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(segment_logits, segment).cpu().numpy()
            
            # 4. 根据设定的 keep_ratio 计算阈值，保留 Loss (困惑度) 最高的词
            token_log_probs = -token_losses  # CrossEntropy ≈ -log p

            sorted_log_probs = np.sort(token_log_probs)[::-1]

            keep_num = max(1, int(len(segment) * keep_ratio))
            gamma = sorted_log_probs[keep_num - 1] - 0.1
            kept_indices = np.where(token_log_probs >= gamma)[0]
            kept_indices = np.sort(kept_indices)
            
            # 将保留下来的 token 存入历史记录，供下一个 segment 评估时使用
            kept_tokens = segment[kept_indices].tolist()
            kept_context_ids.extend(kept_tokens)
            
        # 5. 解码最终保留的全部 Token
        # final_kept_tensor = torch.tensor(kept_context_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        # compressed_text = self.tokenizer.decode(final_kept_tensor[0], skip_special_tokens=True)
        print(kept_context_ids)
        compressed_text = self.safe_decode_long(
            kept_context_ids,
            max_len=1024
        )
        return compressed_text

    def safe_decode_long(self, token_ids, max_len=1024):
        chunks = []
        
        for i in range(0, len(token_ids), max_len):
            chunk_ids = token_ids[i:i+max_len]
            
            text = self.tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            chunks.append(text.strip())
        
        # 用换行而不是空格拼（更自然）
        return "\n".join(chunks)

# ==========================================
# 测试 Demo
# ==========================================
if __name__ == "__main__":

    # 模拟一个稍微长一点的英文数据集场景
    user_query = "Who was the man behind The Chipmunks?"
    contexts =[
                "The Chipmunks - Biography | Billboard\nThe Chipmunks\nAlvin Simon Theodore Ross Bagdasarian David Seville\nPossibly the most popular TV and musical cartoon of all time, the Chipmunks enjoyed several periods of prosperity -- beginning with the '60s era of adolescent Baby Boomers, cresting in the '80s, when the Boomers' children were growing up, and riding the wave clear into the new millennium.\nThe man who brought the Chipmunks to life, Ross Bagdasarian, was born on January 27, 1919, in Fresno, California. He came to Los Angeles in 1950, and appeared in the films Viva Zapata, Stalag 17, and Rear Window. Bagdasarian also worked as a songwriter, reaching the charts first in 1956, as his production of Alfi & Harry's \"The Trouble with Harry\" hit number 44. He later charted two solo singles (recorded as David Seville), \"Armen's Theme\" and \"Gotta Get to Your House.\" In 1958, Bagdasarian began experimenting with a novel technique -- recording normal vocals but then speeding up the playback on a tape machine. The process yielded the number one hit \"Witch Doctor\" in early 1958, and the phenomenon mushroomed later that year when his Christmas gimmick single \"The Chipmunk Song\" spent four weeks at the top of the charts. \"Alvin's Harmonica\" reached number three just two months later, and Christmas reissues of \"The Chipmunk Song\" charted in the Top 40 over the next four years. The Alvin Show premiered on prime-time television in 1961, with all voices supplied by Bagdasarian. It only ran for one year, but was a success in a Saturday-morning slot. Five more Chipmunks singles charted in the early '60s, and five LPs also did well, including a Beatles cover album in 1964.\nAlthough Bagdasarian died in 1972, his son Ross Jr. revived Alvin, Simon, and Theodore in 1979 on Saturday mornings and on the 1980 album Chipmunk Punk. The series became more popular than in the '60s, and albums of the Chipmunks singing country, Christmas, rock, and Hollywood favorites were big sellers, though they didn't enjoy chart success. Although the cartoon was no longer in production by the '90s, new Chipmunks records continued appearing, among them 1998's A-Files: Alien Files.\nIn 2007, a film series debuted with Alvin and the Chipmunks -- the first being so successful that it spawned three sequels: 2009's Alvin and the Chipmunks: The Squeakquel, 2011's Alvin and the Chipmunks: Chipwrecked, and 2015's Alvin and the Chipmunks 4. A revival of the TV series was also planned to premiere on Nickelodeon in early 2015. ~ John Bush, Rovi\nRelated Artists",
                "Alvin and the Chipmunks (2007) - IMDb\nIMDb\n17 January 2017 4:34 PM, UTC\nNEWS\nThere was an error trying to load your rating for this title.\nSome parts of this page won't work property. Please reload or try later.\nX Beta I'm Watching This!\nKeep track of everything you watch; tell your friends.\nError\nAlvin and the Chipmunks\u00a0( 2007 )\nPG |\nA struggling songwriter named Dave Seville finds success when he comes across a trio of singing chipmunks: mischievous leader Alvin, brainy Simon, and chubby, impressionable Theodore.\nDirector:\nFrom $2.99 (SD) on Amazon Video\nON\u00a0TV\nUser Lists\nRelated lists from IMDb users\na list of 43 titles\ncreated 01\u00a0Apr\u00a02012\na list of 38 titles\ncreated 09\u00a0May\u00a02012\na list of 48 titles\ncreated 20\u00a0Oct\u00a02012\na list of 23 titles\ncreated 06\u00a0Mar\u00a02013\na list of 42 titles\ncreated 30\u00a0Dec\u00a02015\nTitle: Alvin and the Chipmunks (2007)\n5.2/10\nWant to share IMDb's rating on your own site? Use the HTML below.\nYou must be a registered user to use the IMDb rating plugin.\n2 wins & 2 nominations. See more awards \u00a0\u00bb\nVideos\nThe world famous singing pre-teen chipmunk trio return to contend with the pressures of school, celebrity, and a rival female music group known as The Chipettes.\nDirector: Betty Thomas\nPlaying around while aboard a cruise ship, the Chipmunks and Chipettes accidentally go overboard and end up marooned in a tropical paradise. They discover their new turf is not as deserted as it seems.\nDirector: Mike Mitchell\n\u00a0 \u00a0 1 2 3 4 5 6 7 8 9 10 5.1/10 X \u00a0\nThrough a series of misunderstandings, Alvin, Simon and Theodore come to believe that Dave is going to propose to his new girlfriend in Miami...and dump them. They have three days to get to him and stop the proposal, saving themselves not only from losing Dave but possibly from gaining a terrible stepbrother.\nDirector: Walt Becker\nWhen the evil wizard Gargamel chases the tiny blue Smurfs out of their village, they tumble from their magical world into New York City.\nDirector: Raja Gosnell\nJon Arbuckle buys a second pet, a dog named Odie. However, Odie is then abducted and it is up to Jon's cat, Garfield, to find and rescue the canine.\nDirector: Peter Hewitt\nJon and Garfield visit the United Kingdom, where a case of mistaken cat identity finds Garfield ruling over a castle. His reign is soon jeopardized by the nefarious Lord Dargis, who has designs on the estate.\nDirector: Tim Hill\nThe Smurfs team up with their human friends to rescue Smurfette, who has been kidnapped by Gargamel since she knows a secret spell that can turn the evil sorcerer's newest creation - creatures called the Naughties - into real Smurfs.\nDirector: Raja Gosnell\nStuart and Snowbell set out across town to rescue a friend.\nDirector: Rob Minkoff\nThe Little family adopt a charming young mouse named Stuart, but the family cat wants rid of him.\nDirector: Rob Minkoff\n\u00a0 \u00a0 1 2 3 4 5 6 7 8 9 10 6.2/10 X \u00a0\nBoog, a domesticated 900lb. Grizzly bear, finds himself stranded in the woods 3 days before Open Season. Forced to rely on Elliot, a fast-talking mule deer, the two form an unlikely friendship and must quickly rally other forest animals if they are to form a rag-tag army against the hunters.\nDirectors: Roger Allers, Jill Culton, and 1 more credit \u00a0\u00bb\nStars: Ashton Kutcher,  Martin Lawrence,  Debra Messing\n\u00a0 \u00a0 1 2 3 4 5 6 7 8 9 10 6.2/10 X \u00a0\nBarry B. Benson, a bee just graduated from college, is disillusioned at his lone career choice: making honey. On a special trip outside the hive, Barry's life is saved by Vanessa, a florist in New York City. As their relationship blossoms, he discovers humans actually eat honey, and subsequently decides to sue them."
            ],
    # rag_context = """
    # Large language models (LLMs) have shown astonishing capabilities in various applications. 
    # However, the prompts fed to these models are becoming increasingly lengthy, which significantly 
    # increases inference costs. To address this, many researchers are looking into prompt compression.
    # A simple method is to remove tokens based on perplexity independently. However, this ignores the 
    # interdependence between compressed contents. If we delete a word, the context for the following 
    # words changes drastically.
    # Therefore, the main advantage of the proposed iterative algorithm is that it can better model the 
    # interdependence between compressed contents. By dividing the text into chunks and continuously 
    # updating the prefix with previously compressed results, the algorithm dynamically preserves the 
    # most crucial semantic information step by step.
    # """
    
    rag_context = "\n".join(str(context) for context in contexts)
    compressor = QueryAwareIterativeLLMLingua(model_name="/home/melonmelon/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")

    print("\n--- 正在执行 Iterative Token-level Prompt Compression (ITPC) ---")
    # 按照论文配置，采用分段压缩
    compressed_result = compressor.compress_context(
        query=user_query, 
        context=rag_context, 
        keep_ratio=0.45, 
        segment_size=100 # 为了在短文本中演示分段效果，这里人为把分段设小。长文本建议用论文默认的 100。
    )
    
    print("\n[原始 Context]:")
    print(rag_context.strip())
    print("-" * 40)
    print("\n[ITPC 压缩后的 Context]:")
    print(compressed_result)