import bm25s
import Stemmer
from datasets import load_dataset
import requests
import json
import re

# ---------------- 配置部分 ----------------
API_KEY = "sk-zwzhzgrwaqagcagxfyxhobouyycpuqpwkjbsfwbgxcevrueg" 
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
# ----------------------------------------

class RAGPipeline:
    def __init__(self):
        self.retriever = None
        self.corpus_texts = []
        self.corpus_ids = []
        self.stemmer = Stemmer.Stemmer("english")
        self.api_key = API_KEY

    def load_resources(self):
        """
        加载数据并构建索引。
        Streamlit 会缓存此函数结果，避免重复运行。
        """
        print("1. [Pipeline] Loading dataset HQ-small (Collection)...")
        dataset = load_dataset("izhx/COMP5423-25Fall-HQ-small", split="collection")
        self.corpus_texts = dataset['text']
        self.corpus_ids = dataset['id']

        print("2. [Pipeline] Building BM25 Index...")
        self.retriever = bm25s.BM25()
        tokenized_corpus = bm25s.tokenize(self.corpus_texts, stopwords="en", stemmer=self.stemmer)
        self.retriever.index(tokenized_corpus)
        print("3. [Pipeline] System Ready!")

    def retrieve(self, query, top_k=3):
        """执行实时检索"""
        tokenized_query = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer)
        results = self.retriever.retrieve(tokenized_query, k=top_k)
        
        doc_indices = results.documents[0]
        doc_scores = results.scores[0]
        
        retrieved_docs = []
        for idx, score in zip(doc_indices, doc_scores):
            # 兼容 app.py 可能用到 doc_id 或 id
            retrieved_docs.append({
                "id": self.corpus_ids[idx],
                "doc_id": self.corpus_ids[idx], 
                "text": self.corpus_texts[idx],
                "content": self.corpus_texts[idx], # 兼容不同命名习惯
                "score": float(score)
            })
        return retrieved_docs

    def _call_llm(self, messages, model_name="Qwen/Qwen2.5-7B-Instruct", temperature=0.1, max_tokens=512):
        """通用 LLM API 调用函数"""
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"Error: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

    def generate(self, query, context_docs, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """
        单次生成的标准方法（用于简单模式或 Final Answer）
        """
        context_str = ""
        for i, doc in enumerate(context_docs):
            context_str += f"[{i+1}] {doc['text']}\n"

        # 使用你要求的优化版 Prompt
        system_prompt = (
            "You are an expert in multi-hop Question Answering.\n"
            "Your task is to answer the question based **strictly on the provided context documents**.\n\n"
            "Reasoning Guidelines:\n"
            "1. **Bridge Entity**: This is a multi-hop task. You must identify the **bridge entity** or connection point between different documents to link the facts together.\n"
            "2. **Extremely Concise**: The final answer must be **EXTREMELY concise** (e.g., an entity name, a date, a place, or 'yes'/'no'). **Do not use full sentences**.\n\n"
            "Response Format Requirement:\n"
            "1. First, output a JSON object named `thought_process` containing a list of your reasoning steps (e.g., finding the bridge entity).\n"
            "2. Then, output 'Answer: <final answer>'.\n\n"
            "Example Output:\n"
            "```json\n"
            "{\"thought_steps\": [\"Identify the director of Inception (Bridge Entity).\", \"Find out which film directed by him won an Oscar based on the bridge entity.\"]}\n"
            "```\n"
            "Answer: The Film Name"
        )
        
        user_content = f"Context:\n{context_str}\n\nQuestion: {query}\n"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        return self._call_llm(messages, model_name=model_name)

    def run_with_logs(self, session_id, question, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """
        Agentic Workflow 核心入口。
        执行：分解 -> 分步检索 -> 最终汇总
        返回：(final_answer, all_docs, logs)
        """
        logs = []
        all_retrieved_docs = []
        known_facts = []

        # --- Step 1: 问题分解 (Decomposition) ---
        decompose_prompt = (
            "You are a helpful assistant that breaks down complex multi-hop questions.\n"
            "Input: A complex question.\n"
            "Output: A JSON list of simple sub-questions that need to be answered step-by-step to solve the original question.\n"
            "Example Input: Which film directed by the director of Inception won an Oscar?\n"
            "Example Output: [\"Who is the director of Inception?\", \"Which films directed by him won an Oscar?\"]\n\n"
            f"Question: {question}\n"
            "Output (JSON list only):"
        )
        
        decompose_response = self._call_llm([{"role": "user", "content": decompose_prompt}], model_name="Qwen/Qwen2.5-7B-Instruct") # 使用较强模型分解
        
        # 解析子问题
        sub_questions = [question] # 默认回退
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\[.*\]', decompose_response, re.DOTALL)
            if json_match:
                sub_questions = json.loads(json_match.group(0))
                # 记录 Log
                logs.append({
                    "type": "plan",
                    "step": 1,
                    "sub_questions": sub_questions,
                    "raw_response": decompose_response
                })
            else:
                logs.append({"type": "info", "content": "Complex decomposition failed, using original question."})
        except:
            logs.append({"type": "info", "content": "JSON parsing failed, using original question."})

        # --- Step 2: 分步执行 (Iterative Execution) ---
        current_context_docs = []
        
        for idx, sub_q in enumerate(sub_questions):
            # A. 检索
            docs = self.retrieve(sub_q, top_k=3)
            current_context_docs.extend(docs) # 累积上下文
            all_retrieved_docs.extend(docs)
            
            # 去重
            unique_docs = {d['id']: d for d in current_context_docs}.values()
            current_context_docs = list(unique_docs)
            
            # B. 中间回答 (Optional: 让模型基于当前文档回答子问题，增加解释性)
            # 为了速度，这里我们只做检索，最后统一回答；或者你可以选择每一步都回答。
            # 这里我们选择记录日志，展示检索结果
            logs.append({
                "type": "observe",
                "step": idx + 1,
                "sub_question": sub_q,
                "docs": docs  # 记录这一步检出的文档
            })

        # --- Step 3: 最终生成 (Final Synthesis) ---
        # 使用累积的所有文档进行最终回答
        final_response = self.generate(question, current_context_docs, model_name=model_name)
        
        # 尝试解析最终答案
        final_answer_text = final_response
        thought_process = {"error": "No thought process parsed"}
        
        try:
            if "Answer:" in final_response:
                final_answer_text = final_response.split("Answer:")[-1].strip()
            
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', final_response, re.DOTALL)
            if json_match:
                thought_process = json.loads(json_match.group(1))
        except:
            pass

        logs.append({
            "type": "final",
            "full_response": final_response,
            "parsed_answer": final_answer_text,
            "thought_process": thought_process
        })

        # 去重后的所有文档
        unique_all_docs = list({d['id']: d for d in all_retrieved_docs}.values())

        return final_answer_text, unique_all_docs, logs