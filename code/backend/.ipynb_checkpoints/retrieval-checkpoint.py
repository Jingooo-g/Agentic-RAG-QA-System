import os
import sys

# ==============================================================================
# 1. 【新增】自动配置 Hugging Face 国内镜像
# ==============================================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any
import numpy as np
import json
from tqdm import tqdm
import time
# 【新增】用于自动下载数据
from datasets import load_dataset 

class BGEVectorizer:
    # 【修改】默认路径改为 Hugging Face 模型 ID，这样所有人都能自动下载
    def __init__(self, device: str = None, model_path: str = "BAAI/bge-large-en-v1.5"):
        """
        初始化BGE向量化器
        Args:
            device: 运行设备，默认为自动选择 (cuda 或 cpu)
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"正在加载模型: {self.model_path} (使用 HF 镜像源)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
        except Exception as e:
            print(f"模型加载失败，请检查网络。错误: {e}")
            sys.exit(1)
            
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"BGE模型加载完成，使用设备: {self.device}")
        # BGE模型可能需要指令前缀
        self.instruction = "为这个句子生成表示："

    def get_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        将文本列表转换为向量
        """
        all_embeddings = []

        # 计算总批次数
        total_batches = (len(texts) + batch_size - 1) // batch_size

        if show_progress:
            # 创建详细的进度条
            pbar = tqdm(
                total=len(texts),
                desc="生成文本向量",
                unit="text",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )

        start_time = time.time()

        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_start_time = time.time()

                # 添加指令前缀
                batch_with_instruction = [self.instruction + text for text in batch_texts]

                # 编码文本
                encoded_input = self.tokenizer(
                    batch_with_instruction,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                # 生成嵌入
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    # 使用平均池化获取句子嵌入
                    embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                    # 归一化嵌入
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

                batch_time = time.time() - batch_start_time

                if show_progress:
                    # 更新进度条
                    pbar.update(len(batch_texts))
                    # 计算预估剩余时间
                    elapsed_time = time.time() - start_time
                    processed_texts = min(i + batch_size, len(texts))
                    if processed_texts > 0:
                        texts_per_second = processed_texts / elapsed_time
                        remaining_texts = len(texts) - processed_texts
                        remaining_time = remaining_texts / texts_per_second if texts_per_second > 0 else 0

                        pbar.set_postfix({
                            "批次": f"{i // batch_size + 1}/{total_batches}",
                            "速度": f"{texts_per_second:.1f}文本/秒",
                            "剩余时间": f"{remaining_time:.1f}秒",
                            "设备": self.device
                        })

        finally:
            if show_progress:
                pbar.close()

        total_time = time.time() - start_time
        if show_progress:
            print(f"向量生成完成！总共处理 {len(texts)} 个文本，耗时 {total_time:.2f} 秒")
            print(f"平均速度: {len(texts) / total_time:.2f} 文本/秒")

        return np.vstack(all_embeddings)

    def mean_pooling(self, model_output, attention_mask):
        """
        使用平均池化从模型输出中获取句子嵌入
        """
        token_embeddings = model_output[0]  # 第一个元素包含所有token的嵌入
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# 【修改】稍微修改了这个函数，让他使用类里面的方法，避免逻辑重复
# 但为了保持兼容性，还是保留了独立函数的形式
def encode_sentence_simple(sentence: str, model_path: str = "BAAI/bge-large-en-v1.5") -> np.ndarray:
    """
    简化版的单句编码函数
    """
    # 这里为了简单，直接实例化一个临时的 vectorizer，或者你可以复用全局变量
    # 为了性能，建议在外部循环调用 vectorizer.get_embeddings
    # 但为了不改动你队友下面的逻辑，这里保持原样，只是路径变了
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    
    instruction = "为这个句子生成表示："
    text_with_instruction = instruction + sentence
    encoded_input = tokenizer(text_with_instruction, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        token_embeddings = model_output[0]
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()[0]


def process_jsonl_data(input_file: str, output_file: str, text_key: str = "text", id_key: str = "id"):
    """
    处理JSONL数据，将文本转换为向量 (Build Index 使用)
    """
    # 初始化BGE向量化器
    vectorizer = BGEVectorizer()

    # 读取输入数据（JSONL格式）
    data = []
    texts = []
    ids = []

    print(f"正在读取文件用于构建索引: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
                if text_key in item and id_key in item:
                    texts.append(item[text_key])
                    ids.append(item[id_key])
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 (第 {line_num} 行): {e}")
                continue

    print(f"找到 {len(texts)} 个文本需要处理")

    # 生成嵌入向量
    embeddings = vectorizer.get_embeddings(texts, batch_size=64) # 稍微调大了 batch_size

    # 构建输出数据并写入JSONL文件
    print(f"正在保存向量库到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (item_id, text, embedding) in enumerate(zip(ids, texts, embeddings)):
            output_item = {
                id_key: item_id,
                text_key: text,
                "embedding": embedding.tolist(),  
                "embedding_dim": len(embedding)
            }
            json_line = json.dumps(output_item, ensure_ascii=False)
            f.write(json_line)
            f.write('\n')

    print(f"处理完成！索引已构建。")


from sklearn.metrics.pairwise import cosine_similarity


class VectorSearcher:
    def __init__(self, vector_db_path: str):
        """
        初始化向量搜索器
        """
        self.vector_db = self.load_vector_db(vector_db_path)
        self.vector_ids = list(self.vector_db.keys())
        self.vectors = np.array([self.vector_db[vid] for vid in self.vector_ids])
        print(f"向量数据库加载完成，共 {len(self.vector_db)} 个向量")

    def load_vector_db(self, vector_db_path: str) -> Dict[str, List[float]]:
        vector_db = {}
        if not os.path.exists(vector_db_path):
             print(f"错误：找不到向量库文件 {vector_db_path}")
             return {}
             
        with open(vector_db_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载向量库"):
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    if "id" in item and "embedding" in item:
                        vector_db[item["id"]] = item["embedding"]
                except json.JSONDecodeError:
                    continue
        return vector_db

    def search_similar_vectors(self, query_vector: np.ndarray, top_k: int = 5) -> List[Any]:
        # 计算余弦相似度
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        # 返回(id, 分数)元组
        return [(self.vector_ids[i], float(similarities[i])) for i in top_indices]


def evaluate_retrieval_performance(
        test_file: str,
        vector_db_file: str,
        output_file: str = None,
        text_key: str = "text",
        id_key: str = "id",
        answer_key: str = "answer",
        supporting_ids_key: str = "supporting_ids",
        top_k: int = 5,
        model_path: str = None
):
    """
    评估检索性能 (用于 Train 和 Validation)
    """
    # 初始化向量搜索器和编码器
    searcher = VectorSearcher(vector_db_file)
    # 这里为了效率，应该实例化一次 BGEVectorizer，而不是每次 encode 都 new 一个
    # 但为了不改动太多逻辑，这里稍微优化一下：
    vectorizer = BGEVectorizer(model_path=model_path if model_path else "BAAI/bge-large-en-v1.5")

    # 读取测试数据
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取查询数据"):
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                test_data.append(item)
            except json.JSONDecodeError:
                continue
    print(f"成功读取 {len(test_data)} 条查询数据")

    # 批量生成查询向量 (优化点：不要逐个生成，太慢了)
    queries = [item[text_key] for item in test_data if text_key in item]
    # 使用 vectorizer 批量生成
    query_embeddings = vectorizer.get_embeddings(queries, batch_size=32)

    # 搜索并记录结果
    results = []
    valid_idx = 0
    
    for i, item in enumerate(tqdm(test_data, desc="检索中")):
        if text_key not in item: continue
        
        query_vector = query_embeddings[valid_idx]
        valid_idx += 1
        
        # 搜索最相似的向量
        top_k_ids = searcher.search_similar_vectors(query_vector, top_k=top_k)

        # 记录结果 (保留原始数据中的 answer 和 supporting_ids 以便后续训练/评估)
        result_item = {
            "id": item.get(id_key, str(i)),
            "text": item[text_key],
            "answer": item.get(answer_key, ""),
            "retrieved_docs": top_k_ids  # 格式为 [[doc_id, score], ...]
        }
        # 如果有 supporting_ids，也带上
        if supporting_ids_key in item:
            result_item["supporting_ids"] = item[supporting_ids_key]
            
        results.append(result_item)

    # 输出结果
    print(f"\n=== 检索完成 ===")
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"结果已保存为JSONL格式到: {output_file}")
    return 

# Test 的评估函数逻辑其实和上面一样，只是不需要 answer 字段
# 但为了保持你队友的代码结构，我保留它，只是把逻辑替换成上面的优化版本
def evaluate_retrieval_performance_test(test_file, vector_db_file, output_file, text_key="text", id_key="id", top_k=5, model_path=None):
    return evaluate_retrieval_performance(test_file, vector_db_file, output_file, text_key, id_key, None, None, top_k, model_path)


# ==============================================================================
# 【新增】辅助函数：自动下载或检查数据
# ==============================================================================
def get_or_download_data(data_dir, split_name):
    """
    检查本地是否有数据，没有则从 HuggingFace 下载
    """
    file_path = os.path.join(data_dir, f"{split_name}.jsonl")
    if os.path.exists(file_path):
        print(f"检测到本地文件: {file_path}")
        return file_path
    
    print(f"本地未找到 {split_name} 集，正在从 HuggingFace (镜像) 下载...")
    try:
        dataset = load_dataset("izhx/COMP5423-25Fall-HQ-small", split=split_name)
        # 存到本地，方便后续直接读取
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"已下载并保存至: {file_path}")
        return file_path
    except Exception as e:
        print(f"下载失败: {e}")
        return None

# ==============================================================================
# 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # 1. 定义通用路径 (大家都能用的相对路径)
    DATA_DIR = "data"
    VECTOR_DB_FILE = os.path.join(DATA_DIR, "bge_vector_db.jsonl")
    
    # 2. 准备数据 (Train, Validation, Test, Collection)
    print(">>> 步骤 1/5: 检查并下载数据...")
    collection_file = get_or_download_data(DATA_DIR, "collection")
    train_file = get_or_download_data(DATA_DIR, "train")
    val_file = get_or_download_data(DATA_DIR, "validation")
    test_file = get_or_download_data(DATA_DIR, "test") # 如果test还没发布，这里可能会报错，不过没关系

    # 3. 构建索引 (使用 Collection)
    # 只有当索引文件不存在时才构建
    if not os.path.exists(VECTOR_DB_FILE):
        print("\n>>> 步骤 2/5: 构建向量索引 (Build Index)...")
        process_jsonl_data(
            input_file=collection_file, 
            output_file=VECTOR_DB_FILE
        )
    else:
        print(f"\n>>> 步骤 2/5: 向量索引已存在 ({VECTOR_DB_FILE})，跳过构建。")

    # 4. 检索 Train 集 (用于训练生成模型)
    print("\n>>> 步骤 3/5: 检索 Train 集 (生成训练数据)...")
    train_output = "train_retrieval.jsonl"
    evaluate_retrieval_performance(
        test_file=train_file,
        vector_db_file=VECTOR_DB_FILE,
        output_file=train_output,
        top_k=5 # 训练时通常取Top5或Top10
    )

    # 5. 检索 Validation 集并评估
    print("\n>>> 步骤 4/5: 检索 Validation 集并算分...")
    val_output = "val_retrieval.jsonl"
    evaluate_retrieval_performance(
        test_file=val_file,
        vector_db_file=VECTOR_DB_FILE,
        output_file=val_output,
        top_k=10
    )
    
    # 【核心需求】直接在代码里调用 eval_retrieval.py 算分
    if os.path.exists("eval_retrieval.py"):
        print(">>> 正在调用 eval_retrieval.py 计算分数...")
        # 使用 os.system 调用评估脚本
        # 命令格式: python eval_retrieval.py --gold data/validation.jsonl --pred val_retrieval.jsonl
        cmd = f"python eval_retrieval.py --gold {val_file} --pred {val_output}"
        os.system(cmd)
    else:
        print("警告: 当前目录下未找到 eval_retrieval.py，无法自动算分。请手动运行评估脚本。")

    # 6. 检索 Test 集 (用于提交)
    if test_file:
        print("\n>>> 步骤 5/5: 检索 Test 集 (生成提交文件)...")
        test_output = "test_prediction.jsonl"
        evaluate_retrieval_performance_test(
            test_file=test_file,
            vector_db_file=VECTOR_DB_FILE,
            output_file=test_output,
            top_k=10
        )
        print(f"提交文件已生成: {test_output}")
    else:
        print("\n>>> Test 集暂未下载，跳过生成提交文件。")

    print("\n所有任务完成！")