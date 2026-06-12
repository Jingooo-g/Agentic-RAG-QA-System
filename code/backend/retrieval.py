import os
import sys

# Automatically configure Hugging Face domestic mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any
import numpy as np
import json
from tqdm import tqdm
import time
from datasets import load_dataset

# Libraries required for BM25
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import string

# Ensure punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class BGEVectorizer:
    def __init__(self, device: str = None, model_path: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize BGE vectorizer
        Args:
            device: Running device, defaults to automatic selection (cuda or cpu)
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading model: {self.model_path} (using HF mirror source)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
        except Exception as e:
            print(f"Model loading failed, please check network connection. Error: {e}")
            sys.exit(1)

        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"BGE model loading complete, using device: {self.device}")
        # BGE model may require instruction prefix
        self.instruction = "Generate representation for this sentence:"

    def get_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Convert list of texts to vectors
        """
        all_embeddings = []

        # Calculate total number of batches
        total_batches = (len(texts) + batch_size - 1) // batch_size

        if show_progress:
            # Create detailed progress bar
            pbar = tqdm(
                total=len(texts),
                desc="Generating text vectors",
                unit="text",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )

        start_time = time.time()

        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_start_time = time.time()

                # Add instruction prefix
                batch_with_instruction = [self.instruction + text for text in batch_texts]

                # Encode texts
                encoded_input = self.tokenizer(
                    batch_with_instruction,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                # Generate embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    # Use mean pooling to get sentence embeddings
                    embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                    # Normalize embeddings
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

                batch_time = time.time() - batch_start_time

                if show_progress:
                    # Update progress bar
                    pbar.update(len(batch_texts))
                    # Calculate estimated remaining time
                    elapsed_time = time.time() - start_time
                    processed_texts = min(i + batch_size, len(texts))
                    if processed_texts > 0:
                        texts_per_second = processed_texts / elapsed_time
                        remaining_texts = len(texts) - processed_texts
                        remaining_time = remaining_texts / texts_per_second if texts_per_second > 0 else 0

                        pbar.set_postfix({
                            "Batch": f"{i // batch_size + 1}/{total_batches}",
                            "Speed": f"{texts_per_second:.1f} texts/sec",
                            "Remaining Time": f"{remaining_time:.1f}s",
                            "Device": self.device
                        })

        finally:
            if show_progress:
                pbar.close()

        total_time = time.time() - start_time
        if show_progress:
            print(f"Vector generation complete! Total processed {len(texts)} texts, took {total_time:.2f} seconds")
            print(f"Average speed: {len(texts) / total_time:.2f} texts/sec")

        return np.vstack(all_embeddings)

    def mean_pooling(self, model_output, attention_mask):
        """
        Use mean pooling to get sentence embeddings from model output
        """
        token_embeddings = model_output[0]  # The first element contains embeddings for all tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Simplified single sentence encoding function
def encode_sentence_simple(sentence: str, model_path: str = "BAAI/bge-large-en-v1.5") -> np.ndarray:
    """
    Simplified single sentence encoding function
    """
    # For performance, it's better to call vectorizer.get_embeddings in a loop outside
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    instruction = "Generate representation for this sentence:"
    text_with_instruction = instruction + sentence
    encoded_input = tokenizer(text_with_instruction, padding=True, truncation=True, max_length=512,
                              return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        token_embeddings = model_output[0]
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                        min=1e-9)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()[0]


def process_jsonl_data(input_file: str, output_file: str, text_key: str = "text", id_key: str = "id"):
    """
    Process JSONL data, convert text to vectors (Used for Build Index)
    """
    # Initialize BGE vectorizer
    vectorizer = BGEVectorizer()

    # Reading input data (JSONL format)
    data = []
    texts = []
    ids = []

    print(f"Reading file for index building: {input_file}")
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
                print(f"JSON parsing error (Line {line_num}): {e}")
                continue

    print(f"Found {len(texts)} texts to process")

    # Generating embedding vectors
    embeddings = vectorizer.get_embeddings(texts, batch_size=64)  # Increased batch_size slightly

    # Building output data and writing to JSONL file
    print(f"Saving vector database to: {output_file}")
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

    print(f"Processing complete! Index built.")


from sklearn.metrics.pairwise import cosine_similarity


class VectorSearcher:
    def __init__(self, vector_db_path: str):
        """
        Initialize vector searcher
        """
        self.vector_db = self.load_vector_db(vector_db_path)
        self.vector_ids = list(self.vector_db.keys())
        self.vectors = np.array([self.vector_db[vid] for vid in self.vector_ids])
        print(f"Vector database loaded, total {len(self.vector_db)} vectors")

    def load_vector_db(self, vector_db_path: str) -> Dict[str, List[float]]:
        vector_db = {}
        if not os.path.exists(vector_db_path):
            print(f"Error: Vector database file not found {vector_db_path}")
            return {}

        with open(vector_db_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading vector database"):
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
        # Calculate cosine similarity
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        # Returns (id, score) tuple
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
    Evaluate retrieval performance (Used for Train and Validation)
    """
    # Initialize vector searcher and encoder
    searcher = VectorSearcher(vector_db_file)
    # For efficiency, BGEVectorizer should be instantiated once, not initialized every time
    # But to minimize major logic changes, we apply a small optimization here:
    vectorizer = BGEVectorizer(model_path=model_path if model_path else "BAAI/bge-large-en-v1.5")

    # Reading test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading query data"):
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                test_data.append(item)
            except json.JSONDecodeError:
                continue
    print(f"Successfully read {len(test_data)} query data points")

    # Batch generating query vectors (Optimization: Avoid generating one by one, too slow)
    queries = [item[text_key] for item in test_data if text_key in item]
    # Use vectorizer for batch generation
    query_embeddings = vectorizer.get_embeddings(queries, batch_size=32)

    # Searching and recording results
    results = []
    valid_idx = 0

    for i, item in enumerate(tqdm(test_data, desc="Retrieving")):
        if text_key not in item: continue

        query_vector = query_embeddings[valid_idx]
        valid_idx += 1

        # Search for the most similar vectors
        top_k_ids = searcher.search_similar_vectors(query_vector, top_k=top_k)

        # Record results (preserving original data answer and supporting_ids for subsequent training/evaluation)
        result_item = {
            "id": item.get(id_key, str(i)),
            "text": item[text_key],
            "answer": item.get(answer_key, ""),
            "retrieved_docs": top_k_ids  # Format is [[doc_id, score], ...]
        }
        # If supporting_ids are available, include them
        if supporting_ids_key in item:
            result_item["supporting_ids"] = item[supporting_ids_key]

        results.append(result_item)

    # Outputting results
    print(f"\n=== Retrieval complete ===")
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Results saved to JSONL format at: {output_file}")
    return


# Test evaluation function logic is the same, but excludes the answer field
def evaluate_retrieval_performance_test(test_file, vector_db_file, output_file, text_key="text", id_key="id", top_k=5,
                                        model_path=None):
    return evaluate_retrieval_performance(test_file, vector_db_file, output_file, text_key, id_key, None, None, top_k,
                                          model_path)


# Helper function: Auto download or check data
def get_or_download_data(data_dir, split_name):
    """
    Checks for local data, downloads from HuggingFace if missing
    """
    file_path = os.path.join(data_dir, f"{split_name}.jsonl")
    if os.path.exists(file_path):
        print(f"Local file detected: {file_path}")
        return file_path

    print(f"Local {split_name} split not found, downloading from HuggingFace (mirror)...")
    try:
        dataset = load_dataset("izhx/COMP5423-25Fall-HQ-small", split=split_name)
        # Save locally for subsequent direct reading
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        with open(file_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Downloaded and saved to: {file_path}")
        return file_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


# BM25 Retriever Class
class BM25Retriever:
    def __init__(self, local_file: str = "data/collection.jsonl"):
        """
        Initialize BM25 retriever (Prioritizes reading local collection.jsonl)
        """
        print(">>> [BM25] Initializing Index... (This might take a minute)")
        self.documents = []

        # 1. Attempting to read local file (Downloaded by get_or_download_data)
        if os.path.exists(local_file):
            print(f">>> [BM25] Loading from local file: {local_file}")
            with open(local_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "id" in item and ("text" in item or "content" in item):
                            self.documents.append({
                                "doc_id": item["id"],
                                "content": item.get("text", item.get("content"))
                            })
                    except:
                        continue
        else:
            # 2. If not local, fallback to HuggingFace loading
            print(">>> [BM25] Local file not found, downloading from HuggingFace...")
            dataset = load_dataset("izhx/COMP5423-25Fall-HQ-small", split="collection")
            for item in dataset:
                if "id" in item and "text" in item:
                    self.documents.append({
                        "doc_id": item["id"],
                        "content": item["text"]
                    })

        # 3. Building BM25 Index
        print(f">>> [BM25] Building Index for {len(self.documents)} documents...")
        self.tokenized_docs = self._preprocess_docs()
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(">>> [BM25] Ready!")

    def _preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        return [token for token in tokens if len(token) >= 2]

    def _preprocess_docs(self) -> List[List[str]]:
        # Displaying progress using tqdm
        return [self._preprocess_text(doc["content"]) for doc in tqdm(self.documents, desc="Tokenizing Docs")]

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        tokenized_query = self._preprocess_text(query)
        if not tokenized_query:
            return [{"doc_id": f"empty_{i}", "score": 0.0, "content": "No info"} for i in range(top_k)]

        doc_scores = self.bm25.get_scores(tokenized_query)
        top_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_doc_indices:
            doc = self.documents[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "id": doc["doc_id"],  # Compatibility field
                "score": round(float(doc_scores[idx]), 4),
                "content": doc["content"],
                "text": doc["content"]  # Compatibility field
            })

        return results
bm25_text_retriever = BM25Retriever()

# Main Execution Logic
if __name__ == "__main__":
    # Define common paths (Relative paths usable by everyone)
    DATA_DIR = "data"
    VECTOR_DB_FILE = os.path.join(DATA_DIR, "bge_vector_db.jsonl")

    # Prepare data (Train, Validation, Test, Collection)
    print(">>> Step 1/5: Checking and downloading data...")
    collection_file = get_or_download_data(DATA_DIR, "collection")
    train_file = get_or_download_data(DATA_DIR, "train")
    val_file = get_or_download_data(DATA_DIR, "validation")
    test_file = get_or_download_data(DATA_DIR, "test")

    # Building index (using Collection)
    # Only build index if file does not exist
    if not os.path.exists(VECTOR_DB_FILE):
        print("\n>>> Step 2/5: Building vector index (Build Index)...")
        process_jsonl_data(
            input_file=collection_file,
            output_file=VECTOR_DB_FILE
        )
    else:
        print(f"\n>>> Step 2/5: Vector index already exists ({VECTOR_DB_FILE}), skipping build.")

    # Retrieving Train Set (Generating training data)
    print("\n>>> Step 3/5: Retrieving Train Set (Generating training data)...")
    train_output = "train_retrieval.jsonl"
    evaluate_retrieval_performance(
        test_file=train_file,
        vector_db_file=VECTOR_DB_FILE,
        output_file=train_output,
        top_k=5  # TopK is usually 5 or 10 for training
    )

    # Retrieving Validation Set and scoring
    print("\n>>> Step 4/5: Retrieving Validation Set and scoring...")
    val_output = "val_retrieval.jsonl"
    evaluate_retrieval_performance(
        test_file=val_file,
        vector_db_file=VECTOR_DB_FILE,
        output_file=val_output,
        top_k=10
    )

    # Directly calling eval_retrieval.py for scoring
    if os.path.exists("eval_retrieval.py"):
        print(">>> Calling eval_retrieval.py to calculate scores...")
        # Using os.system to call evaluation script
        # Command format: python eval_retrieval.py --gold data/validation.jsonl --pred val_retrieval.jsonl
        cmd = f"python eval_retrieval.py --gold {val_file} --pred {val_output}"
        os.system(cmd)
    else:
        print(
            "Warning: eval_retrieval.py not found in current directory, skipping automatic scoring. Please run evaluation script manually.")

    # Retrieving Test Set (For submission)
    if test_file:
        print("\n>>> Step 5/5: Retrieving Test Set (For submission)...")
        test_output = "test_prediction.jsonl"
        evaluate_retrieval_performance_test(
            test_file=test_file,
            vector_db_file=VECTOR_DB_FILE,
            output_file=test_output,
            top_k=10
        )
        print(f"Submission file generated: {test_output}")
    else:
        print("\n>>> Test set not yet downloaded, skipping submission file generation.")

    print("\nAll tasks complete!")