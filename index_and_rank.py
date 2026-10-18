import os
import time
import joblib
import numpy as np
import pandas as pd
import faiss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

nltk.download("stopwords", quiet=True)


class QueryPreprocessor:
    """Applies identical tokenization and stemming to incoming search queries."""

    def __init__(self):
        self.tokenizer = RegexpTokenizer(r"\b[a-zA-Z]{2,}\b")
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update({
            "enron", "ect", "hou", "corp", "com", "http", "www",
            "mailto", "subject", "cc", "bcc", "re", "fw", "fwd"
        })
        self.stemmer = PorterStemmer()

    def preprocess(self, query: str) -> list[str]:
        tokens = self.tokenizer.tokenize(query.lower())
        return [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words
        ]


class EnronSearchEngine:
    """
    Enterprise Email Search Engine supporting:
    - Multi-field lexical indexing (Subject, Body, Combined)
    - Thread-level search aggregation
    - Ranking Algorithms: TF-IDF, BM25, Sentence Transformer
    - Hybrid Model: Reciprocal Rank Fusion (RRF)
    """

    def __init__(
        self,
        parquet_path: str = "enron_preprocessed.parquet",
        index_dir: str = "enron_indices",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.parquet_path = parquet_path
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model_name
        self.query_processor = QueryPreprocessor()

        self.df = None
        self.tfidf_vectorizers = {}
        self.tfidf_matrices = {}
        self.bm25_models = {}
        self.semantic_model = None
        self.faiss_index = None

    def build_or_load_indices(self, force_rebuild: bool = False):
        """Loads existing indices from disk or builds them from scratch."""
        os.makedirs(self.index_dir, exist_ok=True)

        print(f"Loading preprocessed corpus from {self.parquet_path}...")
        self.df = pd.read_parquet(self.parquet_path)
        # Fill any potential NaNs with empty strings
        for col in ["stemmed_combined", "stemmed_subject", "stemmed_body", "semantic_text"]:
            self.df[col] = self.df[col].fillna("")

        tfidf_path = os.path.join(self.index_dir, "tfidf_bundle.joblib")
        bm25_path = os.path.join(self.index_dir, "bm25_bundle.joblib")
        faiss_path = os.path.join(self.index_dir, "semantic_faiss.index")

        # 1. Load or Build TF-IDF Indices (Combined, Subject, Body)
        if not force_rebuild and os.path.exists(tfidf_path):
            print("Loading cached TF-IDF indices...")
            bundle = joblib.load(tfidf_path)
            self.tfidf_vectorizers = bundle["vectorizers"]
            self.tfidf_matrices = bundle["matrices"]
        else:
            print("Building TF-IDF indices (combined, subject, body)...")
            field_map = {
                "combined": "stemmed_combined",
                "subject": "stemmed_subject",
                "body": "stemmed_body"
            }
            for scope, col in field_map.items():
                vec = TfidfVectorizer(max_features=100_000, ngram_range=(1, 2), sublinear_tf=True)
                mat = vec.fit_transform(self.df[col])
                self.tfidf_vectorizers[scope] = vec
                self.tfidf_matrices[scope] = mat
            joblib.dump(
                {"vectorizers": self.tfidf_vectorizers, "matrices": self.tfidf_matrices},
                tfidf_path
            )

        # 2. Load or Build BM25 Index
        if not force_rebuild and os.path.exists(bm25_path):
            print("Loading cached BM25 indices...")
            self.bm25_models = joblib.load(bm25_path)
        else:
            print("Building BM25 indices (combined, subject, body)...")
            for scope, col in [
                ("combined", "stemmed_combined"),
                ("subject", "stemmed_subject"),
                ("body", "stemmed_body")
            ]:
                tokenized_corpus = [doc.split() for doc in self.df[col].tolist()]
                self.bm25_models[scope] = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
            joblib.dump(self.bm25_models, bm25_path)

        # 3. Load Sentence Transformer & Load or Build FAISS Dense Index
        print(f"Loading Sentence Transformer model ({self.embedding_model_name})...")
        self.semantic_model = SentenceTransformer(self.embedding_model_name)

        if not force_rebuild and os.path.exists(faiss_path):
            print("Loading cached FAISS semantic index...")
            self.faiss_index = faiss.read_index(faiss_path)
        else:
            print("Encoding documents with Sentence Transformer (this may take a few minutes)...")
            embeddings = self.semantic_model.encode(
                self.df["semantic_text"].tolist(),
                batch_size=128,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalized so Inner Product == Cosine Similarity
            ).astype("float32")

            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings)
            faiss.write_index(self.faiss_index, faiss_path)

        print("All enterprise indices ready!\n")

    # -------------------------------------------------------------------------
    # Core Ranking Algorithms
    # -------------------------------------------------------------------------

    def search_tfidf(self, query: str, top_k: int = 100, scope: str = "combined") -> list[tuple[int, float]]:
        """Ranks documents using TF-IDF and cosine similarity."""
        stemmed_tokens = self.query_processor.preprocess(query)
        if not stemmed_tokens:
            return []
        processed_query = " ".join(stemmed_tokens)

        vec = self.tfidf_vectorizers[scope]
        mat = self.tfidf_matrices[scope]

        query_vec = vec.transform([processed_query])
        scores = cosine_similarity(query_vec, mat).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def search_bm25(self, query: str, top_k: int = 100, scope: str = "combined") -> list[tuple[int, float]]:
        """Ranks documents using BM25Okapi probabilistic retrieval."""
        stemmed_tokens = self.query_processor.preprocess(query)
        if not stemmed_tokens:
            return []

        bm25 = self.bm25_models[scope]
        scores = bm25.get_scores(stemmed_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def search_semantic(self, query: str, top_k: int = 100) -> list[tuple[int, float]]:
        """Ranks documents using Sentence Transformer embeddings and FAISS inner-product search."""
        if not query.strip():
            return []

        query_emb = self.semantic_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.faiss_index.search(query_emb, top_k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1 and score > 0
        ]

    # -------------------------------------------------------------------------
    # Hybrid Model: Reciprocal Rank Fusion (RRF)
    # -------------------------------------------------------------------------

    def search_rrf(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool: int = 100,
        rrf_k: int = 60,
        scope: str = "combined"
    ) -> list[tuple[int, float]]:
        """
        Combines rankings from TF-IDF, BM25, and Sentence Transformer using Reciprocal Rank Fusion:
        RRF_Score(d) = sum_{m in Models} 1 / (rrf_k + rank_m(d))
        """
        tfidf_results = self.search_tfidf(query, top_k=candidate_pool, scope=scope)
        bm25_results = self.search_bm25(query, top_k=candidate_pool, scope=scope)
        semantic_results = self.search_semantic(query, top_k=candidate_pool)

        rrf_scores = {}

        for ranked_list in [tfidf_results, bm25_results, semantic_results]:
            for rank_idx, (doc_id, _) in enumerate(ranked_list, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank_idx))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(int(doc_id), float(score)) for doc_id, score in sorted_docs]

    # -------------------------------------------------------------------------
    # Unified Query Interface (Supports Body, Subject, and Thread Search)
    # -------------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        model: str = "rrf",
        scope: str = "combined",
        top_k: int = 10,
        group_by_thread: bool = False
    ) -> pd.DataFrame:
        """
        Executes a search query and returns a formatted DataFrame of results.
        - model: 'tfidf', 'bm25', 'semantic', or 'rrf'
        - scope: 'combined', 'subject', or 'body'
        - group_by_thread: If True, consolidates results by thread_id and adds thread message counts.
        """
        fetch_k = top_k * 5 if group_by_thread else top_k

        model = model.lower()
        if model == "tfidf":
            raw_hits = self.search_tfidf(query_text, top_k=fetch_k, scope=scope)
        elif model == "bm25":
            raw_hits = self.search_bm25(query_text, top_k=fetch_k, scope=scope)
        elif model == "semantic":
            raw_hits = self.search_semantic(query_text, top_k=fetch_k)
        elif model == "rrf":
            raw_hits = self.search_rrf(
                query_text, top_k=fetch_k, candidate_pool=max(100, fetch_k * 2), scope=scope
            )
        else:
            raise ValueError(f"Unknown model '{model}'. Choose from: tfidf, bm25, semantic, rrf.")

        if not raw_hits:
            return pd.DataFrame()

        doc_ids = [doc_id for doc_id, _ in raw_hits]
        scores = [score for _, score in raw_hits]

        results_df = self.df.iloc[doc_ids].copy()
        results_df["score"] = scores

        if group_by_thread:
            # Count total emails belonging to each thread in the entire corpus
            thread_counts = self.df["thread_id"].value_counts().to_dict()
            results_df["thread_size"] = results_df["thread_id"].map(thread_counts)
            # Keep the highest-scoring email per conversation thread
            results_df = results_df.drop_duplicates(subset=["thread_id"], keep="first")

        results_df = results_df.head(top_k).copy()
        results_df["rank"] = range(1, len(results_df) + 1)

        cols = [
            "rank", "doc_id", "score", "thread_id",
            "date", "sender", "recipients", "clean_subject", "clean_body"
        ]
        if group_by_thread:
            cols.insert(4, "thread_size")

        return results_df[cols]


def main():
    engine = EnronSearchEngine(
        parquet_path="enron_preprocessed.parquet",
        index_dir="enron_indices"
    )
    engine.build_or_load_indices(force_rebuild=False)

    sample_query = "California energy crisis power trading FERC investigation"
    print("=" * 80)
    print(f"SAMPLE QUERY: '{sample_query}'")
    print("=" * 80)

    for model_name in ["tfidf", "bm25", "semantic", "rrf"]:
        start_t = time.perf_counter()
        results = engine.query(
            query_text=sample_query,
            model=model_name,
            scope="combined",
            top_k=5,
            group_by_thread=False
        )
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        print(f"\n--- Top 5 Results | Model: {model_name.upper()} ({elapsed_ms:.1f} ms) ---")
        if results.empty:
            print("No matching documents found.")
            continue

        for _, row in results.iterrows():
            snippet = row["clean_body"][:110].replace("\n", " ")
            print(
                f"Rank {row['rank']} | DocID: {row['doc_id']:<6} | Score: {row['score']:.4f} | "
                f"Subject: {row['clean_subject'][:45]:<45} | Body: {snippet}..."
            )


if __name__ == "__main__":
    main()