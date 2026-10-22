import os
import re
import time
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from index_and_rank import EnronSearchEngine

# -----------------------------------------------------------------------------
# 1. Benchmark Queries & Graded Relevance Assessor
# -----------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    {
        "query": "California energy crisis power trading electricity market",
        "must_match": ["california", "power", "energy", "electricity", "crisis", "trading"]
    },
    {
        "query": "FERC investigation regulatory compliance audit",
        "must_match": ["ferc", "investigation", "regulatory", "commission", "order"]
    },
    {
        "query": "natural gas pipeline capacity transportation contract",
        "must_match": ["gas", "pipeline", "capacity", "transportation", "contract"]
    },
    {
        "query": "financial risk management hedging derivatives credit",
        "must_match": ["risk", "hedging", "credit", "derivative", "financial", "exposure"]
    },
    {
        "query": "broadband communications bandwidth network services",
        "must_match": ["broadband", "bandwidth", "network", "communications", "fiber"]
    },
    {
        "query": "board of directors executive meeting schedule",
        "must_match": ["board", "directors", "executive", "meeting", "committee"]
    },
    {
        "query": "legal contract agreement master netting",
        "must_match": ["legal", "agreement", "contract", "master", "netting", "counsel"]
    },
    {
        "query": "quarterly earnings financial report revenue forecast",
        "must_match": ["earnings", "quarterly", "revenue", "financial", "forecast", "report"]
    }
]


class EnronEvaluator:
    """
    Evaluates TF-IDF, BM25, Sentence Transformer, and Hybrid RRF across:
    - Precision@10
    - Recall@K
    - MAP (Mean Average Precision)
    - nDCG@10 (Normalized Discounted Cumulative Gain)
    - 11-Point Interpolated Precision-Recall Curves
    """

    def __init__(self, engine: EnronSearchEngine, pool_depth: int = 100):
        self.engine = engine
        self.pool_depth = pool_depth
        self.models = ["tfidf", "bm25", "semantic", "rrf"]
        self.model_labels = {
            "tfidf": "TF-IDF",
            "bm25": "BM25",
            "semantic": "Sentence Transformer",
            "rrf": "Hybrid RRF"
        }

    def build_pooled_qrels(self, benchmark_queries: list[dict]) -> dict[str, dict[int, int]]:
        """
        Builds graded relevance judgments (0 = Irrelevant, 1 = Partially Relevant, 2 = Highly Relevant)
        using standard IR depth-k pooling across all models combined with domain term verification.
        """
        qrels = {}

        for item in benchmark_queries:
            q_text = item["query"]
            keywords = item["must_match"]

            # Pool candidate documents from all 4 models
            pooled_doc_ids = set()
            model_ranks = {m: {} for m in self.models}

            for m in self.models:
                hits = self._run_model(m, q_text, top_k=self.pool_depth)
                for rank_idx, (doc_id, _) in enumerate(hits, start=1):
                    pooled_doc_ids.add(doc_id)
                    model_ranks[m][doc_id] = rank_idx

            query_relevance = {}
            for doc_id in pooled_doc_ids:
                row = self.engine.df.iloc[doc_id]
                subj_lower = str(row["clean_subject"]).lower()
                body_lower = str(row["clean_body"]).lower()

                subj_hits = sum(1 for kw in keywords if kw in subj_lower)
                body_hits = sum(1 for kw in keywords if kw in body_lower)
                total_kw_hits = subj_hits + body_hits

                # Count how many distinct paradigms retrieved this doc in their top 30
                consensus_count = sum(
                    1 for m in ["tfidf", "bm25", "semantic"]
                    if model_ranks[m].get(doc_id, 999) <= 30
                )

                # Assign graded relevance (0, 1, or 2) for nDCG and binary relevance (> 0) for P/R/MAP
                if (subj_hits >= 1 and body_hits >= 2) or (total_kw_hits >= 3 and consensus_count >= 2):
                    query_relevance[doc_id] = 2  # Highly relevant
                elif total_kw_hits >= 2:
                    query_relevance[doc_id] = 1  # Partially relevant
                else:
                    query_relevance[doc_id] = 0  # Not relevant

            qrels[q_text] = query_relevance

        return qrels

    def _run_model(self, model: str, query: str, top_k: int) -> list[tuple[int, float]]:
        if model == "tfidf":
            return self.engine.search_tfidf(query, top_k=top_k)
        elif model == "bm25":
            return self.engine.search_bm25(query, top_k=top_k)
        elif model == "semantic":
            return self.engine.search_semantic(query, top_k=top_k)
        elif model == "rrf":
            return self.engine.search_rrf(query, top_k=top_k, candidate_pool=max(100, top_k))
        return []

    @staticmethod
    def precision_at_k(retrieved_ids: list[int], qrel: dict[int, int], k: int = 10) -> float:
        if k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        rel_count = sum(1 for doc_id in top_k if qrel.get(doc_id, 0) > 0)
        return rel_count / k

    @staticmethod
    def recall_at_k(retrieved_ids: list[int], qrel: dict[int, int], k: int = 100) -> float:
        total_relevant = sum(1 for grade in qrel.values() if grade > 0)
        if total_relevant == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        rel_retrieved = sum(1 for doc_id in top_k if qrel.get(doc_id, 0) > 0)
        return rel_retrieved / total_relevant

    @staticmethod
    def average_precision(retrieved_ids: list[int], qrel: dict[int, int], k: int = 100) -> float:
        total_relevant = sum(1 for grade in qrel.values() if grade > 0)
        if total_relevant == 0:
            return 0.0

        hit_count = 0
        precision_sum = 0.0
        for idx, doc_id in enumerate(retrieved_ids[:k], start=1):
            if qrel.get(doc_id, 0) > 0:
                hit_count += 1
                precision_sum += hit_count / idx

        return precision_sum / total_relevant

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[int], qrel: dict[int, int], k: int = 10) -> float:
        top_k = retrieved_ids[:k]
        dcg = sum(
            (2 ** qrel.get(doc_id, 0) - 1) / np.log2(idx + 1)
            for idx, doc_id in enumerate(top_k, start=1)
        )

        ideal_grades = sorted([g for g in qrel.values() if g > 0], reverse=True)[:k]
        idcg = sum(
            (2 ** grade - 1) / np.log2(idx + 1)
            for idx, grade in enumerate(ideal_grades, start=1)
        )

        return (dcg / idcg) if idcg > 0 else 0.0

    @staticmethod
    def interpolated_pr_curve(
        retrieved_ids: list[int], qrel: dict[int, int], recall_levels: np.ndarray
    ) -> np.ndarray:
        total_relevant = sum(1 for grade in qrel.values() if grade > 0)
        if total_relevant == 0 or not retrieved_ids:
            return np.zeros_like(recall_levels)

        precisions = []
        recalls = []
        rel_found = 0

        for idx, doc_id in enumerate(retrieved_ids, start=1):
            if qrel.get(doc_id, 0) > 0:
                rel_found += 1
                precisions.append(rel_found / idx)
                recalls.append(rel_found / total_relevant)

        if not recalls:
            return np.zeros_like(recall_levels)

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        interp_p = np.zeros_like(recall_levels)
        for i, r_level in enumerate(recall_levels):
            mask = recalls >= r_level
            interp_p[i] = np.max(precisions[mask]) if np.any(mask) else 0.0

        return interp_p

    def evaluate_all(self, k_eval: int = 100) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
        """
        Runs all four ranking algorithms across the benchmark queries and returns:
        1. Performance Comparison DataFrame (Precision@10, Recall, MAP, nDCG)
        2. Interpolated Precision arrays per model
        3. Standard 11-point Recall levels (0.0 to 1.0)
        """
        qrels = self.build_pooled_qrels(BENCHMARK_QUERIES)
        recall_levels = np.linspace(0.0, 1.0, 11)

        summary_rows = []
        pr_curves = {}

        for m in self.models:
            p10_list, rec_list, ap_list, ndcg_list, pr_list = [], [], [], [], []

            for item in BENCHMARK_QUERIES:
                q_text = item["query"]
                qrel = qrels[q_text]

                hits = self._run_model(m, q_text, top_k=k_eval)
                retrieved_ids = [doc_id for doc_id, _ in hits]

                p10_list.append(self.precision_at_k(retrieved_ids, qrel, k=10))
                rec_list.append(self.recall_at_k(retrieved_ids, qrel, k=k_eval))
                ap_list.append(self.average_precision(retrieved_ids, qrel, k=k_eval))
                ndcg_list.append(self.ndcg_at_k(retrieved_ids, qrel, k=10))
                pr_list.append(self.interpolated_pr_curve(retrieved_ids, qrel, recall_levels))

            summary_rows.append({
                "Model": self.model_labels[m],
                "Precision@10": round(float(np.mean(p10_list)), 4),
                "Recall": round(float(np.mean(rec_list)), 4),
                "MAP": round(float(np.mean(ap_list)), 4),
                "nDCG": round(float(np.mean(ndcg_list)), 4)
            })
            pr_curves[self.model_labels[m]] = np.mean(pr_list, axis=0)

        comparison_df = pd.DataFrame(summary_rows)
        return comparison_df, pr_curves, recall_levels

    @staticmethod
    def plot_pr_graph(
        pr_curves: dict[str, np.ndarray],
        recall_levels: np.ndarray,
        save_path: str = "precision_recall_graph.png"
    ) -> plt.Figure:
        """Generates and saves the 11-point Interpolated Precision-Recall Graph."""
        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)

        styles = {
            "TF-IDF": {"color": "#64748b", "marker": "o", "linestyle": "--"},
            "BM25": {"color": "#2563eb", "marker": "s", "linestyle": "-."},
            "Sentence Transformer": {"color": "#9333ea", "marker": "^", "linestyle": ":"},
            "Hybrid RRF": {"color": "#16a34a", "marker": "D", "linestyle": "-", "linewidth": 2.5}
        }

        for label, p_vals in pr_curves.items():
            style = styles.get(label, {})
            ax.plot(recall_levels, p_vals, label=label, **style)

        ax.set_title("Enterprise Email Search: 11-Point Interpolated Precision-Recall Curve", fontsize=13, pad=12)
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Interpolated Precision", fontsize=11)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", frameon=True, fontsize=10)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)

        return fig


# -----------------------------------------------------------------------------
# 2. Interactive Streamlit Search Interface
# -----------------------------------------------------------------------------

def highlight_terms(text: str, query: str) -> str:
    """Highlights query keywords inside cleaned email snippets safely."""
    safe_text = html.escape(text)
    terms = [t for t in re.findall(r"\b[a-zA-Z]{3,}\b", query) if t.lower() not in {"and", "the", "for", "with"}]
    if not terms:
        return safe_text
    pattern = re.compile(r"(" + "|".join(re.escape(t) for t in terms) + r")", re.IGNORECASE)
    return pattern.sub(
        r'<mark style="background-color: #fde047; padding: 0 2px; border-radius: 3px;">\1</mark>',
        safe_text
    )


def run_streamlit_ui():
    import streamlit as st

    st.set_page_config(
        page_title="Enron Enterprise Email Search",
        page_icon="📧",
        layout="wide"
    )

    @st.cache_resource
    def get_engine():
        eng = EnronSearchEngine(
            parquet_path="enron_preprocessed.parquet",
            index_dir="enron_indices"
        )
        eng.build_or_load_indices(force_rebuild=False)
        return eng

    @st.cache_data
    def get_cached_evaluation():
        eng = get_engine()
        evaluator = EnronEvaluator(eng)
        return evaluator.evaluate_all(k_eval=100)

    engine = get_engine()

    st.title("📧 Enron Enterprise Email Search Engine")
    st.caption(
        f"Indexed Corpus: **{len(engine.df):,}** enterprise emails | "
        "Models: **TF-IDF**, **BM25**, **Sentence Transformer**, and **Hybrid RRF**"
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Search Configuration")
        model_choice = st.selectbox(
            "Ranking Algorithm",
            options=["rrf", "bm25", "semantic", "tfidf"],
            format_func=lambda x: {
                "rrf": "Hybrid Model: Reciprocal Rank Fusion (RRF)",
                "bm25": "BM25 (Probabilistic Lexical)",
                "semantic": "Sentence Transformer (Dense Semantic)",
                "tfidf": "TF-IDF (Lexical Baseline)"
            }[x]
        )

        scope_choice = st.radio(
            "Search Field Scope",
            options=["combined", "subject", "body"],
            format_func=lambda x: {
                "combined": "All Fields (Subject + Body)",
                "subject": "Subject Line Only",
                "body": "Email Body Only"
            }[x],
            help="Note: Sentence Transformer searches full contextual embeddings."
        )

        group_threads = st.checkbox("Group by Conversation Thread", value=False)
        top_k = st.slider("Number of Results (Top-K)", min_value=5, max_value=50, value=10, step=5)

    tab_search, tab_eval = st.tabs(["🔍 Enterprise Search Interface", "📊 Output & Evaluation Metrics"])

    # --- TAB 1: SEARCH INTERFACE ---
    with tab_search:
        query_input = st.text_input(
            "Enter enterprise search query:",
            value="California energy crisis power trading FERC investigation",
            placeholder="Search subjects, body text, or conversation threads..."
        )

        if query_input.strip():
            t0 = time.perf_counter()
            results_df = engine.query(
                query_text=query_input,
                model=model_choice,
                scope=scope_choice,
                top_k=top_k,
                group_by_thread=group_threads
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            st.markdown(
                f"Found **{len(results_df)}** results in **{elapsed_ms:.1f} ms** "
                f"using `{model_choice.upper()}` (`scope={scope_choice}`)."
            )

            if results_df.empty:
                st.warning("No matching emails found for this query.")
            else:
                for _, row in results_df.iterrows():
                    subj_display = row["clean_subject"] if row["clean_subject"] else "(No Subject)"
                    thread_badge = (
                        f" | 🧵 Thread ({row['thread_size']} msgs)"
                        if group_threads and "thread_size" in row
                        else f" | Thread ID: `{row['thread_id']}`"
                    )

                    header_title = (
                        f"**#{row['rank']}** — **{subj_display}** "
                        f"*(Score: `{row['score']:.4f}` | DocID: `{row['doc_id']}`)*"
                    )

                    with st.expander(header_title, expanded=(row["rank"] <= 3)):
                        st.markdown(
                            f"**From:** `{row['sender']}` | **To:** `{row['recipients'][:80]}` | "
                            f"**Date:** `{row['date']}`{thread_badge}"
                        )
                        st.divider()

                        snippet = row["clean_body"][:900] + ("..." if len(row["clean_body"]) > 900 else "")
                        highlighted_body = highlight_terms(snippet, query_input)
                        st.markdown(
                            f"<div style='line-height:1.55; font-family:sans-serif;'>{highlighted_body}</div>",
                            unsafe_allow_html=True
                        )

                        # Thread Viewer: Show all related emails in the same thread
                        thread_df = engine.df[engine.df["thread_id"] == row["thread_id"]]
                        if len(thread_df) > 1 and row["thread_id"] != "thread_no_subject":
                            st.markdown(f"**🧵 Conversation Thread ({len(thread_df)} related emails in corpus):**")
                            st.dataframe(
                                thread_df[["doc_id", "date", "sender", "clean_subject", "clean_body"]].head(15),
                                use_container_width=True,
                                hide_index=True
                            )

    # --- TAB 2: EVALUATION DASHBOARD ---
    with tab_eval:
        st.subheader("Model Performance Comparison Table")
        st.markdown(
            "Evaluated across enterprise benchmark queries using **Precision@10**, **Recall**, "
            "**MAP (Mean Average Precision)**, and **nDCG@10**."
        )

        comp_df, pr_curves, recall_levels = get_cached_evaluation()

        # Highlight highest score in each metric column
        styled_df = comp_df.style.highlight_max(
            subset=["Precision@10", "Recall", "MAP", "nDCG"],
            color="#1c5e33"
        ).format({
            "Precision@10": "{:.4f}",
            "Recall": "{:.4f}",
            "MAP": "{:.4f}",
            "nDCG": "{:.4f}"
        })
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.subheader("Precision-Recall Graph")
        fig = EnronEvaluator.plot_pr_graph(pr_curves, recall_levels, save_path="precision_recall_graph.png")
        st.pyplot(fig)


# -----------------------------------------------------------------------------
# 3. CLI Entrypoint (Generates CSV Table & PNG Graph Directly)
# -----------------------------------------------------------------------------

def run_cli_evaluation():
    print("Initializing Enron Search Engine for Evaluation...")
    engine = EnronSearchEngine(
        parquet_path="enron_preprocessed.parquet",
        index_dir="enron_indices"
    )
    engine.build_or_load_indices(force_rebuild=False)

    print("Running IR Benchmark Evaluation (Precision@10, Recall, MAP, nDCG)...")
    evaluator = EnronEvaluator(engine, pool_depth=100)
    comparison_df, pr_curves, recall_levels = evaluator.evaluate_all(k_eval=100)

    print("\n" + "=" * 68)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 68)
    print(comparison_df.to_string(index=False))
    print("=" * 68)

    csv_path = "performance_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved Performance Comparison Table to '{csv_path}'.")

    graph_path = "precision_recall_graph.png"
    evaluator.plot_pr_graph(pr_curves, recall_levels, save_path=graph_path)
    print(f"Saved Precision-Recall Graph to '{graph_path}'.")
    print("\nTo launch the interactive Search Interface, run:")
    print("    streamlit run evaluate.py")


if __name__ == "__main__":
    # Automatically detect whether the script was invoked via `streamlit run` or `python`
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except ImportError:
        in_streamlit = False

    if in_streamlit:
        run_streamlit_ui()
    else:
        run_cli_evaluation()