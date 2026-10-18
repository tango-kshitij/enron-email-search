# 📧 Enterprise Email Search Engine using Enron Dataset

An end-to-end Enterprise Information Retrieval (IR) search engine built on the **Enron Email Dataset** (500,000+ real enterprise email messages from senior management). This system enables multi-field search across **email body text, subject lines, and reconstructed conversation threads**, combining classical lexical retrieval (**TF-IDF**, **BM25**) with dense neural embeddings (**Sentence Transformers**) via **Reciprocal Rank Fusion (RRF)**.

---

## 🎯 Project Overview & Standardized Tasks

1. **Text Preprocessing (`preprocess_enron.py`)**
   * **Header & Body Cleaning:** Parses RFC 822 raw email strings to extract metadata (`Message-ID`, `Date`, `From`, `To`, `Subject`) and strips forwarded blocks, inline reply headers, quoted lines (`>`), HTML tags, and URLs.
   * **Conversation Thread Reconstruction:** Normalizes `Re:`, `Fwd:`, and `FW:` prefixes and hashes normalized subject lines to group related emails into unified `Thread-ID` buckets.
   * **Tokenization & Stemming:** Applies lowercase normalization, regex tokenization, custom enterprise stop-word removal, and Porter Stemming (`PorterStemmer`) for lexical indices while retaining clean, natural text for dense semantic models.

2. **Enterprise Document Indexing (`index_and_rank.py`)**
   * **Multi-Field Lexical Indices:** Maintains dedicated indices for `Subject Only`, `Body Only`, and `Combined` (with $2\times$ subject field boosting).
   * **Dense Vector Index:** Encodes contextual email representations into 384-dimensional normalized vectors indexed via **FAISS** (`IndexFlatIP`).

3. **Ranking Algorithms & Hybrid Fusion (`index_and_rank.py`)**
   * **TF-IDF:** Sublinear TF scaling with unigram/bigram features (`scikit-learn`) and cosine similarity ranking.
   * **BM25:** Probabilistic Okapi BM25 ranking (`rank_bm25`, $k_1 = 1.5, b = 0.75$) with document length normalization.
   * **Sentence Transformer:** Dense semantic retrieval using `all-MiniLM-L6-v2` (`sentence-transformers` + `faiss-cpu`).
   * **Hybrid Model — Reciprocal Rank Fusion (RRF):** Combines rankings from all three retrieval paradigms without requiring score calibration:
     $$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad (k = 60)$$

4. **Output & Evaluation (`evaluate_and_ui.py`)**
   * **IR Evaluation Suite:** Benchmarks all four models using **Precision@10**, **Recall**, **MAP (Mean Average Precision)**, and **nDCG@10 (Normalized Discounted Cumulative Gain)**.
   * **Visualizations & Artifacts:** Generates a **Performance Comparison Table** (`performance_comparison.csv`) and an **11-Point Interpolated Precision-Recall Graph** (`precision_recall_graph.png`).
   * **Interactive Search Interface:** A full-featured **Streamlit** web application supporting keyword highlighting, field scope filtering, conversation thread grouping, and live evaluation metrics.

---

## 📂 Project Structure

```text
├── preprocess_enron.py          # Step 1: RFC 822 parsing, cleaning, thread grouping, stemming
├── index_and_rank.py            # Step 2: TF-IDF, BM25, FAISS indexing & Reciprocal Rank Fusion (RRF)
├── evaluate_and_ui.py           # Step 3: IR evaluation (P@10, Recall, MAP, nDCG), PR curve & Streamlit UI
├── requirements.txt             # Python dependencies
├── .gitignore                   # Excludes large dataset & index binaries from Git
├── performance_comparison.csv   # Generated evaluation table across all 4 models
└── precision_recall_graph.png   # Generated 11-point interpolated Precision-Recall plot
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
cd <YOUR-REPO-NAME>
```

### 2. Create a Virtual Environment & Install Dependencies
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the Enron Dataset
1. Download the **Enron Email Dataset** from [Kaggle (`wcukierski/enron-email-dataset`)](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset).
2. Extract `emails.csv` and place it in the root directory of this project.

---

## 🚀 Usage Guide

### Step 1: Preprocess the Dataset
Run the preprocessing pipeline to parse headers/bodies, reconstruct conversation threads, and generate stemmed and semantic columns:
```bash
python preprocess_enron.py
```
* **Output:** `enron_preprocessed.parquet` *(By default, `SAMPLE_SIZE = 50000` is configured in `preprocess_enron.py` for fast local prototyping; set `SAMPLE_SIZE = None` to process all 517,401 emails).*

### Step 2: Build the Enterprise Indices & Test Ranking Models
Build and cache the TF-IDF matrices, BM25 models, and FAISS dense vector index:
```bash
python index_and_rank.py
```
* **Output:** Saves serialized indices inside `enron_indices/` and prints top-5 sample search results across `TF-IDF`, `BM25`, `Sentence Transformer`, and `Hybrid RRF`.

### Step 3: Run the Evaluation Benchmark (CLI Mode)
Compute **Precision@10**, **Recall**, **MAP**, and **nDCG** across enterprise benchmark queries:
```bash
python evaluate_and_ui.py
```
* **Output:** Prints the **Performance Comparison Table** to the console, saves `performance_comparison.csv`, and exports `precision_recall_graph.png`.

### Step 4: Launch the Interactive Search Interface (Web UI)
Start the Streamlit web application:
```bash
streamlit run evaluate_and_ui.py
```
* Opens `http://localhost:8501` in your browser with two tabs:
  * **🔍 Enterprise Search Interface:** Toggle between ranking models, filter by `All Fields`, `Subject Line Only`, or `Email Body Only`, group results by conversation thread, and inspect highlighted snippets.
  * **📊 Output & Evaluation Metrics:** View the formatted **Performance Comparison Table** and **11-Point Interpolated Precision-Recall Curve**.

---

## 📊 Evaluation Metrics Implemented

* **Precision@10 ($P@10$):** Fraction of top-10 retrieved emails that are relevant:
  $$P@10 = \frac{|\{\text{Relevant}\} \cap \{\text{Top 10 Retrieved}\}|}{10}$$
* **Recall@K:** Fraction of all relevant emails in the pooled ground truth retrieved within the top $K$ results:
  $$\text{Recall}@K = \frac{|\{\text{Relevant}\} \cap \{\text{Top } K \text{ Retrieved}\}|}{|\{\text{All Relevant}\}|}$$
* **Mean Average Precision ($\text{MAP}$):** Mean of the Average Precision scores across all benchmark queries $Q$:
  $$\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{m_q} \sum_{k=1}^{K} P@k \cdot \text{rel}_q(k)$$
* **Normalized Discounted Cumulative Gain ($\text{nDCG}@10$):** Evaluates graded relevance ($\text{rel}_i \in \{0, 1, 2\}$) with logarithmic rank discounting:
  $$\text{nDCG}@10 = \frac{\text{DCG}@10}{\text{IDCG}@10}, \quad \text{DCG}@10 = \sum_{i=1}^{10} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

---

## 🧠 Learning Outcomes

* **Semi-Structured Text Search:** Parsing RFC 822 headers, stripping reply noise, and weighting structural email fields (`Subject` vs. `Body`).
* **Enterprise Archival IR:** Reconstructing multi-message email threads and indexing high-volume archival communications.
* **Hybrid Reciprocal Rank Fusion (RRF):** Combining exact lexical matching (BM25/TF-IDF) with dense semantic retrieval (Sentence Transformers + FAISS) to maximize both precision and recall.