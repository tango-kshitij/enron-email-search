import os
import re
import hashlib
from email.parser import Parser
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from tqdm import tqdm

# Enable tqdm progress bars for pandas operations
tqdm.pandas()

# Ensure required NLTK resources are downloaded
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


class EnronPreprocessor:
    """
    Preprocessing pipeline for the Kaggle Enron Email Dataset (emails.csv).
    Handles Header/Body Cleaning, Thread Grouping, Tokenization, and Stemming.
    """

    def __init__(self):
        self.email_parser = Parser()
        self.tokenizer = RegexpTokenizer(r"\b[a-zA-Z]{2,}\b")  # Keep words with >= 2 letters
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

        # Add common email-specific noise tokens to stop words
        self.stop_words.update({
            "enron", "ect", "hou", "corp", "com", "http", "www",
            "mailto", "subject", "cc", "bcc", "re", "fw", "fwd"
        })

        # Compiled regex patterns for fast execution across 500k+ rows
        self.re_reply_prefix = re.compile(r"^(\s*(re|fw|fwd)\s*(\[\d+\])?\s*:\s*)+", re.IGNORECASE)
        self.re_forwarded_block = re.compile(
            r"(-+\s*Original Message\s*-+|-+\s*Forwarded by.*?-+)[\s\S]*",
            re.IGNORECASE
        )
        self.re_inline_headers = re.compile(
            r"^(From|Sent|To|Cc|Bcc|Subject|Date):.*$",
            re.IGNORECASE | re.MULTILINE
        )
        self.re_quoted_lines = re.compile(r"^\s*>.*$", re.MULTILINE)
        self.re_html_tags = re.compile(r"<[^>]+>")
        self.re_urls_emails = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
        self.re_whitespace = re.compile(r"\s+")

    def parse_raw_message(self, raw_message: str) -> dict:
        """
        Extracts key metadata headers and raw body payload from an RFC 822 email string.
        """
        if not isinstance(raw_message, str):
            return {
                "message_id": "", "date": "", "sender": "",
                "recipients": "", "subject": "", "raw_body": ""
            }

        msg = self.email_parser.parsestr(raw_message)

        # Extract payload (handle multipart gracefully if present)
        if msg.is_multipart():
            parts = [
                part.get_payload(decode=True).decode("utf-8", errors="ignore")
                for part in msg.walk()
                if part.get_content_type() == "text/plain" and part.get_payload(decode=True)
            ]
            body = "\n".join(parts)
        else:
            payload = msg.get_payload(decode=True)
            body = payload.decode("utf-8", errors="ignore") if payload else msg.get_payload()

        return {
            "message_id": str(msg.get("Message-ID", "")).strip(),
            "date": str(msg.get("Date", "")).strip(),
            "sender": str(msg.get("From", "")).strip(),
            "recipients": str(msg.get("To", "")).strip(),
            "subject": str(msg.get("Subject", "")).strip(),
            "raw_body": str(body) if body else ""
        }

    def clean_subject_and_get_thread_key(self, subject: str) -> str:
        """
        Strips 'Re:', 'Fwd:', and 'FW:' prefixes and normalizes whitespace
        to reconstruct conversation threads.
        """
        if not subject:
            return ""
        cleaned = self.re_reply_prefix.sub("", subject)
        cleaned = self.re_whitespace.sub(" ", cleaned).strip()
        return cleaned

    def clean_body_text(self, body: str) -> str:
        """
        Cleans the email body by removing forwarded chains, quoted replies,
        inline headers, HTML tags, URLs, and excess whitespace.
        """
        if not body:
            return ""

        # 1. Truncate or strip redundant 'Original Message' / 'Forwarded by' blocks
        text = self.re_forwarded_block.sub(" ", body)

        # 2. Remove inline header lines and quoted lines ('>')
        text = self.re_inline_headers.sub(" ", text)
        text = self.re_quoted_lines.sub(" ", text)

        # 3. Strip HTML tags, URLs, and email addresses
        text = self.re_html_tags.sub(" ", text)
        text = self.re_urls_emails.sub(" ", text)

        # 4. Normalize whitespace
        text = self.re_whitespace.sub(" ", text).strip()
        return text

    def tokenize_and_stem(self, text: str) -> str:
        """
        Lowercases, tokenizes, removes stop words, and applies Porter stemming.
        Returns a space-joined string of stemmed tokens for TF-IDF and BM25.
        """
        if not text:
            return ""

        tokens = self.tokenizer.tokenize(text.lower())
        stemmed_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words
        ]
        return " ".join(stemmed_tokens)

    @staticmethod
    def generate_thread_id(norm_subject: str) -> str:
        """
        Generates a deterministic Thread ID from the normalized subject line.
        Emails without a subject get a unique placeholder or 'no_subject' bucket.
        """
        if not norm_subject:
            return "thread_no_subject"
        subject_hash = hashlib.md5(norm_subject.lower().encode("utf-8")).hexdigest()[:12]
        return f"thread_{subject_hash}"

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the end-to-end preprocessing pipeline on the loaded DataFrame.
        """
        print("1/5 Parsing email headers and bodies...")
        parsed_records = df["message"].progress_apply(self.parse_raw_message)
        parsed_df = pd.DataFrame(parsed_records.tolist(), index=df.index)
        df = pd.concat([df[["file"]], parsed_df], axis=1)

        print("2/5 Cleaning subjects and reconstructing Thread IDs...")
        df["clean_subject"] = df["subject"].progress_apply(self.clean_subject_and_get_thread_key)
        df["thread_id"] = df["clean_subject"].apply(self.generate_thread_id)

        print("3/5 Cleaning email body text...")
        df["clean_body"] = df["raw_body"].progress_apply(self.clean_body_text)

        # Drop raw_body to save memory and filter out completely empty messages
        df.drop(columns=["raw_body"], inplace=True)
        initial_count = len(df)
        df = df[(df["clean_subject"] != "") | (df["clean_body"] != "")].copy()
        df.drop_duplicates(subset=["clean_subject", "clean_body", "sender"], inplace=True)
        print(f"    Removed {initial_count - len(df)} empty or duplicate emails.")

        print("4/5 Creating semantic text field (for Sentence Transformer)...")
        df["semantic_text"] = (
            "Subject: " + df["clean_subject"] + " | Body: " + df["clean_body"]
        ).str.strip()

        print("5/5 Tokenizing and stemming text (for TF-IDF & BM25)...")
        df["stemmed_subject"] = df["clean_subject"].progress_apply(self.tokenize_and_stem)
        df["stemmed_body"] = df["clean_body"].progress_apply(self.tokenize_and_stem)

        # Boost subject weight (2x) in combined lexical field for enterprise indexing
        df["stemmed_combined"] = (
            (df["stemmed_subject"] + " ") * 2 + df["stemmed_body"]
        ).str.strip()

        # Assign a clean integer doc_id for downstream FAISS and inverted index lookups
        df.reset_index(drop=True, inplace=True)
        df["doc_id"] = df.index

        return df


def main():
    # Path to the Kaggle Enron dataset CSV
    INPUT_CSV_PATH = "emails.csv"
    OUTPUT_PARQUET_PATH = "enron_preprocessed.parquet"

    # Set SAMPLE_SIZE = None to process all 517,401 emails, or an integer (e.g., 50000) for fast prototyping
    SAMPLE_SIZE = 50000

    if not os.path.exists(INPUT_CSV_PATH):
        raise FileNotFoundError(
            f"Could not find '{INPUT_CSV_PATH}'. Download it from Kaggle and place it in the project directory."
        )

    print(f"Loading dataset from {INPUT_CSV_PATH} (nrows={SAMPLE_SIZE})...")
    df_raw = pd.read_csv(INPUT_CSV_PATH, nrows=SAMPLE_SIZE)
    print(f"Loaded {len(df_raw):,} raw emails.\n")

    preprocessor = EnronPreprocessor()
    df_processed = preprocessor.process_dataframe(df_raw)

    print(f"\nSaving {len(df_processed):,} preprocessed emails to {OUTPUT_PARQUET_PATH}...")
    df_processed.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print("Done! Sample output columns:")
    print(df_processed[[
        "doc_id", "thread_id", "clean_subject", "stemmed_combined", "semantic_text"
    ]].head(3))


if __name__ == "__main__":
    main()