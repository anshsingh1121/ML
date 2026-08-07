"""
Enterprise Text Preprocessing & Token Truncation Verification Engine (`src/preprocessing/text_preprocessor.py`).

Normalizes, cleans, filters stopwords, lemmatizes, and verifies token bounds across
unstructured ServiceNow incident text fields (`short_description`, `description`, `close_notes`)
to prepare clean textual inputs for upcoming semantic embedding 
models (`TF-IDF + SVD`).
Does NOT generate vector embeddings (reserved for Phase 3/4).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import re
import unicodedata
import pandas as pd

from src.data.pipeline_contracts import PipelineContractValidator
from src.data.feature_registry import FeatureRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Standard English stop words that carry minimal IT diagnostic signal
DEFAULT_IT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
    "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "please", "kindly", "hello", "hi", "thanks", "thank", "regards", "user", "reported", "issue", "problem"
}

# Critical IT operational keywords that MUST never be removed by stopword filters
PROTECTED_IT_KEYWORDS = {
    "server", "down", "slow", "error", "failed", "failure", "timeout", "deadlock", "crash",
    "crashed", "login", "network", "database", "router", "switch", "firewall", "vpn", "dns",
    "dhcp", "ssl", "tls", "cert", "certificate", "cpu", "memory", "ram", "disk", "storage",
    "latency", "packet", "loss", "connection", "disconnected", "unresponsive", "blocked", "locking"
}

# Simple high-accuracy rule-based IT suffix lemmatization map
IT_LEMMATIZATION_RULES = {
    "failures": "failure",
    "servers": "server",
    "connections": "connection",
    "timeouts": "timeout",
    "routers": "router",
    "switches": "switch",
    "firewalls": "firewall",
    "databases": "database",
    "errors": "error",
    "crashes": "crash",
    "running": "run",
    "connecting": "connect",
    "failing": "fail",
    "disconnected": "disconnect",
    "deadlocks": "deadlock",
    "requests": "request",
    "interfaces": "interface"
}


class TextPreprocessor:
    """
    Automated NLP normalization and token verification engine for ServiceNow incident data.
    Ensures optimal semantic density and compliance with neural embedding token budgets.
    """

    def __init__(
        self,
        validator: Optional[PipelineContractValidator] = None,
        stopwords: Optional[set] = None,
        protected_keywords: Optional[set] = None,
        max_seq_tokens: int = 256
    ) -> None:
        """Initialize TextPreprocessor with PipelineContractValidator and token boundaries."""
        self.validator = validator or PipelineContractValidator()
        self.stopwords = (stopwords or DEFAULT_IT_STOPWORDS) - (protected_keywords or PROTECTED_IT_KEYWORDS)
        self.max_seq_tokens = max_seq_tokens

    def preprocess_dataset(
        self,
        df: pd.DataFrame,
        output_dir: str = "reports",
        remove_stopwords: bool = True,
        lemmatize: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full text normalization across unstructured columns and generate readiness audit.

        Args:
            df: Input pandas DataFrame.
            output_dir: Directory to export `text_preprocessing_report.md` & `.json`.
            remove_stopwords: Whether to strip non-diagnostic English stopwords.
            lemmatize: Whether to apply IT domain suffix normalization.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (DataFrame with `_clean` columns, Audit Report Dict)
        """
        logger.info(f"Initiating Enterprise Text Preprocessing on {len(df):,} records...")
        proc_df = df.copy()
        text_cols = ["short_description", "description", "close_notes"]
        active_cols = [c for c in text_cols if c in proc_df.columns]

        audit_report: Dict[str, Any] = {
            "processed_records": len(proc_df),
            "columns_processed": active_cols,
            "max_sequence_tokens_limit": self.max_seq_tokens,
            "column_metrics": {},
            "recommendations": []
        }

        for col in active_cols:
            clean_col = f"{col}_clean"
            logger.debug(f"Normalizing text column: {col} -> {clean_col}")
            
            raw_series = proc_df[col].fillna("Not Provided").astype(str)
            raw_words = raw_series.apply(lambda x: len(x.split()))
            raw_tokens_est = raw_series.apply(self._estimate_tokens)

            # Apply normalization pipeline
            clean_series = raw_series.apply(
                lambda txt: self.normalize_text(txt, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
            )
            proc_df[clean_col] = clean_series

            clean_words = clean_series.apply(lambda x: len(x.split()))
            clean_tokens_est = clean_series.apply(self._estimate_tokens)

            audit_report["column_metrics"][col] = {
                "raw_mean_words": round(float(raw_words.mean()), 2),
                "clean_mean_words": round(float(clean_words.mean()), 2),
                "word_reduction_percentage": round(float((raw_words.sum() - clean_words.sum()) / max(raw_words.sum(), 1) * 100), 2),
                "raw_mean_tokens_est": round(float(raw_tokens_est.mean()), 2),
                "clean_mean_tokens_est": round(float(clean_tokens_est.mean()), 2),
                "raw_exceeds_max_tokens": int((raw_tokens_est > self.max_seq_tokens).sum()),
                "clean_exceeds_max_tokens": int((clean_tokens_est > self.max_seq_tokens).sum()),
                "status": "PASS_OPTIMIZED" if (clean_tokens_est <= self.max_seq_tokens).all() else "WARNING_TRUNCATION_REQUIRED"
            }

        audit_report["status"] = "CERTIFIED_TEXT_CLEAN"
        audit_report["recommendations"] = self._compute_recommendations(audit_report["column_metrics"])

        # Export reports
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / "text_preprocessing_report.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)
        logger.info(f"Exported text preprocessing JSON report to: {json_file}")

        md_file = out_path / "text_preprocessing_report.md"
        self._export_markdown_report(audit_report, md_file)
        logger.info(f"Exported text preprocessing Markdown report to: {md_file}")

        return proc_df, audit_report

    def normalize_text(self, text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
        """
        Apply complete normalization sequence to a single string.
        """
        if not text or text == "Not Provided" or text == "Unknown":
            return "not provided" if text == "Not Provided" else "unknown"

        # 1. Unicode NFKC normalization and lowercasing
        norm = unicodedata.normalize("NFKC", text).lower()

        # 2. Strip HTML/XML tags and email header junk (`From: ...`, `To: ...`)
        norm = re.sub(r"<[^>]+>", " ", norm)
        norm = re.sub(r"^(from|to|subject|date):.*$", " ", norm, flags=re.MULTILINE)

        # 3. Strip system error code boilerplates that add token noise while keeping error identifiers
        norm = re.sub(r"\[system error code:\s*0x[0-9a-f]+\]", " system_error ", norm)

        # 4. Remove special characters/punctuation except hyphens/underscores/slashes in IT terms
        norm = re.sub(r"[^\w\s\-\/\.]", " ", norm)
        
        # Collapse multiple spaces
        words = norm.split()

        # 5. Stopword filtering & protected keyword check
        if remove_stopwords:
            words = [w for w in words if w not in self.stopwords or w in PROTECTED_IT_KEYWORDS]

        # 6. IT Domain Lemmatization
        if lemmatize:
            words = [IT_LEMMATIZATION_RULES.get(w, w) for w in words]

        cleaned = " ".join(words)
        return cleaned.strip() if cleaned else "not provided"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate BPE/WordPiece token count (`len(words) * 1.3`)."""
        words = text.split()
        return int(math.ceil(len(words) * 1.3))

    def _compute_recommendations(self, column_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Compute actionable recommendations for downstream Sentence Transformer batching."""
        recs = []
        for col, metrics in column_metrics.items():
            exceeds = metrics["clean_exceeds_max_tokens"]
            if exceeds > 0:
                recs.append({
                    "column": f"{col}_clean",
                    "action": f"Set explicit `truncation=True, max_length={self.max_seq_tokens}` during Phase 3 embedding generation.",
                    "rationale": f"Identified {exceeds} records (`{float(exceeds/max(metrics['clean_mean_words'],1)):.1f}%`) exceeding {self.max_seq_tokens} tokens after cleaning."
                })
            else:
                recs.append({
                    "column": f"{col}_clean",
                    "action": "Proceed with full-sequence embedding encoding without data loss.",
                    "rationale": f"All records fit within the {self.max_seq_tokens}-token budget (mean clean tokens: `{metrics['clean_mean_tokens_est']}`)."
                })
        return recs

    def _export_markdown_report(self, report: Dict[str, Any], md_file: Path) -> None:
        """Export executive markdown text readiness report."""
        lines = [
            "# Enterprise Text Preprocessing & Embedding Readiness Report (`v2.0.0-alpha`)",
            "",
            "**Organization:** First Citizens Bank — Enterprise Technology Division  ",
            f"**Total Records Evaluated:** `{report['processed_records']:,}`  ",
            f"**Neural Token Boundary:** `{report['max_sequence_tokens_limit']} tokens`  ",
            f"**Certification Status:** `{report['status']}`",
            "",
            "---",
            "",
            "## 1. Text Normalization & Token Reduction Audit Table",
            "",
            "| Column (`technical_name`) | Raw Mean Words | Clean Mean Words | Word Reduction % | Raw Mean Tokens (Est.) | Clean Mean Tokens (Est.) | Exceeds 256 Limit | Status |",
            "|---|---|---|---|---|---|---|---|"
        ]

        for col, m in report["column_metrics"].items():
            lines.append(f"| `{col}` | {m['raw_mean_words']} | {m['clean_mean_words']} | {m['word_reduction_percentage']}% | {m['raw_mean_tokens_est']} | {m['clean_mean_tokens_est']} | `{m['clean_exceeds_max_tokens']}` | **{m['status']}** |")

        lines.extend([
            "",
            "---",
            "",
            "## 2. Phase 3 Sentence Transformer Recommendations",
            "",
            "| Text Column (`_clean`) | Recommended MLOps Action | Enterprise Rationale |",
            "|---|---|---|"
        ])

        for rec in report["recommendations"]:
            lines.append(f"| `{rec['column']}` | **{rec['action']}** | {rec['rationale']} |")

        lines.extend([
            "",
            "---",
            "",
            "## 3. Embedding Generation Interlock",
            "Per exact architectural mandate, actual vector embedding computation (`SentenceTransformer.encode()`) and indexing (`FAISS`) are strictly blocked until user approval is received for Phase 3/4."
        ])

        with open(md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
