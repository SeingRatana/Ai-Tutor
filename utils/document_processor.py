import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

# Optional deps
try:
    import khmernltk  # pip install khmernltk
    _HAS_KHMER_NLTK = True
except Exception:
    _HAS_KHMER_NLTK = False

try:
    import regex as re  # pip install regex (supports \X for grapheme clusters)
except Exception:
    import re  # will work, but without \X grapheme support

try:
    from charset_normalizer import from_path as detect_charset  # pip install charset-normalizer
    _HAS_CHARSET_DETECT = True
except Exception:
    _HAS_CHARSET_DETECT = False


@dataclass
class FileDocument:
    path: str
    text: str
    metadata: Dict = field(default_factory=dict)


class KhmerAwareTxtProcessor:
    """
    .txt-only loader + chunker with Khmer-aware tokenization.

    Strategy:
      - Load text (UTF-8 first, else charset-normalizer).
      - If text seems Khmer:
          * Use khmernltk.word_tokenize if available (true words),
          * else fall back to grapheme clusters via `regex` \X,
        so chunks don't split combining marks.
      - Otherwise, default to word-based tokenization.
    """
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        normalize_whitespace: bool = True
    ):
        if chunk_size <= 0: raise ValueError("chunk_size must be > 0")
        if not (0 <= chunk_overlap < chunk_size): raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.normalize_whitespace = normalize_whitespace
        logger.info(
            f"KhmerAwareTxtProcessor(chunk_size={chunk_size}, overlap={chunk_overlap}, "
            f"normalize_whitespace={normalize_whitespace})"
        )

    # ---------- Loading ----------

    def load_files(self, paths: Iterable[str]) -> List[FileDocument]:
        docs: List[FileDocument] = []
        for path in paths:
            doc = self._load_single_txt(path)
            if doc and doc.text.strip():
                docs.append(doc)
        return docs

    def load_folder(self, folder: str, recursive: bool = True) -> List[FileDocument]:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        paths: List[str] = []
        if recursive:
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(".txt"):
                        paths.append(os.path.join(root, f))
        else:
            for f in os.listdir(folder):
                full = os.path.join(folder, f)
                if os.path.isfile(full) and f.lower().endswith(".txt"):
                    paths.append(full)
        return self.load_files(paths)

    # ---------- Chunking ----------

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        if not text:
            return []

        if self.normalize_whitespace:
            text = self._normalize_spaces(text)

        is_khmer = self._looks_like_khmer(text)

        # Choose tokenizer
        if is_khmer and _HAS_KHMER_NLTK:
            tokens = khmernltk.word_tokenize(text)  # list of words
            join_tokens = " ".join
        elif is_khmer:
            # Grapheme clusters: avoids splitting Khmer combining marks
            # If `regex` is missing, fallback to codepoint-by-codepoint
            pattern = r"\X" if hasattr(re, "VERSION1") or r"\X" in getattr(re, "__doc__", "") else r"."
            tokens = re.findall(pattern, text)
            join_tokens = "".join
        else:
            # Non-Khmer: whitespace tokenization is fine
            tokens = text.split()
            join_tokens = " ".join

        n = len(tokens)
        if n <= self.chunk_size:
            return [{"text": join_tokens(tokens), "metadata": metadata or {}}]

        chunks: List[Dict] = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, n, step):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunks.append({"text": join_tokens(chunk_tokens), "metadata": metadata or {}})
        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks: List[Dict] = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            if text:
                all_chunks.extend(self.chunk_text(text, metadata))
        return all_chunks

    def chunk_filedocs(self, file_docs: List[FileDocument]) -> List[Dict]:
        out: List[Dict] = []
        for d in file_docs:
            meta = {"source": d.path, **(d.metadata or {})}
            out.extend(self.chunk_text(d.text, meta))
        return out

    # ---------- Internals ----------

    def _load_single_txt(self, path: str) -> Optional[FileDocument]:
        if not os.path.isfile(path):
            logger.warning(f"Not a file: {path}")
            return None
        if not path.lower().endswith(".txt"):
            logger.info(f"Skipping unsupported file type: {path}")
            return None

        text = ""
        try:
            # 1) Try UTF-8 first
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            # 2) Try charset detection if available
            if _HAS_CHARSET_DETECT:
                try:
                    res = detect_charset(path).best()
                    if res and res.encoding:
                        with open(path, "r", encoding=res.encoding, errors="ignore") as f:
                            text = f.read()
                    else:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                except Exception:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

        if self.normalize_whitespace:
            text = self._normalize_spaces(text)

        meta = {
            "filename": os.path.basename(path),
            "filesize_bytes": os.path.getsize(path),
            "modified_time": self._safe_mtime_iso(path),
            "filetype": "txt",
        }
        return FileDocument(path=path, text=text, metadata=meta)

    def _normalize_spaces(self, text: str) -> str:
        # Keep line breaks, collapse excessive spaces; don’t ruin Khmer combining sequences
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _safe_mtime_iso(self, path: str) -> Optional[str]:
        try:
            ts = os.path.getmtime(path)
            from datetime import datetime
            return datetime.fromtimestamp(ts).isoformat()
        except Exception:
            return None

    def _looks_like_khmer(self, text: str, threshold: float = 0.3) -> bool:
        # Count codepoints in Khmer blocks: U+1780–U+17FF, U+19E0–U+19FF
        khmer = sum(1 for ch in text if (0x1780 <= ord(ch) <= 0x17FF) or (0x19E0 <= ord(ch) <= 0x19FF))
        letters = sum(1 for ch in text if ch.isalpha())
        total = letters if letters > 0 else max(len(text), 1)
        return (khmer / total) >= threshold


# ---------- Example ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    p = KhmerAwareTxtProcessor(chunk_size=500, chunk_overlap=50)

    # A) Specific files
    files = [r"D:\Seing Ratana Notebook\Chatbot Project\Dataset\Data.txt"]
    docs = p.load_files(files)
    chunks = p.chunk_filedocs(docs)
    logger.info(f"Loaded {len(docs)} docs; created {len(chunks)} chunks.")
