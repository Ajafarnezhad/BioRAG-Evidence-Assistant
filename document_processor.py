"""
Production-ready Document Processor for RAG Systems

Features:
- PDF / TXT / DOCX support
- Page-level extraction
- Smart chunking with overlap
- Metadata tracking
- Text cleaning
"""

import os
import uuid
import logging
import re
from datetime import datetime
from typing import List, Dict, Literal

import PyPDF2
import docx

logger = logging.getLogger(__name__)

ChunkStrategy = Literal[
    "fixed",
    "sentence",
    "paragraph",
    "recursive"
]

class DocumentProcessor:
    """
    Document processor responsible for:
    - Loading documents
    - Extracting text
    - Chunking
    - Attaching metadata
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # --------------------------------------------------
    # PDF Loader
    # --------------------------------------------------

    def load_pdf(self, file_path: str) -> List[Dict]:
        pages = []

        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()

                    if text:
                        pages.append({
                            "text": text,
                            "page": page_num + 1
                        })

        except Exception as e:
            logger.error(f"PDF load error: {file_path} | {str(e)}")

        return pages

    # --------------------------------------------------
    # TXT Loader
    # --------------------------------------------------

    def load_txt(self, file_path: str) -> List[Dict]:

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            return [{"text": text, "page": 1}]

        except Exception as e:
            logger.error(f"TXT load error: {file_path} | {str(e)}")
            return []

    # --------------------------------------------------
    # DOCX Loader
    # --------------------------------------------------

    def load_docx(self, file_path: str) -> List[Dict]:

        try:
            document = docx.Document(file_path)

            full_text = []
            for para in document.paragraphs:
                full_text.append(para.text)

            text = "\n".join(full_text)

            return [{"text": text, "page": 1}]

        except Exception as e:
            logger.error(f"DOCX load error: {file_path} | {str(e)}")
            return []

    # --------------------------------------------------
    # Single Document Loader
    # --------------------------------------------------

    def load_document(self, file_path: str) -> List[Dict]:

        filename = os.path.basename(file_path)

        if filename.endswith(".pdf"):
            pages = self.load_pdf(file_path)

        elif filename.endswith(".txt"):
            pages = self.load_txt(file_path)

        elif filename.endswith(".docx"):
            pages = self.load_docx(file_path)

        else:
            logger.warning(f"Unsupported file type: {filename}")
            return []

        return [
            {
                "text": self.clean_text(page["text"]),
                "page": page["page"],
                "source": filename
            }
            for page in pages
        ]

    # --------------------------------------------------
    # Text Cleaning
    # --------------------------------------------------

    def clean_text(self, text: str) -> str:

        text = text.replace("\t", " ")

        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ ]+', ' ', text)

        return text.strip()
    
    
    # --------------------------------------------------
    # Chunking
    # --------------------------------------------------

    # Fixed size chunking strategy
    def fixed_size_chunking(self, text: str, apply_overlap: bool = True) -> List[str]:

        chunks = []
        start = 0

        while start < len(text):

            end = start + self.chunk_size
            chunks.append(text[start:end].strip())

            start = end

        return self.apply_overlap(chunks) if apply_overlap else chunks

    # Sentence boundary chunking strategy
    def sentence_chunking(self, text: str) -> List[str]:

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        chunks = []
        current = ""

        for sentence in sentences:

            if len(sentence) > self.chunk_size:

                if current:
                    chunks.append(current.strip())
                    current = ""

                chunks.extend(
                    self.fixed_size_chunking(sentence, apply_overlap=False)
                )

                continue

            if len(current) + len(sentence) <= self.chunk_size:
                current += " " + sentence

            else:
                chunks.append(current.strip())
                current = sentence

        if current:
            chunks.append(current.strip())

        return self.apply_overlap(chunks)

    # Paragraph boundary chunking strategy
    def paragraph_chunking(self, text: str) -> List[str]:

        paragraphs = text.split("\n")

        chunks = []
        current = ""

        for para in paragraphs:

            para = para.strip()

            if not para:
                continue

            if len(para) > self.chunk_size:

                if current:
                    chunks.append(current.strip())
                    current = ""

                chunks.extend(
                    self.sentence_chunking(para)
                )

                continue

            if len(current) + len(para) <= self.chunk_size:
                current += "\n" + para

            else:
                chunks.append(current.strip())
                current = para

        if current:
            chunks.append(current.strip())

        return self.apply_overlap(chunks)

    # Recursive chunking strategy
    def recursive_chunking(self, text: str):

        separators = [
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]

        chunks = self._recursive_split(
            text,
            separators
        )

        return self.apply_overlap(chunks)

    # Recursive splitting helper for semantic chunking
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:

        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        if not separators:
            return [
                text[i:i+self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        separator = separators[0]

        if separator == "":
            return [
                text[i:i+self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        pieces = text.split(separator)

        chunks = []
        current = ""

        for piece in pieces:

            candidate = (
                current + separator + piece
                if current else piece
            )

            if len(candidate) <= self.chunk_size:
                current = candidate

            else:

                if current:
                    chunks.append(current.strip())

                if len(piece) > self.chunk_size:
                    chunks.extend(
                        self._recursive_split(
                            piece,
                            separators[1:]
                        )
                    )
                    current = ""

                else:
                    current = piece

        if current:
            chunks.append(current.strip())

        return chunks

    def chunk_text(self, text: str, strategy: ChunkStrategy = "fixed") -> List[str]:

        if strategy == "fixed":
            return self.fixed_size_chunking(text)

        if strategy == "sentence":
            return self.sentence_chunking(text)

        if strategy == "paragraph":
            return self.paragraph_chunking(text)

        if strategy == "recursive":
            return self.recursive_chunking(text)

        raise ValueError(f"Unknown strategy: {strategy}")
    
    def apply_overlap(
        self,
        chunks: List[str]
    ) -> List[str]:

        if not chunks or self.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):

            prev = chunks[i - 1]
            current = chunks[i]

            overlap = prev[-min(self.chunk_overlap, len(prev)):]

            overlapped.append(
                overlap + " " + current
            )

        return overlapped

    # --------------------------------------------------
    # Load Documents
    # --------------------------------------------------

    def load_documents(self, directory: str) -> List[Dict]:

        documents = []

        if not os.path.exists(directory):
            logger.warning(f"Directory does not exist: {directory}")
            return documents

        for filename in os.listdir(directory):

            path = os.path.join(directory, filename)

            if not filename.endswith((".pdf", ".txt", ".docx")):
                continue

            documents.extend(self.load_document(path))

            logger.info(f"Loaded document: {filename}")

        return documents

    # --------------------------------------------------
    # Process Documents
    # --------------------------------------------------

    

    def process_documents_all_strategies(self, documents: List[Dict]) -> Dict[str, List[Dict]]:

        strategies = [
            "fixed",
            "sentence",
            "paragraph",
            "recursive"
        ]

        results = {
            strategy: []
            for strategy in strategies
        }

        for doc in documents:

            doc_id = str(uuid.uuid4())

            for strategy in strategies:

                chunks = self.chunk_text(
                    doc["text"],
                    strategy=strategy
                )

                for i, chunk in enumerate(chunks):

                    chunk_uid = f"{strategy}:{doc_id}:{doc['page']}:{i}"

                    results[strategy].append({

                        "content": chunk,

                        "source": doc["source"],

                        "metadata": {
                            "strategy": strategy,
                            "chunk_length": len(chunk),
                            "doc_id": doc_id,
                            "chunk_id": chunk_uid,
                            "chunk_index": i,
                            "page": doc["page"],
                            "source": doc["source"],
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })


        benchmark_stats = {}

        for strategy in strategies:

            chunks = results[strategy]

            sizes = [
                len(c["content"])
                for c in chunks
            ]

            benchmark_stats[strategy] = {
                "documents_processed": len(documents),
                "total_chunks": len(chunks),
                "avg_chunks_per_page": (
                    len(chunks) / len(documents)
                    if documents else 0
                ),
                "avg_chunk_size": (
                    sum(sizes) / len(sizes)
                    if sizes else 0
                ),
                "max_chunk_size": (
                    max(sizes)
                    if sizes else 0
                ),
                "min_chunk_size": min(sizes) if sizes else 0,
                "total_characters": sum(sizes)
            }

        logger.info(
            f"Chunk Benchmark Stats: {benchmark_stats}"
        )

        return {
            "chunks": results,
            "stats": benchmark_stats
        }


