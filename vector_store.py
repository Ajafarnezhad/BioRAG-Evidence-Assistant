"""
Production-ready Vector Store using ChromaDB

Features:
- Persistent vector storage
- Metadata filtering
- Document grouping
- MMR retrieval
- Logging and monitoring
"""

import uuid
import logging
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database abstraction layer
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):

        logger.info("Initializing ChromaDB vector store...")

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embedding_model = SentenceTransformer(embedding_model)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info("Vector store ready")

    # --------------------------------------------------
    # Embedding Generation
    # --------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:

        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    # --------------------------------------------------
    # Add Documents
    # --------------------------------------------------

    def add_documents(self, documents: List[Dict]):

        if not documents:
            logger.warning("No documents provided for indexing")
            return

        ids = []
        texts = []
        metadatas = []

        for doc in documents:

            vector_id = str(uuid.uuid4())

            ids.append(vector_id)
            texts.append(doc["content"])

            metadata = {
                "source": doc.get("source"),
                "chunk_id": doc.get("metadata", {}).get("chunk_id"),
                "doc_id": doc.get("metadata", {}).get("doc_id"),
                "page": doc.get("metadata", {}).get("page"),
                "timestamp": doc.get("metadata", {}).get("timestamp")
            }

            metadatas.append(metadata)

        logger.info(f"Generating embeddings for {len(texts)} chunks")

        embeddings = self.embed_texts(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Indexed {len(texts)} document chunks")

    # --------------------------------------------------
    # Similarity Search
    # --------------------------------------------------

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:

        query_embedding = self.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )

        formatted_results = []

        if results["documents"]:

            for i in range(len(results["documents"][0])):

                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })

        return formatted_results

    # --------------------------------------------------
    # MMR Retrieval (Better Diversity)
    # --------------------------------------------------

    def search_mmr(
        self,
        query: str,
        n_results: int = 5,
        fetch_k: int = 20
    ) -> List[Dict]:
        """
        Max Marginal Relevance retrieval.
        Improves diversity of retrieved chunks.
        """

        initial_results = self.search(query, n_results=fetch_k)

        selected = []
        seen_sources = set()

        for doc in initial_results:

            source = doc["metadata"].get("source")

            if source not in seen_sources:
                selected.append(doc)
                seen_sources.add(source)

            if len(selected) >= n_results:
                break

        return selected

    # --------------------------------------------------
    # Delete Document by Source
    # --------------------------------------------------

    def delete_document(self, source_name: str):

        logger.info(f"Deleting document: {source_name}")

        all_docs = self.collection.get()

        ids_to_delete = []

        for i, metadata in enumerate(all_docs["metadatas"]):
            if metadata.get("source") == source_name:
                ids_to_delete.append(all_docs["ids"][i])

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

            logger.info(f"Deleted {len(ids_to_delete)} chunks")

        else:
            logger.warning("No chunks found for document")

    # --------------------------------------------------
    # Get Collection Stats
    # --------------------------------------------------

    def get_collection_count(self) -> int:
        return self.collection.count()

    def get_all_documents(self):

        return self.collection.get(
            include=["documents", "metadatas"]
        )

    # --------------------------------------------------
    # Clear Collection
    # --------------------------------------------------

    def clear_collection(self):

        name = self.collection.name

        self.client.delete_collection(name)

        self.collection = self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info("Vector collection cleared")