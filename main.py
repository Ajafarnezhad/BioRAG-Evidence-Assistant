"""
BioRAG Evidence Assistant
FastAPI + ChromaDB + OpenAI

Author: Amirhossein Jafarnezhad
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from document_processor import DocumentProcessor
from vector_store import VectorStore
from chatbot import RAGChatbot
from evaluation import ChunkingEvaluator

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Environment Setup
# --------------------------------------------------

load_dotenv()

DOCUMENTS_DIR = "./documents"
CHROMA_DIR = "./chroma_db"
CHUNKING_STRATEGIES = ("fixed", "sentence", "paragraph", "recursive")
BENCHMARK_QUERIES = [
    "What are the key points in the document?",
    "Summarize the document.",
    "What are the main requirements or findings?",
    "List the important details mentioned.",
]
EVAL_DATASET_PATH = Path("./eval_dataset.json")

INPUT_PRICE_PER_1K = float(os.getenv("INPUT_PRICE_PER_1K", "0"))
OUTPUT_PRICE_PER_1K = float(os.getenv("OUTPUT_PRICE_PER_1K", "0"))

os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="BioRAG Evidence Assistant",
    version="1.0.0"
)

# --------------------------------------------------
# Global Components (Loaded Once)
# --------------------------------------------------

doc_processor: Optional[DocumentProcessor] = None
vector_stores: Dict[str, VectorStore] = {}
chatbots: Dict[str, RAGChatbot] = {}
latest_chunk_stats: Dict[str, Dict] = {}


# --------------------------------------------------
# Request Schemas
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    include_history: bool = False


class ReloadRequest(BaseModel):
    clear_existing: bool = False


# --------------------------------------------------
# Startup Event
# --------------------------------------------------

@app.on_event("startup")
def startup_event():
    global doc_processor, vector_stores, chatbots

    logger.info("Initializing RAG system...")

    doc_processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=100
    )

    vector_stores = {
        strategy: VectorStore(
            collection_name=f"{strategy}_chunks",
            persist_directory=CHROMA_DIR
        )
        for strategy in CHUNKING_STRATEGIES
    }

    chatbots = {
        strategy: RAGChatbot(vector_stores[strategy])
        for strategy in CHUNKING_STRATEGIES
    }

    logger.info("System initialized successfully")


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "documents_indexed": {
            strategy: vector_stores[strategy].get_collection_count()
            for strategy in CHUNKING_STRATEGIES
        }
    }


def _load_uploaded_document(file_path: str):
    return doc_processor.load_document(file_path)


def _load_eval_dataset() -> List[Dict]:
    if EVAL_DATASET_PATH.exists():
        with EVAL_DATASET_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

    return [
        {"question": question, "expected_answer": None}
        for question in BENCHMARK_QUERIES
    ]


# --------------------------------------------------
# Upload Document
# --------------------------------------------------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    start_time = time.time()

    try:
        file_path = os.path.join(DOCUMENTS_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        logger.info(f"Uploaded file: {file.filename}")

        documents = _load_uploaded_document(file_path)

        if not documents:
            raise HTTPException(status_code=400, detail="No supported text could be extracted from the uploaded file")

        processed = doc_processor.process_documents_all_strategies(documents)
        latest_chunk_stats.update(processed["stats"])

        for strategy in CHUNKING_STRATEGIES:
            vector_stores[strategy].delete_document(file.filename)
            vector_stores[strategy].add_documents(processed["chunks"][strategy])

        duration = round(time.time() - start_time, 2)

        return JSONResponse({
            "message": "Document uploaded and indexed successfully",
            "chunks_created": {
                strategy: len(processed["chunks"][strategy])
                for strategy in CHUNKING_STRATEGIES
            },
            "chunk_stats": processed["stats"],
            "processing_time_sec": duration
        })

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# Query Endpoint
# --------------------------------------------------

@app.post("/query")
async def query_rag(request: QueryRequest):
    start_time = time.time()

    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        results = {}

        for strategy in CHUNKING_STRATEGIES:
            response = chatbots[strategy].chat(
                question=request.question,
                n_results=request.top_k,
                include_history=request.include_history
            )

            results[strategy] = {
                "answer": response["answer"],
                "confidence": response["confidence"],
                "latency_sec": response["latency_sec"],
                "latency_ms": round(response["latency_sec"] * 1000, 2),
                "sources": response["sources"]
            }

        duration = round(time.time() - start_time, 2)

        return JSONResponse({
            "question": request.question,
            "results": results,
            "latency_sec": duration
        })

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# Clear Collection
# --------------------------------------------------

@app.post("/reload")
async def reload_documents(request: ReloadRequest):
    try:
        if request.clear_existing:
            for strategy in CHUNKING_STRATEGIES:
                vector_stores[strategy].clear_collection()
                chatbots[strategy].clear_history()

            latest_chunk_stats.clear()
            return {"message": "Collection cleared successfully"}

        return {"message": "Nothing changed"}

    except Exception as e:
        logger.error(f"Reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# List Documents Count
# --------------------------------------------------

@app.get("/documents")
async def get_documents():
    try:
        return {
            "indexed_documents_count": {
                strategy: vector_stores[strategy].get_collection_count()
                for strategy in CHUNKING_STRATEGIES
            },
            "latest_chunk_stats": latest_chunk_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# Benchmark Endpoint
# --------------------------------------------------

@app.get("/benchmark")
async def benchmark():
    try:
        report = {}
        eval_dataset = _load_eval_dataset()

        for strategy in CHUNKING_STRATEGIES:
            evaluator = ChunkingEvaluator()

            for item in eval_dataset:
                query = item["question"]
                expected_answer = item.get("expected_answer")

                retrieval_start = time.perf_counter()
                retrieved_docs = vector_stores[strategy].search(query, n_results=5)
                retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

                answer_start = time.perf_counter()
                response = chatbots[strategy].chat(
                    question=query,
                    n_results=5,
                    include_history=False,
                    store_history=False
                )
                answer_latency_ms = (time.perf_counter() - answer_start) * 1000

                usage = response.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                query_cost = evaluator.calculate_query_cost(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    input_price_per_1k=INPUT_PRICE_PER_1K,
                    output_price_per_1k=OUTPUT_PRICE_PER_1K
                )
                throughput_qps = evaluator.calculate_qps(
                    1,
                    response.get("latency_sec", 0) if response.get("latency_sec", 0) else answer_latency_ms / 1000
                )

                evaluator.record_query(
                    strategy=strategy,
                    retrieved_ids=[
                        doc.get("metadata", {}).get("chunk_id")
                        for doc in retrieved_docs
                    ],
                    ground_truth_ids=None,
                    retrieval_latency_ms=retrieval_latency_ms,
                    answer_latency_ms=answer_latency_ms,
                    query_cost=query_cost,
                    throughput_qps=throughput_qps,
                    answer=response.get("answer", ""),
                    expected_answer=expected_answer
                )

            report[strategy] = evaluator.generate_report()[strategy]
            report[strategy]["query_count"] = len(eval_dataset)

        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )