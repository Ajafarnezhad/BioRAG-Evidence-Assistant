# BioRAG Evidence Assistant

> Existing RAG systems hallucinate on long documents. I investigated whether chunking strategy is a root cause and benchmarked four approaches across retrieval quality, latency, throughput, and answer accuracy.

**Author:** Amirhossein Jafarnezhad

## Overview

Retrieval-Augmented Generation (RAG) systems often fail on long documents despite using powerful LLMs and high-quality embeddings.

A common assumption is that hallucinations originate from the language model itself. In practice, many failures occur much earlier in the pipeline when relevant context is fragmented during chunking and never retrieved.

This project benchmarks multiple chunking strategies to understand their impact on retrieval performance and downstream answer quality.

## Problem Statement

Given the same:

* Document
* Embedding model
* Vector database
* Retrieval logic
* LLM

How much does chunking strategy alone affect RAG performance?

## Chunking Strategies Evaluated

### 1. Fixed-Size Chunking

Splits text into fixed-length character windows.

Pros:

* Simple
* Fast

Cons:

* Breaks semantic boundaries
* High context fragmentation

---

### 2. Sentence-Aware Chunking

Splits content using sentence boundaries.

Pros:

* Preserves sentence integrity

Cons:

* May separate related paragraphs

---

### 3. Paragraph-Aware Chunking

Splits content using paragraph boundaries.

Pros:

* Better local context preservation

Cons:

* Inconsistent chunk sizes

---

### 4. Recursive Chunking

Hierarchical splitting:

```text
Paragraph
    ↓
Sentence
    ↓
Word
```

Pros:

* Preserves semantic structure
* Reduces context fragmentation

Cons:

* More complex implementation

---

## Architecture

```text
User Upload
      │
      ▼
Document Processor
      │
      ├── Fixed Chunking
      ├── Sentence Chunking
      ├── Paragraph Chunking
      └── Recursive Chunking
      │
      ▼
4 Independent Chroma Collections
      │
      ▼
Retriever
      │
      ▼
LLM
      │
      ▼
Benchmark Results
```

Each chunking strategy is indexed independently to ensure fair comparison.

## Tech Stack

* Python
* FastAPI
* ChromaDB
* Sentence Transformers
* OpenAI
* NumPy

## Features

* PDF / DOCX / TXT ingestion
* Multi-strategy chunk generation
* Independent vector indexes
* Side-by-side answer comparison
* Benchmarking framework
* Retrieval latency measurement
* Throughput estimation
* Evaluation dataset support

## Benchmark Metrics

### Retrieval Latency

Measures retrieval performance.

* p50 latency
* p95 latency

### Answer Accuracy

Measures correctness against a manually created evaluation dataset.

### Cost Per Query

Measures inference cost.

### Throughput

Measures estimated queries processed per second.

### Retrieval Coverage

Measures whether relevant information was successfully retrieved.

## Dataset

Evaluation performed on:

* Infosys Integrated Annual Report 2024–25
* 400+ pages
* Long-form enterprise document

## Sample Results

| Strategy  | Accuracy | Retrieval Coverage |
| --------- | -------- | ------------------ |
| Fixed     | 18%      | 58%                |
| Sentence  | 12%      | 60%                |
| Paragraph | 16%      | 66%                |
| Recursive | 18%      | 68%                |

### Key Finding

Recursive chunking achieved the highest retrieval coverage while maintaining comparable retrieval latency.

Several retrieval failures observed in fixed-size chunking were caused by context fragmentation, resulting in retrieval misses and downstream hallucinations.

## Example Failure

Question:

```text
How many active clients does Infosys have?
```

Fixed-Size Chunking:

```text
I don't have enough information.
```

Recursive Chunking:

```text
Infosys has 1,869 active clients.
```

Question:

```text
What were total revenues in FY25?
```

Fixed-Size Chunking:

```text
₹5,669 crore
```

Recursive Chunking:

```text
₹1,62,990 crore
```

These examples demonstrate how retrieval failures can propagate into incorrect answers.

## Running Locally

### Clone Repository

```bash
git clone <repository-url>
cd BioRAG-Evidence-Assistant
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

### Start API

```bash
uvicorn main:app --reload
```

## Upload Document

```bash
curl -X POST \
"http://localhost:8000/upload" \
-F "file=@annual_report.pdf"
```

## Query

```bash
curl -X POST \
"http://localhost:8000/query" \
-H "Content-Type: application/json" \
-d '{
  "question":"Who is the CEO of Infosys?"
}'
```

## Benchmark

```bash
curl http://localhost:8000/benchmark
```

Returns:

```json
{
  "fixed": {},
  "sentence": {},
  "paragraph": {},
  "recursive": {}
}
```

## Engineering Tradeoffs

### Chosen

* Multiple vector collections for fair benchmarking
* Chunk overlap to reduce boundary information loss
* Recursive chunking for improved context preservation

### Rejected

* Large chunk sizes (>1500 chars)
* No-overlap chunking
* Single shared vector collection across strategies

## What I Would Improve Next

* Semantic chunking using embeddings
* Hybrid retrieval (BM25 + Dense Retrieval)
* Re-ranking
* Recall@K evaluation dataset
* Distributed vector databases (Qdrant / Weaviate)

## Key Takeaway

Many RAG hallucinations are retrieval problems before they become LLM problems.

The model can only answer from the context it receives. If chunking breaks the context, answer quality suffers regardless of model size.