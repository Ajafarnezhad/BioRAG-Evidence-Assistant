# ARCHITECTURE.md

# BioRAG Evidence Assistant Architecture

## Overview

This project investigates a common but often overlooked cause of RAG hallucinations:

> Poor chunking strategies can fragment context, leading to retrieval failures and ultimately incorrect answers.

The goal is to isolate chunking as the experimental variable and measure its impact on retrieval quality, latency, throughput, and answer accuracy.

---

# System Architecture

```text
                    ┌─────────────────┐
                    │  User Uploads   │
                    │ PDF / DOCX /TXT │
                    └────────┬────────┘
                             │
                             ▼
                 ┌─────────────────────┐
                 │ Document Processor  │
                 └────────┬────────────┘
                          │
      ┌───────────────────┼───────────────────┐
      │                   │                   │
      ▼                   ▼                   ▼
 Fixed Chunking    Sentence Chunking   Paragraph Chunking
      │                   │                   │
      └──────────────┬────┴──────────────┬────┘
                     │                   │
                     ▼                   ▼
             Recursive Chunking     Metadata
                     │
                     ▼
     ┌─────────────────────────────────────┐
     │  Strategy-Specific Chunk Sets       │
     └─────────────────────────────────────┘
                     │
                     ▼
     ┌─────────────────────────────────────┐
     │ Sentence Transformer Embeddings     │
     └─────────────────────────────────────┘
                     │
                     ▼
 ┌──────────────────────────────────────────────┐
 │ Chroma Collections                           │
 │                                              │
 │ fixed_chunks                                 │
 │ sentence_chunks                              │
 │ paragraph_chunks                             │
 │ recursive_chunks                             │
 └──────────────────────────────────────────────┘
                     │
                     ▼
              Query Execution
                     │
                     ▼
     ┌─────────────────────────────────────┐
     │ Strategy-Specific Retrieval          │
     └─────────────────────────────────────┘
                     │
                     ▼
     ┌─────────────────────────────────────┐
     │ OpenAI LLM Answer Generation         │
     └─────────────────────────────────────┘
                     │
                     ▼
            Benchmark Evaluation
```

---

# Design Goal

The benchmark is designed to answer a single question:

> How much of RAG performance is determined by chunking strategy alone?

To ensure fairness, the following remain constant:

* Embedding model
* Vector database
* Retrieval method
* LLM
* Query set

Only the chunking strategy changes.

---

# Component Breakdown

## 1. Document Processor

Responsible for:

* PDF extraction
* DOCX extraction
* TXT extraction
* Text cleaning
* Metadata generation
* Chunk creation

### Output

```json
{
  "content": "...",
  "metadata": {
    "strategy": "recursive",
    "chunk_id": "...",
    "page": 42,
    "source": "annual_report.pdf"
  }
}
```

---

## 2. Chunking Layer

### Fixed-Size Chunking

```text
[500 chars]
[500 chars]
[500 chars]
```

Advantages:

* Simple
* Fast

Disadvantages:

* Breaks semantic boundaries
* High fragmentation risk

---

### Sentence-Aware Chunking

```text
Sentence 1
Sentence 2
Sentence 3
```

Advantages:

* Preserves sentence meaning

Disadvantages:

* Can separate related paragraphs

---

### Paragraph-Aware Chunking

```text
Paragraph A
Paragraph B
```

Advantages:

* Better local context

Disadvantages:

* Chunk size variance

---

### Recursive Chunking

Hierarchical splitting:

```text
Paragraph
     ↓
Sentence
     ↓
Word
```

Advantages:

* Preserves structure
* Reduces fragmentation

Disadvantages:

* More computational complexity

---

# Why Four Independent Indexes?

A common benchmarking mistake is storing all chunk outputs in a single vector database.

This introduces contamination and makes comparisons unreliable.

Instead:

```text
fixed_chunks
sentence_chunks
paragraph_chunks
recursive_chunks
```

Each strategy receives:

* Independent embeddings
* Independent retrieval pipeline
* Independent benchmark metrics

This isolates chunking as the experimental variable.

---

# Retrieval Pipeline

```text
User Question
      │
      ▼
Embedding Generation
      │
      ▼
Vector Search
      │
      ▼
Top-K Chunks
      │
      ▼
Prompt Construction
      │
      ▼
OpenAI LLM
      │
      ▼
Final Answer
```

The same query is executed against all four pipelines.

---

# Benchmarking Architecture

## Metrics

### Retrieval Coverage

Measures whether relevant information was retrieved.

```text
Retrieved Relevant Chunks
─────────────────────────
Total Relevant Chunks
```

---

### Retrieval Latency

Measured at retrieval time only.

Metrics:

* p50 latency
* p95 latency

Purpose:

Evaluate vector search efficiency.

---

### Throughput

Estimated using:

```text
QPS = Queries / Second
```

Purpose:

Evaluate scalability.

---

### Answer Accuracy

Measures answer correctness against a manually curated evaluation set.

Example:

```json
{
  "question": "Who is the CEO of Infosys?",
  "expected_answer": "Salil Parekh"
}
```

---

### Cost Per Query

Includes:

* Embedding cost
* LLM inference cost

Purpose:

Evaluate production viability.

---

# Experimental Results

## Retrieval Coverage

| Strategy  | Coverage |
| --------- | -------- |
| Fixed     | 58%      |
| Sentence  | 60%      |
| Paragraph | 66%      |
| Recursive | 68%      |

Observation:

Recursive chunking consistently retrieved more relevant context.

---

# Failure Analysis

## Active Clients Example

Question:

```text
How many active clients does Infosys have?
```

Fixed Chunking:

```text
I don't have enough information.
```

Recursive Chunking:

```text
Infosys has 1,869 active clients.
```

Root Cause:

Relevant information was split across chunk boundaries.

---

## Revenue Example

Question:

```text
What was FY25 revenue?
```

Fixed Chunking:

```text
₹5,669 crore
```

Recursive Chunking:

```text
₹1,62,990 crore
```

Root Cause:

Incorrect retrieval caused downstream hallucination.

---

# Tradeoffs Considered

## Retrieval Quality vs Cost

Larger chunks:

Advantages:

* More context

Disadvantages:

* Higher token cost
* More irrelevant information

Decision:

Use moderate chunk sizes with overlap.

---

## Simplicity vs Context Preservation

Fixed chunking:

* Easy implementation
* Poor context retention

Recursive chunking:

* Higher complexity
* Better retrieval performance

Decision:

Favor retrieval quality over implementation simplicity.

---

## Storage vs Benchmark Accuracy

Maintaining four vector collections increases storage requirements.

Benefits:

* Fair comparison
* Experimental isolation

Decision:

Accept storage overhead.

---

# What Was Tried and Rejected

## Large Chunks (>1500 characters)

Problem:

* Increased prompt size
* Reduced retrieval precision

Rejected.

---

## No Overlap

Problem:

* Information loss at chunk boundaries

Rejected.

---

## Single Shared Vector Collection

Problem:

* Benchmark contamination

Rejected.

---

# Scaling Considerations (10x Growth)

If document volume increased by 10x:

## Replace ChromaDB

Current:

```text
Single-node ChromaDB
```

Future:

* Qdrant
* Weaviate
* Pinecone

---

## Hybrid Retrieval

Current:

```text
Dense Retrieval Only
```

Future:

```text
Dense Retrieval
+
BM25
+
Re-ranking
```

---

## Semantic Chunking

Current:

```text
Recursive Chunking
```

Future:

```text
Embedding-Based Semantic Chunking
```

Expected Benefit:

Improved retrieval precision.

---

## Distributed Evaluation

Current:

```text
Single-node benchmarking
```

Future:

```text
Parallel evaluation workers
```

Expected Benefit:

Higher throughput and faster experimentation.

---

# Key Takeaway

The benchmark demonstrates that many RAG hallucinations are retrieval failures before they become LLM failures.

The quality of retrieved context often has a larger impact on answer quality than changing the underlying model.

Improving chunking strategy significantly improves retrieval effectiveness while maintaining similar latency characteristics.