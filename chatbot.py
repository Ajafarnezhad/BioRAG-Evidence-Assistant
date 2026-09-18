"""
Production-Ready RAG Chatbot
OpenAI + Controlled Prompting + Confidence Scoring
"""

import os
import time
import logging
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    Production-oriented RAG chatbot with:
    - Controlled prompting
    - Source citation enforcement
    - Confidence scoring
    - Optional conversation memory
    """

    def __init__(self, vector_store, api_key: Optional[str] = None):
        load_dotenv()

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

        self.model = "gpt-4o-mini"

        self.vector_store = vector_store
        self.conversation_history: List[Dict] = []

    # --------------------------------------------------
    # Context Builder
    # --------------------------------------------------

    def build_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Construct strict context block with numbered sources.
        """
        context_parts = []

        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc["metadata"].get("source", "Unknown")
            chunk_id = doc["metadata"].get("chunk_id", "N/A")

            context_parts.append(
                f"[Source {i}] (File: {source}, Chunk: {chunk_id})\n"
                f"{doc['content']}\n"
            )

        return "\n".join(context_parts)

    # --------------------------------------------------
    # Prompt Builder
    # --------------------------------------------------

    def build_prompt(
        self,
        question: str,
        context: str,
        include_history: bool = False
    ) -> str:
        """
        Strongly constrained prompt to reduce hallucination.
        """

        history_block = ""

        if include_history and self.conversation_history:
            history_entries = self.conversation_history[-3:]
            history_block = "\n\nPREVIOUS CONVERSATION:\n"

            for entry in history_entries:
                history_block += f"Q: {entry['question']}\n"
                history_block += f"A: {entry['answer']}\n\n"

        prompt = f"""
You are a retrieval-based AI assistant.

You MUST follow these rules strictly:

1. Answer ONLY using the provided context.
2. If the answer is not explicitly supported by context, respond:
   "I don't have enough information in the provided documents."
3. Cite sources using format: [Source X]
4. Do NOT fabricate.
5. Do NOT infer beyond the context.

{history_block}

CONTEXT:
{context}

QUESTION:
{question}

Provide your response in this format:

Answer:
<your answer here>

Sources Used:
[List of source numbers]

Confidence:
High / Medium / Low
"""

        return prompt.strip()

    # --------------------------------------------------
    # Confidence Estimation
    # --------------------------------------------------

    def compute_confidence(
        self,
        retrieved_docs: List[Dict]
    ) -> str:
        """
        Simple heuristic confidence scoring based on
        retrieval similarity distance.
        """

        distances = [
            doc.get("distance")
            for doc in retrieved_docs
            if doc.get("distance") is not None
        ]

        if not distances:
            return "Low"

        avg_distance = sum(distances) / len(distances)

        # Lower cosine distance = better match
        if avg_distance < 0.2:
            return "High"
        elif avg_distance < 0.5:
            return "Medium"
        else:
            return "Low"

    # --------------------------------------------------
    # Chat Method
    # --------------------------------------------------

    def chat(
        self,
        question: str,
        n_results: int = 5,
        include_history: bool = False,
        store_history: bool = True
    ) -> Dict:

        if not question.strip():
            return {
                "answer": "Question cannot be empty.",
                "sources": [],
                "confidence": "Low",
                "latency_sec": 0
            }

        start_time = time.time()

        # 1️⃣ Retrieve relevant chunks
        retrieved_docs = self.vector_store.search(
            question,
            n_results=n_results
        )

        if not retrieved_docs:
            return {
                "answer": "I don't have enough information in the provided documents.",
                "sources": [],
                "confidence": "Low",
                "latency_sec": 0
            }

        # 2️⃣ Build context
        context = self.build_context(retrieved_docs)

        # 3️⃣ Build prompt
        prompt = self.build_prompt(
            question,
            context,
            include_history=include_history
        )

        # 4️⃣ Call AI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_answer = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            usage_data = None

            if usage is not None:
                usage_data = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0
                }
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            return {
                "answer": "Error generating response.",
                "sources": [],
                "confidence": "Low",
                "latency_sec": round(time.time() - start_time, 2),
                "usage": None
            }

        # 5️⃣ Extract sources used
        unique_sources = list({
            doc["metadata"].get("source")
            for doc in retrieved_docs
        })

        # 6️⃣ Compute confidence
        confidence = self.compute_confidence(retrieved_docs)

        # 7️⃣ Store history
        if store_history:
            self.conversation_history.append({
                "question": question,
                "answer": raw_answer,
                "sources": unique_sources
            })

        latency = round(time.time() - start_time, 2)

        return {
            "answer": raw_answer,
            "sources": unique_sources,
            "confidence": confidence,
            "latency_sec": latency,
            "retrieved_docs": retrieved_docs,
            "usage": usage_data if 'usage_data' in locals() else None
        }

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def clear_history(self):
        self.conversation_history = []

    def get_history(self) -> List[Dict]:
        return self.conversation_history