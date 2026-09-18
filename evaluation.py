"""
RAG Chunking Strategy Evaluation

Benchmarks:
- Recall@K
- Retrieval Latency (p50, p95)
- Cost Per Query
- Throughput (QPS)
"""

import time
from typing import Dict, List, Optional
import numpy as np


class ChunkingEvaluator:

    def __init__(self):

        self.results = {
            "fixed": self._empty_metrics(),
            "sentence": self._empty_metrics(),
            "paragraph": self._empty_metrics(),
            "recursive": self._empty_metrics()
        }

    def _empty_metrics(self):

        return {
            "recall_scores": [],
            "retrieval_latencies_ms": [],
            "answer_latencies_ms": [],
            "answer_accuracies": [],
            "query_costs": [],
            "throughput_qps": []
        }

    # --------------------------------------------------
    # Recall@K
    # --------------------------------------------------

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        ground_truth_ids: List[str],
        k: int = 5
    ) -> float:

        retrieved = set(retrieved_ids[:k])
        ground_truth = set(ground_truth_ids)

        if not ground_truth:
            return 0.0

        hits = len(
            retrieved.intersection(
                ground_truth
            )
        )

        return hits / len(ground_truth)

    # --------------------------------------------------
    # Cost
    # --------------------------------------------------

    @staticmethod
    def calculate_query_cost(
        prompt_tokens: int,
        completion_tokens: int,
        input_price_per_1k: float,
        output_price_per_1k: float
    ) -> float:

        input_cost = (
            prompt_tokens / 1000
        ) * input_price_per_1k

        output_cost = (
            completion_tokens / 1000
        ) * output_price_per_1k

        return input_cost + output_cost

    # --------------------------------------------------
    # Answer Accuracy
    # --------------------------------------------------

    @staticmethod
    def calculate_answer_accuracy(
        expected_answer: Optional[str],
        answer: str
    ) -> Optional[float]:

        if not expected_answer:
            return None

        return 1.0 if expected_answer.lower() in answer.lower() else 0.0

    # --------------------------------------------------
    # Record Query Result
    # --------------------------------------------------

    def record_query(
        self,
        strategy: str,
        retrieved_ids: List[str],
        ground_truth_ids: Optional[List[str]],
        retrieval_latency_ms: float,
        answer_latency_ms: float,
        query_cost: float,
        throughput_qps: float,
        answer: str,
        expected_answer: Optional[str],
        k: int = 5
    ):

        if ground_truth_ids:
            recall = self.recall_at_k(
                retrieved_ids,
                ground_truth_ids,
                k
            )

            self.results[strategy][
                "recall_scores"
            ].append(recall)

        self.results[strategy][
            "retrieval_latencies_ms"
        ].append(retrieval_latency_ms)

        self.results[strategy][
            "answer_latencies_ms"
        ].append(answer_latency_ms)

        accuracy = self.calculate_answer_accuracy(
            expected_answer,
            answer
        )

        if accuracy is not None:
            self.results[strategy][
                "answer_accuracies"
            ].append(accuracy)

        self.results[strategy][
            "query_costs"
        ].append(query_cost)

        self.results[strategy][
            "throughput_qps"
        ].append(throughput_qps)

    # --------------------------------------------------
    # Measure Retrieval Latency
    # --------------------------------------------------

    @staticmethod
    def timed_retrieval(
        retriever,
        query: str,
        k: int = 5
    ):

        start = time.perf_counter()

        results = retriever.search(
            query,
            n_results=k
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        return results, latency_ms

    # --------------------------------------------------
    # Measure Throughput
    # --------------------------------------------------

    @staticmethod
    def calculate_qps(
        total_queries: int,
        total_time_seconds: float
    ) -> float:

        if total_time_seconds == 0:
            return 0

        return (
            total_queries /
            total_time_seconds
        )

    # --------------------------------------------------
    # Aggregate Metrics
    # --------------------------------------------------

    def generate_report(self) -> Dict:

        report = {}

        for strategy, metrics in self.results.items():

            recalls = metrics["recall_scores"]
            retrieval_latencies = metrics["retrieval_latencies_ms"]
            answer_latencies = metrics["answer_latencies_ms"]
            accuracies = metrics["answer_accuracies"]
            costs = metrics["query_costs"]
            qps = metrics["throughput_qps"]

            report[strategy] = {

                "accuracy": round(
                    np.mean(accuracies),
                    4
                ) if accuracies else None,

                "recall_at_k": round(
                    np.mean(recalls),
                    4
                ) if recalls else None,

                "retrieval_latency_ms": {

                    "p50": round(
                        np.percentile(
                            retrieval_latencies,
                            50
                        ),
                        2
                    ) if retrieval_latencies else 0,

                    "p95": round(
                        np.percentile(
                            retrieval_latencies,
                            95
                        ),
                        2
                    ) if retrieval_latencies else 0
                },

                "answer_latency_ms": {

                    "p50": round(
                        np.percentile(
                            answer_latencies,
                            50
                        ),
                        2
                    ) if answer_latencies else 0,

                    "p95": round(
                        np.percentile(
                            answer_latencies,
                            95
                        ),
                        2
                    ) if answer_latencies else 0
                },

                "cost_per_query": round(
                    np.mean(costs),
                    6
                ) if costs else 0,

                "throughput_qps": round(
                    np.mean(qps),
                    2
                ) if qps else 0
            }

        return report