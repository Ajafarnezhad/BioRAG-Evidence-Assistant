import json
import os
import sys
import time
from typing import List, Dict
from vector_store import VectorStore
from chatbot import RAGChatbot
from evaluation import ChunkingEvaluator

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("Loading evaluation dataset (50 questions)...")
    with open("eval_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    strategies = ["fixed", "sentence", "paragraph", "recursive"]
    vector_stores = {}
    chatbots = {}
    
    print("Initializing Vector Stores and Chatbots...")
    for strategy in strategies:
        vector_stores[strategy] = VectorStore(
            collection_name=f"{strategy}_chunks",
            persist_directory="./chroma_db"
        )
        chatbots[strategy] = RAGChatbot(vector_stores[strategy])
    
    evaluator = ChunkingEvaluator()
    detailed_results = {s: [] for s in strategies}
    
    print("\nRunning full QA evaluation (this may take a few minutes as it queries the LLM)...")
    
    for idx, item in enumerate(dataset):
        question = item["question"]
        expected_answer = item.get("expected_answer")
        if not expected_answer:
            continue
            
        print(f"[{idx+1}/50] Question: {question}")
        
        for strategy in strategies:
            # 1. Measure Retrieval Latency
            retrieval_start = time.perf_counter()
            retrieved_docs = vector_stores[strategy].search(question, n_results=5)
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
            
            # Check if expected answer is present in retrieved chunks (Retrieval Coverage)
            retrieved_chunks_text = [doc["content"] for doc in retrieved_docs]
            retrieved_contains_answer = any(expected_answer.lower() in c.lower() for c in retrieved_chunks_text)
            
            # 2. Query Chatbot (LLM)
            answer_start = time.perf_counter()
            response = chatbots[strategy].chat(
                question=question,
                n_results=5,
                include_history=False,
                store_history=False
            )
            answer_latency_ms = (time.perf_counter() - answer_start) * 1000
            
            answer = response.get("answer", "")
            
            # 3. Calculate metrics
            usage = response.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            
            query_cost = evaluator.calculate_query_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_price_per_1k=float(os.getenv("INPUT_PRICE_PER_1K", "0")),
                output_price_per_1k=float(os.getenv("OUTPUT_PRICE_PER_1K", "0"))
            )
            
            throughput_qps = evaluator.calculate_qps(
                1,
                response.get("latency_sec", 0) if response.get("latency_sec", 0) else answer_latency_ms / 1000
            )
            
            # Record query in evaluator
            evaluator.record_query(
                strategy=strategy,
                retrieved_ids=[doc.get("metadata", {}).get("chunk_id") for doc in retrieved_docs],
                ground_truth_ids=None,
                retrieval_latency_ms=retrieval_latency_ms,
                answer_latency_ms=answer_latency_ms,
                query_cost=query_cost,
                throughput_qps=throughput_qps,
                answer=answer,
                expected_answer=expected_answer
            )
            
            # Store details for report
            accuracy = evaluator.calculate_answer_accuracy(expected_answer, answer)
            
            detailed_results[strategy].append({
                "question": question,
                "expected_answer": expected_answer,
                "retrieved_contains_answer": retrieved_contains_answer,
                "answer": answer,
                "accuracy": accuracy,
                "retrieved_chunks": retrieved_chunks_text
            })
            
    # Generate aggregated report
    report = evaluator.generate_report()
    
    # Write beautiful Markdown report
    print("\nGenerating evaluation report...")
    report_md = []
    report_md.append("# RAG Chunking Evaluation Report\n")
    report_md.append("This report summarizes the performance of 4 different chunking strategies on the 50-question evaluation dataset.\n")
    
    report_md.append("## Summary Metrics Table\n")
    report_md.append("| Strategy | QA Accuracy | Retrieval Coverage | P50 Retrieval Latency | P95 Retrieval Latency | P50 Answer Latency | P95 Answer Latency | Avg Cost |\n")
    report_md.append("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
    
    for strategy in strategies:
        strat_metrics = report[strategy]
        # Calculate retrieval coverage manually from detailed results
        successful_retrievals = sum(1 for r in detailed_results[strategy] if r["retrieved_contains_answer"])
        coverage_pct = (successful_retrievals / len(dataset)) * 100
        
        report_md.append(
            f"| **{strategy}** | {strat_metrics['accuracy']*100:.1f}% | {coverage_pct:.1f}% "
            f"| {strat_metrics['retrieval_latency_ms']['p50']:.1f}ms | {strat_metrics['retrieval_latency_ms']['p95']:.1f}ms "
            f"| {strat_metrics['answer_latency_ms']['p50']:.1f}ms | {strat_metrics['answer_latency_ms']['p95']:.1f}ms "
            f"| ${strat_metrics['cost_per_query']:.6f} |\n"
        )
        
    report_md.append("\n## Evidence of Chunking-Induced Hallucinations\n")
    report_md.append("Below are specific cases showing how poor chunking (especially `fixed`) directly causes retrieval failure or context fragmentation, forcing the LLM to hallucinate or fail.\n")
    
    evidence_idx = 1
    for idx in range(len(dataset)):
        fixed_res = detailed_results["fixed"][idx]
        recursive_res = detailed_results["recursive"][idx]
        
        # Evidence case: recursive succeeded, fixed failed
        if recursive_res["accuracy"] == 1.0 and fixed_res["accuracy"] == 0.0:
            question = fixed_res["question"]
            expected = fixed_res["expected_answer"]
            
            report_md.append(f"### Case {evidence_idx}: {question}\n")
            report_md.append(f"**Expected Answer:** `{expected}`\n\n")
            
            report_md.append("#### Fixed-Size Chunking (Failure / Hallucination)\n")
            report_md.append(f"- **Retrieved Chunks Contain Expected Answer:** {fixed_res['retrieved_contains_answer']}\n")
            report_md.append(f"- **LLM Generated Answer:**\n```\n{fixed_res['answer']}\n```\n")
            report_md.append("- **Top Retrieved Chunks:**\n")
            for c_idx, chunk in enumerate(fixed_res["retrieved_chunks"][:2]):
                report_md.append(f"  * **Chunk {c_idx+1}:** `... {chunk[:300]} ...`\n")
                
            report_md.append("\n#### Recursive Chunking (Success)\n")
            report_md.append(f"- **Retrieved Chunks Contain Expected Answer:** {recursive_res['retrieved_contains_answer']}\n")
            report_md.append(f"- **LLM Generated Answer:**\n```\n{recursive_res['answer']}\n```\n")
            report_md.append("- **Top Retrieved Chunks:**\n")
            for c_idx, chunk in enumerate(recursive_res["retrieved_chunks"][:2]):
                report_md.append(f"  * **Chunk {c_idx+1}:** `... {chunk[:300]} ...`\n")
                
            report_md.append("\n---\n")
            evidence_idx += 1
            if evidence_idx > 5:
                break
                
    with open("eval_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_md))
        
    print("\nEvaluation completed successfully! Report saved to 'eval_report.md'.")

if __name__ == "__main__":
    main()
