import json
import os
import sys
from typing import List, Dict
from vector_store import VectorStore

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("Loading evaluation dataset...")
    with open("eval_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    strategies = ["fixed", "sentence", "paragraph", "recursive"]
    vector_stores = {}
    
    print("Initializing Vector Stores...")
    for strategy in strategies:
        vector_stores[strategy] = VectorStore(
            collection_name=f"{strategy}_chunks",
            persist_directory="./chroma_db"
        )
    
    results = {s: [] for s in strategies}
    
    print("\nEvaluating retrieval coverage for 50 questions...")
    for item in dataset:
        question = item["question"]
        expected_answer = item.get("expected_answer")
        if not expected_answer:
            continue
            
        for strategy in strategies:
            # Search top 5 chunks
            retrieved = vector_stores[strategy].search(question, n_results=5)
            chunks_text = [doc["content"] for doc in retrieved]
            
            # Check if expected answer is in the retrieved chunks
            found = False
            for chunk in chunks_text:
                if expected_answer.lower() in chunk.lower():
                    found = True
                    break
            
            results[strategy].append({
                "question": question,
                "expected_answer": expected_answer,
                "found": found,
                "retrieved_chunks": chunks_text
            })
            
    # Calculate statistics
    print("\n=== RETRIEVAL COVERAGE STATISTICS ===")
    for strategy in strategies:
        strategy_results = results[strategy]
        total = len(strategy_results)
        successful = sum(1 for r in strategy_results if r["found"])
        coverage = (successful / total) * 100
        print(f"Strategy: {strategy:12} | Coverage: {coverage:.2f}% ({successful}/{total} questions)")

    # Identify evidence of chunking-induced hallucination
    # Find cases where 'recursive' or 'paragraph' succeeded, but 'fixed' failed.
    print("\n=== CONCRETE EVIDENCE OF CHUNKING-INDUCED RETRIEVAL FAILURE ===")
    evidence_count = 0
    for idx, item in enumerate(dataset):
        question = item["question"]
        expected_answer = item.get("expected_answer")
        
        fixed_found = results["fixed"][idx]["found"]
        recursive_found = results["recursive"][idx]["found"]
        
        # We are looking for cases where recursive succeeded but fixed failed
        if recursive_found and not fixed_found:
            evidence_count += 1
            print(f"\n[Evidence Case #{evidence_count}]")
            print(f"Question       : {question}")
            print(f"Expected Answer: {expected_answer}")
            
            print("\n--- Fixed Chunks retrieved (Top 3) ---")
            for i, chunk in enumerate(results["fixed"][idx]["retrieved_chunks"][:3]):
                print(f"Chunk {i+1}: ... {repr(chunk[:250])} ...")
                
            print("\n--- Recursive Chunks retrieved (Top 3) ---")
            for i, chunk in enumerate(results["recursive"][idx]["retrieved_chunks"][:3]):
                print(f"Chunk {i+1}: ... {repr(chunk[:250])} ...")
                
            if evidence_count >= 5:
                break

if __name__ == "__main__":
    main()
