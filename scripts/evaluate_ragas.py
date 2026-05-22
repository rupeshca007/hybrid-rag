import os
import json
from pathlib import Path
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings
from src.generation.chain import run_rag_chain

console = Console()

def run_evaluation(dataset_path: str = "data/golden_dataset.json", output_csv: str = "data/evaluation_results.csv"):
    console.print(f"[bold cyan]▶ Loading Golden Dataset from {dataset_path}[/bold cyan]")
    
    if not os.path.exists(dataset_path):
        console.print(f"[red]✗ Dataset {dataset_path} not found. Run generate_dataset.py first.[/red]")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)[:3]
        
    console.print(f"Loaded {len(golden_data)} golden questions. Running RAG system...")
    
    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    # 1. Generate answers using our RAG pipeline
    for i, item in enumerate(golden_data):
        console.print(f"  Testing Q{i+1}: {item['question']}")
        
        # Call our chain
        res = run_rag_chain(item["question"], chapter_filter=item["metadata"].get("chapter"))
        
        evaluation_data["question"].append(item["question"])
        evaluation_data["ground_truth"].append(item["ground_truth"])
        # res is a dataclass RAGResult, so access it via dot notation
        evaluation_data["answer"].append(res.answer)
        
        # Extract page_content from retrieved sources for RAGAS context
        context_texts = []
        if res.sources:
             for source in res.sources:
                 context_texts.append(source.content_preview)
        
        evaluation_data["contexts"].append(context_texts)
        
    # 2. Setup RAGAS Evaluator
    console.print("\n[bold cyan]▶ Initializing RAGAS Evaluator...[/bold cyan]")
    eval_llm = ChatGroq(model=settings.groq_model, temperature=0)
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(eval_embeddings)
    
    dataset = Dataset.from_dict(evaluation_data)
    
    console.print("\n[bold cyan]▶ Running RAGAS Metrics (Faithfulness, Relevancy, Context Precision)[/bold cyan]")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision
    ]
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    
    # 3. Print Results
    console.print("\n[bold green]✓ Evaluation Complete![/bold green]")
    
    df = result.to_pandas()
    
    table = Table(title="RAGAS Evaluation Scores (Mean)")
    table.add_column("Metric", style="cyan")
    table.add_column("Score (0.0 to 1.0)", style="magenta")
    
    # Calculate means of numeric columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            score = df[col].mean()
            table.add_row(col.replace("_", " ").title(), f"{score:.4f}")
        
    console.print(table)
    
    # 4. Save detailed CSV
    df = result.to_pandas()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    console.print(f"Detailed results saved to [bold]{output_csv}[/bold]")

if __name__ == "__main__":
    run_evaluation()
