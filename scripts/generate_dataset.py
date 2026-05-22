import os
import json
import random
from typing import List, Dict
from pathlib import Path
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.vectorstore.chroma_store import get_vector_store
from config.settings import settings
from langchain_groq import ChatGroq

console = Console()

class QAPair(BaseModel):
    question: str = Field(description="A specific question that can be answered by the text")
    ground_truth: str = Field(description="A concise and accurate answer based ONLY on the text")

def generate_golden_dataset(num_samples: int = 5, output_file: str = "data/golden_dataset.json"):
    console.print(f"[bold cyan]▶ Generating {num_samples} golden Q&A pairs…[/bold cyan]")
    
    store = get_vector_store()
    
    # We can fetch all documents by doing a dummy search with high k
    # Since we can't easily fetch all without knowing ids, we'll search for common words
    results = store.similarity_search("work energy power force", k=100)
    
    if not results:
        console.print("[red]✗ No documents found in ChromaDB![/red]")
        return
        
    # Deduplicate by chunk_id to get unique chunks
    unique_chunks = {doc.metadata.get("chunk_id", str(i)): doc for i, doc in enumerate(results)}.values()
    chunks = list(unique_chunks)
    
    if len(chunks) < num_samples:
        num_samples = len(chunks)
        
    # Randomly select chunks
    selected_chunks = random.sample(chunks, num_samples)
    
    llm = ChatGroq(model=settings.groq_model, temperature=0.1)
    parser = JsonOutputParser(pydantic_object=QAPair)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert dataset curator. Your job is to create a Golden Dataset for testing a RAG system.\n"
                   "Given the following text chunk, generate exactly one Question and one Ground Truth Answer.\n"
                   "The question must be specific. The answer must be factually correct and based strictly on the text.\n"
                   "{format_instructions}"),
        ("user", "Text Chunk:\n\n{text}")
    ])
    
    chain = prompt | llm | parser
    
    dataset = []
    
    for i, doc in enumerate(selected_chunks):
        console.print(f"Generating pair {i+1}/{num_samples}...")
        try:
            res = chain.invoke({
                "text": doc.page_content,
                "format_instructions": parser.get_format_instructions()
            })
            
            dataset.append({
                "question": res["question"],
                "ground_truth": res["ground_truth"],
                "reference_context": [doc.page_content],
                "metadata": doc.metadata
            })
        except Exception as e:
            console.print(f"[yellow]Failed to generate for chunk {i}: {e}[/yellow]")
            
    # Save to file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    console.print(f"[bold green]✓ Successfully generated {len(dataset)} QA pairs and saved to {output_file}[/bold green]")

if __name__ == "__main__":
    generate_golden_dataset(num_samples=10)
