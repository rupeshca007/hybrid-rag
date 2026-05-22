"""
Prompt loader — reads versioned YAML prompts from disk.

Usage:
    system_msg, user_template = load_prompt("rag_qa")
    # user_template is a Python str with {context} and {question} placeholders
"""

from pathlib import Path

import yaml
from rich.console import Console

from config.settings import settings

console = Console()


def load_prompt(prompt_name: str, version: str | None = None) -> tuple[str, str]:
    """
    Load a versioned prompt from the prompts directory.

    Args:
        prompt_name: Name of the YAML file without extension (e.g., "rag_qa").
        version:     Version folder name (e.g., "v1"). Defaults to settings.prompt_version.

    Returns:
        (system_prompt, user_template) as strings.

    Raises:
        FileNotFoundError: if the prompt file does not exist.
    """
    ver = version or settings.prompt_version
    prompt_path = Path(settings.prompts_dir) / ver / f"{prompt_name}.yaml"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt '{prompt_name}' not found at: {prompt_path}\n"
            f"Available versions: {[d.name for d in Path(settings.prompts_dir).iterdir() if d.is_dir()]}"
        )

    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    system_prompt = data.get("system", "").strip()
    user_template = data.get("user", "").strip()

    console.print(
        f"[cyan]Prompt loaded:[/cyan] {prompt_name} "
        f"[dim](version={data.get('version', ver)})[/dim]"
    )
    return system_prompt, user_template


def format_context(docs) -> str:
    """
    Format retrieved LangChain Documents into a readable context string.
    Each chunk includes its source file and page for citation.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        Formatted multi-line string to inject into the prompt's {context}.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page_number", "?")
        content = doc.page_content.strip()
        parts.append(
            f"[Chunk {i} | {filename} | Page {page}]\n{content}"
        )
    return "\n\n".join(parts)
