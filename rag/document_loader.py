import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load configuration from the workspace root, regardless of the current shell path.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def normalize_pdf_text(text: str) -> str:
    # PDF extraction often produces layout artifacts that hurt search quality.
    text = text.replace("\u00a0", " ")

    # Remove PDF decoration and known extraction artifacts.
    text = re.sub(r"(?m)^\s*None\s*$", "", text)
    text = re.sub(r"(?m)^\s*_{10,}\s*$", "", text)
    text = re.sub(
        r"(?m)^Table of Contents\s+Alphabet Inc\.\s+\d+\s*$",
        "",
        text,
    )

    # Preserve paragraph breaks while joining line wraps created by PDF extraction.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def load_documents(docs_directory: Optional[str | Path] = None):
    # Resolve the default document directory relative to this workspace.
    project_root = Path(__file__).resolve().parent.parent

    if docs_directory is None:
        docs_dir = project_root / os.getenv("DOCS_DIR", "docs")
    else:
        docs_dir = Path(docs_directory)

    if not docs_dir.is_absolute():
        docs_dir = project_root / docs_dir

    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    # Clean text before chunking so embeddings and returned answers use the same text.
    documents = loader.load()

    for document in documents:
        document.page_content = normalize_pdf_text(document.page_content)

    # Empty pages should not create empty chunks or vector records.
    return [document for document in documents if document.page_content]
