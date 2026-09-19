import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load configuration from the workspace root, regardless of the current shell path.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_MULTIMEDIA_DIRECTORY_NAME = "multimedia"
_MEDIA_SUBDIRECTORIES = ("documents", "images", "videos")


def ensure_multimedia_directories() -> Path:
    """Create and return the fixed local multimedia directory hierarchy."""
    project_root = Path(__file__).resolve().parent.parent
    multimedia_dir = project_root / _MULTIMEDIA_DIRECTORY_NAME

    for subdirectory in _MEDIA_SUBDIRECTORIES:
        (multimedia_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    return multimedia_dir


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


def load_documents():
    # Create the local hierarchy on a fresh clone before looking for PDFs.
    multimedia_dir = ensure_multimedia_directories()

    # This loader currently handles PDFs; videos and images remain available for
    # future multimodal processing without being sent through PyPDFLoader.
    loader = DirectoryLoader(
        str(multimedia_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    # Clean text before chunking so embeddings and returned answers use the same text.
    documents = loader.load()

    for document in documents:
        document.page_content = normalize_pdf_text(document.page_content)

    # Empty pages should not create empty chunks or vector records.
    return [document for document in documents if document.page_content]
