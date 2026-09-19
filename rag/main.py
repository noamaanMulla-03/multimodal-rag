from document_loader import load_documents
from text_chunking import text_chunker
from embeddings import embedd_chunks


def main():
    # This is an optional command-line entry point for running the pipeline manually.
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    chunks = text_chunker(documents)
    print(f"Generated {len(chunks)} chunks")

    db = embedd_chunks(chunks)
    # Print a small sample to inspect retrieval without using the API.
    results = db.similarity_search("What is this document about?", k=5)

    for index, result in enumerate(results):
        print(f"+++++++++++++ K{index + 1} +++++++++++++")
        print("\n")
        print(result.metadata)
        print("\n")
        print(result.page_content)
        print("\n")

if __name__ == "__main__":
    main()
