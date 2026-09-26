from rag.pdf_loader import load_pdf
from rag.chunker import split_documents

docs = load_pdf("data/annual_reports/BEL_2025.pdf")
chunks = split_documents(docs)

print(f"Pages loaded : {len(docs)}")
print(f"Chunks created: {len(chunks)}")

print("\nMetadata:")
print(chunks[0].metadata)

print("\nFirst chunk:")
print(chunks[0].page_content[:400])