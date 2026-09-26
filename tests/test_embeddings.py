from rag.pdf_loader import load_pdf
from rag.chunker import split_documents
from rag.embeddings import embeddings

docs = load_pdf("data/annual_reports/BEL_2025.pdf")

chunks = split_documents(docs)

vector = embeddings.embed_query(chunks[0].page_content)

print(f"Chunk count : {len(chunks)}")
print(f"Vector size: {len(vector)}")

print(vector[:10])

