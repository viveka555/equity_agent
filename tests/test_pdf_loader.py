from rag.pdf_loader import load_pdf

docs = load_pdf("data/annual_reports/BEL_2025.pdf")

print(f"Pages: {len(docs)}")
print(docs[0].metadata)
print(docs[0].page_content[:300])