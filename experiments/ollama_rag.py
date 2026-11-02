# This file was moved from the repo root to keep the product surface clean.
# Original content preserved.

# THis code includes:
# 1. Uploading and indexing PDFs
# 2. Querying with or without RAG
# 1. Streams responses from local LLaMA 3.1
#    For this uses LlamaIndex instead of LangChain, because:
#     a. LangChainLLM is designed to wrap LangChain-compatible models, but not all of them
#        expose streaming in a way LlamaIndex can detect.
#     b. The native llama_index.llms.ollama.Ollama class is built specifically for this
#        use case and fully supports streaming.
# 2. Uses RAG when collection is selected
# 3. Skips RAG when “🔌 Don’t use RAG” is chosen
# 4. Supports PDF uploads for live indexing
# 5. Displays source citations when available

import os
import sys
import argparse
import gradio as gr
import chromadb
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    SimpleDirectoryReader
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama  # ✅ Native LlamaIndex Ollama integration

NO_RAG_LABEL = "Don't use RAG"  # Match exactly what get_collection_names() returns


def sanitize_metadata(metadata):
    return {k: str(v) if v is not None else "" for k, v in metadata.items()}


def sanitize_name(value):
    import re
    return re.sub(r"[^\w]+", "_", value).strip("_").lower()


def get_collection_names(persist_dir):
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        return [NO_RAG_LABEL] + [col.name for col in client.list_collections()]
    except Exception as e:
        print(f"Failed to list collections: {e}")
        return [NO_RAG_LABEL]


def index_pdf(file_obj, topic, persist_dir):
    try:
        pdf_path = Path(file_obj.name)
        topic_safe = sanitize_name(topic or "untagged")
        pdf_safe = sanitize_name(pdf_path.stem)
        collection_name = f"{pdf_safe}_{topic_safe}"

        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
        documents = []
        for doc in docs:
            meta = sanitize_metadata(doc.metadata or {})
            meta["topic"] = topic
            meta["source"] = pdf_path.name
            # Try to include page label if available
            if hasattr(doc, "page_label"):
                meta["page"] = str(doc.page_label)
            documents.append(Document(text=doc.text, metadata=meta))

        VectorStoreIndex.from_documents(documents, embed_model=embed_model, storage_context=storage_context)
        return f"✅ Indexed: {pdf_path.name} as collection `{collection_name}`"
    except Exception as e:
        return f"❌ Indexing failed: {e}"


def query_index(persist_dir, collection_name, question, verbose=False):
    try:
        if not question.strip():
            return "⚠️ Please enter a valid question."

        llm = Ollama(model="llama3.1", streaming=False)

        if collection_name.strip() == NO_RAG_LABEL:
            if verbose:
                print("⚡ Using LLM only (no retrieval)...")
            return llm.complete(question)

        chroma_client = chromadb.PersistentClient(path=persist_dir)
        if collection_name not in [col.name for col in chroma_client.list_collections()]:
            return f"❌ Collection '{collection_name}' not found."

        # Step 1: Set up vector index
        collection = chroma_client.get_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

        # Step 2: Create query engine with your LLM
        query_engine = index.as_query_engine(llm=llm, streaming=False)

        # Step 3: Query the engine directly
        response = query_engine.query(question)

        # Step 4: Check if any source nodes were returned
        if not response.source_nodes:
            print("⚠️ No relevant embeddings found. Using LLM only.")
            return llm.complete(question)

        # Step 5: Deduplicate citations
        seen_sources = set()
        unique_citations = []
        for node in response.source_nodes:
            source = node.metadata.get("source", "Unknown source")
            if source not in seen_sources:
                seen_sources.add(source)
                unique_citations.append(source)

        citation_text = ""
        if unique_citations:
            citation_text = "\n\n📚 **Sources:**\n" + "\n".join(
                [f"[{i+1}] {src}" for i, src in enumerate(unique_citations)]
            )

        # Step 6: Return final response
        return (response.response or "⚠️ No answer generated.") + citation_text

    except Exception as e:
        return f"Error: {e}"


def build_ui(persist_dir, verbose=False):
    collections = get_collection_names(persist_dir)
    default_collection = collections[0]

    with gr.Blocks(title="RAG Chatbot") as demo:
        gr.Markdown("## 🧠 RAG Chatbot with LLaMA 3.1 (Ollama)")
        gr.Markdown("Ask questions with or without retrieval. Upload PDFs to create new collections.")

        with gr.Row():
            question = gr.Textbox(label="🔍 Ask a question", placeholder="e.g. What does the tablet support?")
            collection_select = gr.Dropdown(label="📁 Collection", choices=collections, value=default_collection)

        answer_output = gr.Textbox(label="💬 Answer", lines=10, interactive=False)
        question_button = gr.Button("Ask")
        question_button.click(
            fn=query_index,
            inputs=[gr.State(persist_dir), collection_select, question, gr.State(verbose)],
            outputs=answer_output
        )

        gr.Markdown("---")
        gr.Markdown("### 📥 Upload PDF for Live Indexing")

        with gr.Row():
            file = gr.File(label="PDF File", file_types=[".pdf"])
            topic = gr.Textbox(label="Topic", placeholder="e.g. HP Tablet User Guide")
            upload_status = gr.Textbox(label="Status", interactive=False)

        upload_button = gr.Button("📄 Index PDF")
        upload_button.click(fn=index_pdf, inputs=[file, topic, gr.State(persist_dir)], outputs=upload_status)

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio RAG chatbot with LLaMA 3.1 via Ollama")
    parser.add_argument("--persist_dir", required=True, help="Path to ChromaDB index directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    try:
        build_ui(args.persist_dir, verbose=args.verbose)
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        sys.exit(1)

