# This file was moved from the repo root to keep the product surface clean.
# Original content preserved.

# This is a Gradio based Web UI code to create Vector DB from PDF files.
# Upload and index PDF documents via browser
# Create or add to existing collections
# Display existing collections and their associated topics from the persist_dir
# Populate a dropdown dynamically with those collection names

import os
from pathlib import Path
from re import sub
from typing import List

import gradio as gr
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings as LlamaSettings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Chunking settings
EMBED_CHUNK_SIZE = 512
EMBED_CHUNK_OVERLAP = 50


def sanitize_metadata(metadata: dict) -> dict:
    return {k: str(v) if v is not None else "" for k, v in metadata.items()}


def sanitize_name(value: str) -> str:
    return sub(r"[^\w]+", "_", value).strip("_").lower()


def load_documents(pdf_path: str, topic: str) -> list:
    pdf_file = Path(pdf_path)
    raw_docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    documents = []

    for i, doc in enumerate(raw_docs):
        if not doc.text:
            print(f"⚠️ Skipping empty doc {i}")
            continue

        meta = sanitize_metadata(doc.metadata or {})
        meta["topic"] = topic
        meta["source"] = str(pdf_file.name)
        if hasattr(doc, "page_label"):
            meta["page"] = str(doc.page_label)

        documents.append(Document(text=doc.text, metadata=meta))

    return documents


def initialize_embedding() -> HuggingFaceEmbedding:
    print("🔧 Initializing embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    LlamaSettings.chunk_size = EMBED_CHUNK_SIZE
    LlamaSettings.chunk_overlap = EMBED_CHUNK_OVERLAP
    return embed_model


def create_vector_index(pdf_path: str, persist_dir: str, topic: str, collection_name: str):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError("Provided file is not a PDF")

    persist_path = Path(persist_dir)
    if persist_path.exists():
        raise FileExistsError(f"Persist directory already exists: {persist_path}")

    persist_path.mkdir(parents=True, exist_ok=True)

    if not collection_name:
        topic_safe = sanitize_name(topic)
        pdf_name = sanitize_name(pdf_file.stem)
        collection_name = f"{pdf_name}_{topic_safe}"

    documents = load_documents(pdf_path, topic)
    if not documents:
        raise ValueError("No valid documents found in PDF")

    embed_model = initialize_embedding()
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    print(f"✅ Created collection: {collection_name}")
    return collection_name


def add_files_to_existing_collection(pdf_path: str, persist_dir: str, topic: str, collection_name: str):
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError("Provided file is not a PDF")

    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(f"Persist directory not found: {persist_path}")

    documents = load_documents(pdf_path, topic)
    if not documents:
        raise ValueError("No valid documents found in PDF")

    embed_model = initialize_embedding()
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    print(f"📦 Added to collection: {collection_name}")
    return collection_name


def list_collections_and_topics(persist_dir: str) -> List[str]:
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        print(f"⚠️ Persist directory does not exist: {persist_dir}")
        return []

    try:
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collections = chroma_client.list_collections()
        items = []

        for col in collections:
            name = col.name
            topic = "Unknown"
            try:
                docs = col.get(limit=1)
                if docs and docs['metadatas']:
                    metadata = docs['metadatas'][0]
                    topic = metadata.get("topic", "Unknown")
            except Exception:
                pass
            items.append(f"{name} ({topic})")
        return items
    except Exception as e:
        print(f"Error fetching collections: {e}")
        return []


def run_indexing(pdf_file, topic, mode, collection_name, persist_dir):
    try:
        file_path = str(pdf_file)  # pdf_file is already a path-like object

        if mode == "create":
            collection_used = create_vector_index(file_path, persist_dir, topic, collection_name)
        else:
            collection_used = add_files_to_existing_collection(file_path, persist_dir, topic, collection_name)

        return f"✅ Indexed successfully into collection '{collection_used}'"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def launch_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 PDF Vector Indexer (ChromaDB)")
        gr.Markdown("Upload a PDF, specify a topic, and create or update a vector index with citation-ready metadata.")

        with gr.Row():
            pdf_input = gr.File(label="Upload PDF")
            topic_input = gr.Textbox(label="Topic")
            mode_input = gr.Radio(choices=["create", "add"], label="Mode", value="create")

        with gr.Row():
            persist_dir_input = gr.Textbox(
                    label="Persist Directory",
                    value="",
                    info="Directory where ChromaDB should store vector embeddings. Must exist or be created during indexing."
            )
            collection_name_input = gr.Textbox(
                label="Collection Name",
                info="Name of the collection to create or add to. Must be specified, e.g. concatenated pdf file name to topic."
            )

        collection_dropdown = gr.Dropdown(label="📖 Existing Collections", choices=[], interactive=True)
        refresh_button = gr.Button("🔄 Refresh Collections")
        result_output = gr.Textbox(label="Status", lines=2)
        debug_output = gr.Textbox(label="Debug Log", lines=2, interactive=False)

        def handle_indexing(pdf_file, topic, mode, name, persist):
            result = run_indexing(pdf_file, topic, mode, name, persist)
            updated = list_collections_and_topics(persist)
            print("🔍 Collections returned:", updated)
            debug_msg = f"Collections returned: {updated}"
            return result, gr.update(choices=updated, value=None), debug_msg

        index_btn = gr.Button("🚀 Run Indexing")
        index_btn.click(
            fn=handle_indexing,
            inputs=[pdf_input, topic_input, mode_input, collection_name_input, persist_dir_input],
            outputs=[result_output, collection_dropdown, debug_output]
        )

        def refresh_dropdown_handler(persist_path):
            choices = list_collections_and_topics(persist_path)
            print("🔄 Refreshed collections:", choices)
            return gr.update(choices=choices, value=None)

        refresh_button.click(
            fn=refresh_dropdown_handler,
            inputs=[persist_dir_input],
            outputs=[collection_dropdown]
        )

        def handle_collection_selection(selection):
            if not selection:
                return gr.update(value=""), gr.update(value="")
            try:
                name, topic = selection.strip().rsplit(" (", 1)
                topic = topic.rstrip(")")
                return gr.update(value=name), gr.update(value=topic)
            except Exception:
                return gr.update(value=""), gr.update(value="")

        collection_dropdown.change(
            fn=handle_collection_selection,
            inputs=[collection_dropdown],
            outputs=[collection_name_input, topic_input]
        )

    demo.launch()


if __name__ == "__main__":
    launch_ui()

