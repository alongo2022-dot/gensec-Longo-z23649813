"""Custom LangChain RAG application for Homework 2 (NotebookLM-style).

Built from the Lab 02.3 RAG examples in ``02_LangChain/07_RAG``, with
additional pre-built LangChain loaders:

* ``WebBaseLoader`` – clean text extraction from web pages
* ``JSONLoader`` – structured JSON glossary / notes files

API keys and cloud settings are read from environment variables only:

* ``GOOGLE_API_KEY`` / ``GOOGLE_MODEL`` – Gemini chat model (AI Studio)
* ``GOOGLE_CLOUD_PROJECT`` – used when ``USE_VERTEX=1`` for embeddings
* ``USER_AGENT`` – polite User-Agent for web / Wikipedia requests
* ``GITHUB_PERSONAL_ACCESS_TOKEN`` – optional, for GitHub file loading
* ``USE_VERTEX`` – set to ``1`` to use Vertex AI embeddings (lab style)

Usage (from the ``hw2`` directory)::

    uv init --bare
    uv add -r requirements.txt
    uv run python app.py load
    uv run python app.py query
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    AsyncHtmlLoader,
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    GithubFileLoader,
    JSONLoader,
    PyPDFDirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

from wikipedia_loader import WikipediaLoader

# Optional Vertex embeddings (same approach as Lab 02.3).
try:
    from langchain_google_vertexai import VertexAIEmbeddings
except ImportError:  # pragma: no cover - package may be absent locally
    VertexAIEmbeddings = None  # type: ignore[misc, assignment]


ROOT = Path(__file__).resolve().parent
PERSIST_DIR = str(ROOT / "rag_data" / ".chromadb")


def build_embeddings():
    """Create the embedding function from environment configuration.

    Returns:
        An embeddings object compatible with Chroma.
    """
    use_vertex = os.getenv("USE_VERTEX", "0") == "1"
    if use_vertex:
        if VertexAIEmbeddings is None:
            raise RuntimeError(
                "langchain-google-vertexai is not installed but USE_VERTEX=1"
            )
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT when USE_VERTEX=1")
        return VertexAIEmbeddings(
            model_name="gemini-embedding-001",
            project=project,
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-west1"),
        )

    # Default: Google AI Studio embeddings (simpler for local Windows use).
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_query",
    )


def build_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma vector database.

    Returns:
        A Chroma vector store rooted at ``rag_data/.chromadb``.
    """
    return Chroma(
        embedding_function=build_embeddings(),
        persist_directory=PERSIST_DIR,
    )


def load_docs(vectorstore: Chroma, docs: list[Document]) -> None:
    """Split documents into chunks and store their embeddings in Chroma.

    Args:
        vectorstore: Target Chroma instance.
        docs: Documents produced by a LangChain loader.
    """
    if not docs:
        print("  (no documents returned by loader)")
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=10)
    splits = splitter.split_documents(docs)
    vectorstore.add_documents(documents=splits)
    print(f"  stored {len(splits)} chunk(s) from {len(docs)} document(s)")


def load_urls(vectorstore: Chroma, urls: list[str]) -> None:
    """Load raw HTML pages with AsyncHtmlLoader (Lab 02.3 style)."""
    print(f"Loading AsyncHtmlLoader URLs: {urls}")
    load_docs(vectorstore, AsyncHtmlLoader(urls).load())


def load_webbase(vectorstore: Chroma, urls: list[str]) -> None:
    """HW2 custom addition: WebBaseLoader for cleaner page text.

    Unlike AsyncHtmlLoader, WebBaseLoader extracts main textual content
    which often yields better RAG chunks for article-style pages.
    """
    print(f"Loading WebBaseLoader URLs (HW2 addition): {urls}")
    header_template = {
        "User-Agent": os.getenv(
            "USER_AGENT",
            "PDXAcademicClient/gensec (Educational; alongo2022@fau.edu)",
        )
    }
    loader = WebBaseLoader(web_paths=urls, header_template=header_template)
    load_docs(vectorstore, loader.load())


def load_json_dir(vectorstore: Chroma, directory: str) -> None:
    """HW2 custom addition: load structured JSON glossary files.

    Uses LangChain's ``JSONLoader`` when ``jq`` is available; otherwise
    parses the homework glossary format directly into Documents.

    Args:
        vectorstore: Target Chroma instance.
        directory: Folder containing ``.json`` files.
    """
    import json

    path = ROOT / directory
    print(f"Loading JSON files from: {path} (HW2 addition)")
    files = sorted(path.glob("**/*.json"))
    if not files:
        print("  (no JSON files found)")
        return
    for json_file in files:
        docs: list[Document] = []
        try:
            # Prefer the official JSONLoader integration when possible.
            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema=".[] | .term + \": \" + .definition",
                text_content=False,
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = f"json:{json_file.name}"
        except Exception:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            docs = [
                Document(
                    page_content=f"{item['term']}: {item['definition']}",
                    metadata={
                        "source": f"json:{json_file.name}",
                        "term": item["term"],
                    },
                )
                for item in raw
            ]
        load_docs(vectorstore, docs)


def load_wikipedia(vectorstore: Chroma, query: str) -> None:
    """Load Wikipedia pages for a query into the vector database."""
    print(f"Loading Wikipedia pages on: {query}")
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    load_docs(vectorstore, docs)


def load_github(vectorstore: Chroma, file_suffix: str) -> None:
    """Load matching files from a public GitHub repository.

    Requires ``GITHUB_PERSONAL_ACCESS_TOKEN`` in the environment.
    """
    if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
        print("Skipping GitHub load (GITHUB_PERSONAL_ACCESS_TOKEN not set)")
        return
    print(f"Loading GitHub files ending with: {file_suffix}")
    loader = GithubFileLoader(
        repo="wu4f/cs475-src",
        branch="main",
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(file_suffix),
    )
    load_docs(vectorstore, loader.load())


def load_youtube(vectorstore: Chroma, video_id: str) -> None:
    """Load a YouTube transcript and add it to the vector database."""
    print(f"Loading YouTube video: {video_id}")
    transcript = YouTubeTranscriptApi().fetch(video_id)
    text = " ".join(entry.text for entry in transcript)
    docs = [Document(page_content=text, metadata={"source": f"youtube:{video_id}"})]
    load_docs(vectorstore, docs)


def load_txt(vectorstore: Chroma, directory: str) -> None:
    """Load ``.txt`` files from a directory."""
    path = str(ROOT / directory)
    print(f"Loading TXT files from: {path}")
    load_docs(
        vectorstore,
        DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader).load(),
    )


def load_pdf(vectorstore: Chroma, directory: str) -> None:
    """Load PDF files from a directory if any exist."""
    path = ROOT / directory
    print(f"Loading PDF files from: {path}")
    if not path.exists() or not any(path.glob("**/*.pdf")):
        print("  (no PDF files found; skipping)")
        return
    load_docs(vectorstore, PyPDFDirectoryLoader(str(path)).load())


def load_docx(vectorstore: Chroma, directory: str) -> None:
    """Load DOCX files from a directory if any exist."""
    path = ROOT / directory
    print(f"Loading DOCX files from: {path}")
    if not path.exists() or not any(path.glob("**/*.docx")):
        print("  (no DOCX files found; skipping)")
        return
    load_docs(
        vectorstore,
        DirectoryLoader(str(path), glob="**/*.docx", loader_cls=Docx2txtLoader).load(),
    )


def load_md(vectorstore: Chroma, directory: str) -> None:
    """Load Markdown files from a directory if any exist."""
    path = ROOT / directory
    print(f"Loading MD files from: {path}")
    if not path.exists() or not any(path.glob("**/*.md")):
        print("  (no Markdown files found; skipping)")
        return
    load_docs(
        vectorstore,
        DirectoryLoader(
            str(path), glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
        ).load(),
    )


def load_csv(vectorstore: Chroma, directory: str) -> None:
    """Load CSV files from a directory if any exist."""
    path = ROOT / directory
    print(f"Loading CSV files from: {path}")
    if not path.exists() or not any(path.glob("**/*.csv")):
        print("  (no CSV files found; skipping)")
        return
    load_docs(
        vectorstore,
        DirectoryLoader(str(path), glob="**/*.csv", loader_cls=CSVLoader).load(),
    )


def ingest_corpus(vectorstore: Chroma) -> None:
    """Load the homework document corpus into Chroma.

    Includes Lab 02.3-style sources plus HW2 WebBaseLoader / JSONLoader.
    """
    # Lab-style HTML loader (raw pages).
    load_urls(
        vectorstore,
        [
            "https://www.pdx.edu/academics/programs/undergraduate/computer-science",
        ],
    )

    # HW2 addition: WebBaseLoader for cleaner extraction.
    load_webbase(
        vectorstore,
        [
            "https://python.langchain.com/docs/concepts/rag/",
            "https://en.wikipedia.org/wiki/Large_language_model",
        ],
    )

    load_wikipedia(vectorstore, "LangChain")
    load_github(vectorstore, "butcher.py")
    load_youtube(vectorstore, "78600iosmis")

    load_txt(vectorstore, "rag_data/txt")
    load_json_dir(vectorstore, "rag_data/json")  # HW2 addition
    load_pdf(vectorstore, "rag_data/pdf")
    load_docx(vectorstore, "rag_data/docx")
    load_md(vectorstore, "rag_data/md")
    load_csv(vectorstore, "rag_data/csv")

    print("\nRAG database sources:")
    list_sources(vectorstore)


def list_sources(vectorstore: Chroma) -> None:
    """Print unique document sources currently stored in Chroma."""
    retriever = vectorstore.as_retriever()
    sources = set()
    for meta in retriever.vectorstore.get()["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])
    for src in sorted(sources):
        print(f"  {src}")


def format_docs(docs: list[Document]) -> str:
    """Join retrieved document contents into one prompt context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore: Chroma):
    """Build the LCEL retrieval-augmented generation chain.

    Args:
        vectorstore: Populated Chroma store.

    Returns:
        A runnable chain that maps a question string to an answer string.
    """
    model_name = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite")
    llm = ChatGoogleGenerativeAI(model=model_name)
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
    )
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def run_query_loop(vectorstore: Chroma) -> None:
    """Interactive CLI for asking questions against the vector database."""
    chain = build_rag_chain(vectorstore)
    print("Welcome to the HW2 GenSec RAG app (NotebookLM-style).")
    print("Ask a question. Blank line exits.\nSources:")
    list_sources(vectorstore)
    while True:
        try:
            line = input("llm>> ")
        except EOFError:
            break
        if not line:
            break
        print(chain.invoke(line))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for loading documents or querying the RAG app."""
    parser = argparse.ArgumentParser(description="HW2 custom LangChain RAG application")
    parser.add_argument(
        "command",
        choices=["load", "query", "sources"],
        help="load=ingest corpus, query=ask questions, sources=list DB sources",
    )
    args = parser.parse_args(argv)

    if not os.getenv("GOOGLE_API_KEY") and os.getenv("USE_VERTEX", "0") != "1":
        print(
            "Warning: GOOGLE_API_KEY is not set. Export it before querying.",
            file=sys.stderr,
        )

    vectorstore = build_vectorstore()

    if args.command == "load":
        ingest_corpus(vectorstore)
    elif args.command == "sources":
        list_sources(vectorstore)
    else:
        run_query_loop(vectorstore)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
