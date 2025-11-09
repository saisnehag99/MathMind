#!/usr/bin/env python3
# ingest.py
"""
Process LaTeX source files from arXiv papers and build FAISS indices.
Handles both Number Theory and Algebraic Topology papers.
"""

import os, json, re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PAPERS_DIR = Path("papers")
OUT_DIR = Path("index")
OUT_DIR.mkdir(exist_ok=True)

# Categories to process (must match fetch_arxiv.py category names)
CATEGORIES = [
    "Number_Theory",
    "Algebraic_Topology",
    "Algebraic_Geometry",
    "Commutative_Algebra",
    "General_Mathematics",
    "General_Topology",
    "Group_Theory",
    "KTheory_Homology",
    "Rings_Algebras",
    "Representation_Theory",
    "Logic"
]

# Choose embedding model: local SBERT or OpenAI
USE_OPENAI_EMBED = bool(os.getenv("OPENAI_API_KEY", False))

if USE_OPENAI_EMBED:
    from langchain_openai import OpenAIEmbeddings
    embedder = OpenAIEmbeddings()
else:
    # smaller, fast SBERT
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    def embed_texts(texts):
        return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embedder = None


def extract_latex_symbols(text):
    """Extract LaTeX mathematical symbols from text"""
    patterns = [
      r"\\[A-Za-z]+(?:\{[^}]*\})?",  # LaTeX commands
      r"[A-Za-z]\_\{?[0-9nkp]+\}?",  # subscripts like a_n, x_0
      r"\\pmod\{[^}]+\}",
      r"\bmod\b\s*\d+",
      r"\b[A-Za-z]{1,3}\([nxs]\)"
    ]
    syms = set()
    for p in patterns:
        for m in re.findall(p, text):
            syms.add(m)
    return list(syms)


def clean_latex_text(text):
    """Minimal cleaning of LaTeX text - remove only comments and preamble"""
    # Remove comments
    text = re.sub(r'%.*', '', text)

    # Remove document class and package declarations (preamble)
    text = re.sub(r'\\documentclass\{[^}]*\}', '', text)
    text = re.sub(r'\\usepackage(\[[^\]]*\])?\{[^}]*\}', '', text)

    # Remove begin/end document markers
    text = re.sub(r'\\begin\{document\}', '', text)
    text = re.sub(r'\\end\{document\}', '', text)

    # Remove bibliography commands but keep content
    text = re.sub(r'\\bibliographystyle\{[^}]*\}', '', text)
    text = re.sub(r'\\bibliography\{[^}]*\}', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def load_tex_file(tex_path):
    """Load and read a .tex file"""
    try:
        with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return clean_latex_text(content)
    except Exception as e:
        print(f"    Warning: Could not read {tex_path.name}: {e}")
        return None


def find_main_tex_file(paper_dir):
    """Find the main .tex file in a paper directory"""
    # Look for common main file names first
    main_names = ['main.tex', 'paper.tex', 'manuscript.tex', 'article.tex']
    for name in main_names:
        main_path = paper_dir / name
        if main_path.exists():
            return main_path

    # Otherwise, find the largest .tex file (likely the main document)
    tex_files = list(paper_dir.glob('*.tex'))
    if not tex_files:
        # Try subdirectories
        tex_files = list(paper_dir.glob('**/*.tex'))

    if tex_files:
        # Return the largest .tex file
        return max(tex_files, key=lambda p: p.stat().st_size if p.exists() else 0)

    return None


def load_and_chunk_latex(paper_dir, arxiv_id):
    """Load LaTeX source and chunk it"""
    # Find main .tex file
    main_tex = find_main_tex_file(paper_dir)

    if not main_tex:
        print(f"    No .tex file found in {paper_dir.name}")
        return []

    # Load and clean the text
    raw_text = load_tex_file(main_tex)
    if not raw_text or len(raw_text) < 100:  # Skip very short files
        return []

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(raw_text)

    # Attach metadata
    md_chunks = []
    for i, c in enumerate(chunks):
        md = {
            "source": str(main_tex),
            "arxiv_id": arxiv_id,
            "chunk_id": f"{arxiv_id}__{i}"
        }
        md_chunks.append((c, md))

    return md_chunks


def build_indices(papers_dir=PAPERS_DIR, out_dir=OUT_DIR):
    """Build FAISS indices from LaTeX sources"""

    texts, metadatas = [], []
    symbol_index = {}  # symbol -> list of chunk_ids

    total_papers = 0
    processed_papers = 0

    print("="*70)
    print("MathMind - LaTeX Source Indexing")
    print("="*70)

    # Process each category
    for category in CATEGORIES:
        category_dir = papers_dir / category / "latex_sources"

        if not category_dir.exists():
            print(f"\n⚠️  Category directory not found: {category_dir}")
            print(f"   Run 'python fetch_arxiv.py' to download papers first.")
            continue

        print(f"\n📂 Processing {category}...")

        # Find all extracted paper directories
        paper_dirs = [d for d in category_dir.iterdir() if d.is_dir()]

        if not paper_dirs:
            print(f"   No paper directories found in {category_dir}")
            continue

        print(f"   Found {len(paper_dirs)} papers")

        for paper_dir in paper_dirs:
            total_papers += 1
            arxiv_id = paper_dir.name

            print(f"   [{total_papers:3d}] Processing {arxiv_id}...", end='')

            # Load and chunk the LaTeX source
            chunks = load_and_chunk_latex(paper_dir, arxiv_id)

            if not chunks:
                print(" ✗ No content")
                continue

            # Add chunks to our collections
            for chunk, md in chunks:
                texts.append(chunk)
                metadatas.append(md)

                # Extract and index symbols
                syms = extract_latex_symbols(chunk)
                for s in syms:
                    symbol_index.setdefault(s, []).append(md["chunk_id"])

            processed_papers += 1
            print(f" ✓ {len(chunks)} chunks")

    # Check if we have any content
    if not texts:
        print("\n" + "="*70)
        print("⚠️  No papers were processed!")
        print("="*70)
        print("\nMake sure you have:")
        print("1. Run 'python fetch_arxiv.py' to download papers")
        print("2. Papers extracted in papers/Number_Theory/latex_sources/")
        print("3. Papers extracted in papers/Algebraic_Topology/latex_sources/")
        print("="*70)
        return False

    print(f"\n{'='*70}")
    print(f"📊 Processing Summary")
    print(f"{'='*70}")
    print(f"Total papers found: {total_papers}")
    print(f"Successfully processed: {processed_papers}")
    print(f"Total chunks created: {len(texts)}")
    print(f"Unique LaTeX symbols: {len(symbol_index)}")

    # Generate embeddings
    print(f"\n⚙️  Generating embeddings...")
    if USE_OPENAI_EMBED:
        print("   Using OpenAI embeddings")
        embeddings = embedder.embed_documents(texts)
        embeddings = np.array(embeddings)
    else:
        print("   Using sentence-transformers (local)")
        embeddings = embed_texts(texts)

    # Build FAISS index
    print(f"   Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))

    # Save everything
    print(f"   Saving indices to {out_dir}/...")
    faiss.write_index(index, str(out_dir / "faiss.index"))

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)

    with open(out_dir / "symbol_index.json", "w") as f:
        json.dump(symbol_index, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ SUCCESS!")
    print(f"{'='*70}")
    print(f"Indices saved to: {out_dir.absolute()}")
    print(f"  - faiss.index: {len(texts)} vectors, dimension {dim}")
    print(f"  - meta.pkl: metadata for {len(texts)} chunks")
    print(f"  - symbol_index.json: {len(symbol_index)} unique symbols")
    print(f"\nNext step: Run 'streamlit run streamlit_app.py' to start the system")
    print(f"{'='*70}")

    return True


if __name__ == "__main__":
    success = build_indices()
    if not success:
        exit(1)
