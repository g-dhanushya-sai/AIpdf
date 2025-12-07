import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_PATH = DATA_DIR / "faiss_store"

import PyPDF2

# Use sentence-transformers + faiss for embeddings and vector search
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    SentenceTransformer = None
    faiss = None

# Fallback TF-IDF if sentence-transformers model download is unavailable
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pickle
except Exception:
    TfidfVectorizer = None
    pickle = None

# Vertex AI via langchain may not be available in this environment; keep a placeholder
try:
    from langchain.llms import VertexAI
except Exception:
    VertexAI = None


def extract_text_from_pdf(file_path: str) -> str:
    text = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text.append(txt)
    return "\n".join(text)


def _split_text(text, chunk_size=1000, chunk_overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def build_vectorstore_from_texts(text_items):
    """
    Build a FAISS index using sentence-transformers. Stores index at VECTORSTORE_PATH and metadata in data/metadata.json
    text_items: list of (text, filename)
    """
    if SentenceTransformer is None or faiss is None:
        raise RuntimeError("Required dependencies missing: install sentence-transformers and faiss-cpu")

    metadatas = []
    texts = []
    for t, fname in text_items:
        chunks = _split_text(t)
        for idx, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({'source': fname, 'chunk': idx})

    if not texts:
        raise RuntimeError('No text extracted from PDFs')

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    # Try sentence-transformers (faiss) first
    if SentenceTransformer is not None and faiss is not None:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # Normalize embeddings for cosine similarity
            import numpy as np
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            embeddings = embeddings / norms

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            faiss.write_index(index, str(VECTORSTORE_PATH / 'index.faiss'))
            import json
            (VECTORSTORE_PATH / 'metadata.json').write_text(json.dumps(metadatas))
            (VECTORSTORE_PATH / 'texts.json').write_text(json.dumps(texts))
            return {'mode': 'faiss', 'index': index, 'metadatas': metadatas, 'texts': texts, 'model_name': 'all-MiniLM-L6-v2'}
        except Exception as e:
            # Fall back to TF-IDF below
            pass

    # Fallback: TF-IDF
    if TfidfVectorizer is None or pickle is None:
        raise RuntimeError('Sentence-transformers unavailable and sklearn/pickle not available for TF-IDF fallback')
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(texts)
    # save matrix and vectorizer
    import numpy as np, json
    np.save(str(VECTORSTORE_PATH / 'tfidf_matrix.npy'), matrix.toarray())
    with open(VECTORSTORE_PATH / 'vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    (VECTORSTORE_PATH / 'metadata.json').write_text(json.dumps(metadatas))
    (VECTORSTORE_PATH / 'texts.json').write_text(json.dumps(texts))
    return {'mode': 'tfidf', 'matrix_path': str(VECTORSTORE_PATH / 'tfidf_matrix.npy'), 'metadatas': metadatas, 'texts': texts}


def load_vectorstore_if_exists():
    import json
    # FAISS-backed index
    if (VECTORSTORE_PATH / 'index.faiss').exists():
        if SentenceTransformer is None or faiss is None:
            raise RuntimeError("Required dependencies missing: sentence-transformers and faiss-cpu")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        index = faiss.read_index(str(VECTORSTORE_PATH / 'index.faiss'))
        metadatas = json.loads((VECTORSTORE_PATH / 'metadata.json').read_text())
        texts = json.loads((VECTORSTORE_PATH / 'texts.json').read_text())
        return {'mode': 'faiss', 'index': index, 'metadatas': metadatas, 'texts': texts, 'model': model}

    # TF-IDF fallback
    if (VECTORSTORE_PATH / 'tfidf_matrix.npy').exists() and (VECTORSTORE_PATH / 'vectorizer.pkl').exists():
        if TfidfVectorizer is None or pickle is None:
            raise RuntimeError('TF-IDF artifacts present but sklearn/pickle not available')
        import numpy as np
        matrix = np.load(str(VECTORSTORE_PATH / 'tfidf_matrix.npy'))
        with open(VECTORSTORE_PATH / 'vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        metadatas = json.loads((VECTORSTORE_PATH / 'metadata.json').read_text())
        texts = json.loads((VECTORSTORE_PATH / 'texts.json').read_text())
        return {'mode': 'tfidf', 'matrix': matrix, 'vectorizer': vectorizer, 'metadatas': metadatas, 'texts': texts}

    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'no files uploaded'}), 400
    text_items = []
    manifest = []
    for f in files:
        filename = f.filename or 'uploaded.pdf'
        safe_name = filename.replace('/', '_')
        outpath = DATA_DIR / safe_name
        f.save(outpath)
        t = extract_text_from_pdf(outpath)
        text_items.append((t, safe_name))
        manifest.append({'filename': safe_name, 'path': str(outpath)})

    # Save manifest
    try:
        import json
        manifest_path = DATA_DIR / 'manifest.json'
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
        else:
            existing = []
        existing.extend(manifest)
        manifest_path.write_text(json.dumps(existing))
    except Exception:
        pass

    try:
        vs = build_vectorstore_from_texts(text_items)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'status': 'ok', 'documents': len(text_items)})


@app.route('/status', methods=['GET'])
def status():
    """Return list of uploaded files and whether a vectorstore exists."""
    import json
    manifest_path = DATA_DIR / 'manifest.json'
    files = []
    if manifest_path.exists():
        try:
            files = json.loads(manifest_path.read_text())
        except Exception:
            files = []
    vs_exists = VECTORSTORE_PATH.exists()
    return jsonify({'files': files, 'vectorstore': bool(vs_exists)})


@app.route('/clear', methods=['POST'])
def clear():
    """Clear vectorstore and manifest (uploaded files remain in data/)."""
    import shutil, json
    try:
        if VECTORSTORE_PATH.exists():
            shutil.rmtree(VECTORSTORE_PATH)
        manifest_path = DATA_DIR / 'manifest.json'
        if manifest_path.exists():
            manifest_path.unlink()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return jsonify({'status': 'cleared'})


@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(force=True)
    question = payload.get('question')
    chat_history = payload.get('chat_history', [])
    if not question:
        return jsonify({'error': 'question required'}), 400

    vs = load_vectorstore_if_exists()
    if vs is None:
        return jsonify({'error': 'no vectorstore found; upload PDFs first'}), 400

    # Perform retrieval using FAISS index
    if not vs:
        return jsonify({'error': 'vectorstore not loaded'}), 500

    hits = []
    mode = vs.get('mode', 'faiss')
    if mode == 'faiss':
        # embed question
        model = vs.get('model')
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
        import numpy as np
        q_emb = model.encode([question], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        k = 4
        D, I = vs['index'].search(q_emb, k)
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = vs['metadatas'][idx]
            text = vs['texts'][idx]
            hits.append({'score': float(score), 'metadata': meta, 'text': text})
    elif mode == 'tfidf':
        # compute cosine similarity between tfidf vectors
        import numpy as np
        vectorizer = vs['vectorizer']
        matrix = vs['matrix']
        q_vec = vectorizer.transform([question]).toarray()[0]
        # normalize
        q_norm = np.linalg.norm(q_vec) + 1e-10
        mat_norms = np.linalg.norm(matrix, axis=1) + 1e-10
        sims = (matrix @ q_vec) / (mat_norms * q_norm)
        top_idx = sims.argsort()[::-1][:4]
        for idx in top_idx:
            hits.append({'score': float(sims[idx]), 'metadata': vs['metadatas'][idx], 'text': vs['texts'][idx]})
    else:
        return jsonify({'error': 'unknown vectorstore mode'}), 500

    # If VertexAI is configured, one could call Gemini here with a prompt combining hits.
    # For now, when Gemini isn't configured, return the retrieved top chunks as the 'answer'.
    if VertexAI is None or (not os.getenv('GOOGLE_APPLICATION_CREDENTIALS') and not os.getenv('GOOGLE_API_KEY')):
        # Build a simple synthesized answer from retrieved chunks
        answer = "\n\n--- Retrieved passages ---\n\n" + "\n\n".join([h['text'] for h in hits])
        sources = [h['metadata'] for h in hits]
        return jsonify({'answer': answer, 'sources': sources})

    # If VertexAI is available, attempt to call it (left as future work)
    return jsonify({'error': 'Gemini/Vertex AI integration not implemented in this environment. Set GOOGLE_APPLICATION_CREDENTIALS and configure TextGenerationModel.'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
