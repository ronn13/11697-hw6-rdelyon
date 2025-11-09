"""
Build QA Dataset with Ollama + FAISS/BM25 Retriever
---------------------------------------------------
This version adds:
  1. Example `seed_urls.txt` of 25 grounded sources.
  2. Integration with the official Ollama Python client instead of direct HTTP.
  3. Post-processing to enforce question-type quotas (20 MC / 20 factoid / 20 list).
"""

import os
import json
import random
import requests
import faiss
import numpy as np
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from ollama import Client

# -------------------------------------------------------------------
#  Example seed URLs (you can export this to seed_urls.txt)
# -------------------------------------------------------------------
seed_urls = [
    # Uganda
    "https://en.wikipedia.org/wiki/Film_in_Uganda",
    "https://en.wikipedia.org/wiki/Music_of_Uganda",
    "https://en.wikipedia.org/wiki/List_of_Ugandan_films",
    "https://ugandanfilmweek.com/",
    "https://www.okayafrica.com/tag/uganda/",

    # Kenya
    "https://en.wikipedia.org/wiki/Cinema_of_Kenya",
    "https://en.wikipedia.org/wiki/Music_of_Kenya",
    "https://en.wikipedia.org/wiki/Kenyan_literature",
    "https://www.okayafrica.com/tag/kenya/",
    "https://nation.africa/kenya/life-and-style/art-culture",
    "https://www.africa.upenn.edu/NEH/kfolklore.htm?ch=1",
    "https://www.johntyman.com/africa/folk/?ch=1",

    # Tanzania
    "https://en.wikipedia.org/wiki/Cinema_of_Tanzania",
    "https://en.wikipedia.org/wiki/Music_of_Tanzania",
    "https://en.wikipedia.org/wiki/Literature_in_Tanzania",
    "https://www.thecitizen.co.tz/tanzania/magazines/the-beat",
    "https://www.okayafrica.com/tag/tanzania/",

    # Rwanda
    "https://en.wikipedia.org/wiki/Cinema_of_Rwanda",
    "https://en.wikipedia.org/wiki/Music_of_Rwanda",
    "https://en.wikipedia.org/wiki/Literature_in_Rwanda",
    "https://www.newtimes.co.rw/section/arts-culture",
    "https://www.okayafrica.com/tag/rwanda/",

    # South Africa focus
    "https://en.wikipedia.org/wiki/Cinema_of_South_Africa",
    "https://en.wikipedia.org/wiki/Music_of_South_Africa",
    "https://en.wikipedia.org/wiki/South_African_literature",
    "https://www.okayafrica.com/tag/south-africa/",
    "https://www.southafricanfilmfestival.com.au/"
]

# -------------------------------------------------------------------
#  Step 1: Fetch and preprocess documents
# -------------------------------------------------------------------

def fetch_text(url):
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ''

documents = []
for url in tqdm(seed_urls, desc="Downloading docs"):
    text = fetch_text(url)
    if len(text) > 300:
        documents.append({"url": url, "text": text[:8000]})

# -------------------------------------------------------------------
# Save documents to data/corpus and ensure at least 50 supporting docs
# -------------------------------------------------------------------
os.makedirs('data/corpus', exist_ok=True)
def _sanitize_filename(name: str) -> str:
    # Keep safe characters for filenames
    return ''.join(c for c in name if c.isalnum() or c in '-_.').strip()[:100]

corpus_files = []
idx = 0
for d in documents:
    base = os.path.basename(d['url'].rstrip('/')) or 'doc'
    name = f"doc_{idx}_{_sanitize_filename(base)}.txt"
    path = os.path.join('data', 'corpus', name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(d['text'])
    corpus_files.append({'url': d['url'], 'filename': name, 'text': d['text']})
    idx += 1

# If we don't have 50 supporting docs, split existing docs into chunks
chunk_i = idx
i = 0
while len(corpus_files) < 50 and i < len(corpus_files):
    # split text into 3000-char chunks
    text = corpus_files[i]['text']
    if len(text) <= 3500:
        i += 1
        continue
    for start in range(0, len(text), 3000):
        if len(corpus_files) >= 50:
            break
        chunk = text[start:start+3000]
        name = f"doc_{chunk_i}_chunk.txt"
        path = os.path.join('data', 'corpus', name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(chunk)
        corpus_files.append({'url': corpus_files[i]['url'], 'filename': name, 'text': chunk})
        chunk_i += 1
    i += 1

print(f"Saved {len(corpus_files)} corpus files under data/corpus/")

# -------------------------------------------------------------------
#  Step 2: Build retriever indices (BM25 + FAISS)
# -------------------------------------------------------------------

doc_texts = [d["text"] for d in corpus_files]
tokenized_docs = [t.split() for t in doc_texts]

bm25 = BM25Okapi(tokenized_docs)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(doc_texts, show_progress_bar=True)
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(np.array(embeddings).astype('float32'))

# -------------------------------------------------------------------
#  Step 3: Generate questions via Ollama Python client
# -------------------------------------------------------------------
client = Client()

question_types = ['multiple_choice', 'factoid', 'list']
questions, answers, evidence = [], [], []

for doc in tqdm(corpus_files, desc="Generating QA"):
    context = doc['text'][:2000]
    q_type = random.choice(question_types)
    prompt = f"Generate 3 {q_type} questions and their correct answers based on this African arts content. Context:\n{context}\nOutput as JSON: [{{'question': ..., 'answers': [...]}}]"

    response = client.generate(model="llama3", prompt=prompt)
    # Robust JSON extraction from model response
    def extract_json_text(text: str):
        if not text:
            return None
        s = text.strip()
        # Remove markdown fences if present
        if s.startswith('```'):
            # take the content inside fences
            parts = s.split('```')
            for part in parts:
                part = part.strip()
                if part and part[0] in '[{':
                    s = part
                    break
        # Try direct load
        try:
            return json.loads(s)
        except Exception:
            pass
        # Try to extract first JSON array
        m = re.search(r'(\[.*\])', s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # Try to extract first JSON object
        m = re.search(r'(\{.*\})', s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # Try line-delimited JSON
        for ln in s.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                return json.loads(ln)
            except Exception:
                continue
        return None

    def normalize_answers(ans_list):
        """Return a list of strings for answers. Handles cases where model returns dicts."""
        out = []
        for a in ans_list or []:
            if isinstance(a, str):
                out.append(a)
            elif isinstance(a, dict):
                # try common keys
                for k in ('text', 'answer', 'value'):
                    if k in a and isinstance(a[k], str):
                        out.append(a[k])
                        break
                else:
                    # fallback: stringify
                    out.append(json.dumps(a, ensure_ascii=False))
            else:
                out.append(str(a))
        return out

    resp_text = getattr(response, 'response', None)
    data = extract_json_text(resp_text)
    if not data:
        # Log a helpful preview and skip
        preview = (resp_text or '')[:400]
        print(f"Error parsing response: no JSON found. Response preview: {preview!r}")
        continue
    # Normalize to list
    if isinstance(data, dict):
        data = [data]
    for item in data:
        try:
            questions.append(f"{item['question']}\t{q_type}")
            answers.append('\t'.join(normalize_answers(item.get('answers', []))))
            evidence.append(f"{doc['url']}\t{doc['filename']}")
        except Exception as e:
            print(f"Skipping malformed item from model: {e}")

# -------------------------------------------------------------------
#  Step 4: Enforce quotas (20 per question type)
# -------------------------------------------------------------------
from collections import defaultdict

categorized = defaultdict(list)
for q, a, e in zip(questions, answers, evidence):
    _, q_type = q.split('\t')
    categorized[q_type].append((q, a, e))

final_qs, final_as, final_es = [], [], []
for q_type in question_types:
    subset = categorized[q_type][:20]
    if len(subset) < 20:
        print(f"Only {len(subset)} {q_type} found; regenerating more...")
        for _ in range(20 - len(subset)):
            cf = random.choice(corpus_files)
            context = cf['text'][:2000]
            prompt = f"Generate one {q_type} question and answer about African film or literature. Context:\n{context}\nOutput JSON {{'question':..., 'answers':[...]}}"
            response = client.generate(model="llama3", prompt=prompt)
            resp_text = getattr(response, 'response', None)
            item = None
            try:
                item = extract_json_text(resp_text)
            except Exception:
                item = None
            if not item:
                preview = (resp_text or '')[:300]
                print(f"Regeneration: no JSON found for {q_type}. Preview: {preview!r}")
                continue
            # if the model returned a list with one element, take first
            if isinstance(item, list) and len(item) > 0:
                item = item[0]
            if not isinstance(item, dict):
                print(f"Regeneration: unexpected JSON type {type(item)}")
                continue
            subset.append((f"{item.get('question','')}\t{q_type}", '\t'.join(item.get('answers',[])), f"{cf['url']}\t{cf['filename']}"))
    for q, a, e in subset:
        final_qs.append(q)
        final_as.append(a)
        final_es.append(e)

    # -------------------------------------------------------------------
    # Ensure we have at least 100 QA pairs in total (and at least 20 per type above)
    # -------------------------------------------------------------------
    total_target = 100
    attempts = 0
    while len(final_qs) < total_target and attempts < 500:
        attempts += 1
        q_type = random.choice(question_types)
        cf = random.choice(corpus_files)
        context = cf['text'][:2000]
        prompt = f"Generate one {q_type} question and answer based on this African arts content. Context:\n{context}\nReturn ONLY a JSON object like {{'question':..., 'answers':[...]}}, no extra text."
        # try a few times for robustness
        got = None
        for _ in range(3):
            resp = client.generate(model="llama3", prompt=prompt)
            txt = getattr(resp, 'response', None)
            try:
                got = extract_json_text(txt)
            except Exception:
                got = None
            if got:
                break
        if not got:
            preview = (txt or '')[:200]
            print(f"Extra generation failed for {q_type}. Preview: {preview!r}")
            continue
        if isinstance(got, list) and len(got) > 0:
            got = got[0]
        if not isinstance(got, dict):
            continue
        q_text = got.get('question', '').strip()
        ans_list = got.get('answers', [])
        norm_answers = normalize_answers(ans_list)
        if not q_text or not norm_answers:
            continue
        final_qs.append(f"{q_text}\t{q_type}")
        final_as.append('\t'.join(norm_answers))
        final_es.append(f"{cf['url']}\t{cf['filename']}")

    print(f"Final QA pairs: {len(final_qs)} (target {total_target})")

# -------------------------------------------------------------------
#  Step 5: Write TSV files
# -------------------------------------------------------------------
os.makedirs('data', exist_ok=True)
with open('data/question.tsv', 'w', encoding='utf-8') as fq, \
     open('data/answer.tsv', 'w', encoding='utf-8') as fa, \
     open('data/evidence.tsv', 'w', encoding='utf-8') as fe:
    fq.write('\n'.join(final_qs))
    fa.write('\n'.join(final_as))
    fe.write('\n'.join(final_es))

print("Dataset creation complete. Files saved in ./data/")
