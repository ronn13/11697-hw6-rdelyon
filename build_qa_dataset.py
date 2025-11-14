"""
Build QA Dataset with Ollama + FAISS/BM25 Retriever - FIXED VERSION
-------------------------------------------------------------------
Key fixes:
  1. Fixed indentation bug in quota enforcement
  2. Improved prompts for retrieval-dependent questions
  3. Better synthetic fallback to ensure 50+ documents
  4. Document diversity tracking
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
from collections import defaultdict

# -------------------------------------------------------------------
#  Load seed URLs from seed.txt
# -------------------------------------------------------------------
seed_file = 'seed.txt'
if not os.path.exists(seed_file):
    print(f"Error: {seed_file} not found. Please create it with one URL per line.")
    exit(1)

seed_urls = []
with open(seed_file, 'r', encoding='utf-8') as f:
    for line in f:
        url = line.strip()
        if url and not url.startswith('#'):
            seed_urls.append(url)

print(f"Loaded {len(seed_urls)} seed URLs from {seed_file}")

# -------------------------------------------------------------------
#  Step 1: Fetch and preprocess documents
# -------------------------------------------------------------------

def fetch_text(url):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        if len(text.strip()) == 0:
            print(f"Warning: {url} returned empty text (no <p> tags found)")
        return text
    except requests.exceptions.Timeout:
        print(f"Timeout fetching {url} (exceeded 15s)")
        return ''
    except requests.exceptions.ConnectionError:
        print(f"Connection error fetching {url} (check internet)")
        return ''
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error {e.response.status_code} fetching {url}")
        return ''
    except Exception as e:
        print(f"Error fetching {url}: {type(e).__name__}: {e}")
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

# IMPROVED: Add more comprehensive synthetic fallback documents
if len(corpus_files) < 50:
    print(f"Warning: Only {len(corpus_files)} documents fetched. Adding synthetic fallback data...")
    
    synthetic_docs = [
        ("Okot p'Bitek Biography", "Okot p'Bitek (1931-1982) was a prominent Ugandan poet, novelist, and anthropologist born in Gulu, northern Uganda. He studied at King's College Budo, Government Training College Mbarara, Bristol University, and Oxford University where he earned a BLitt in social anthropology. His most famous work, 'Song of Lawino' (1966), written originally in Acholi and later translated to English, became one of the most widely read literary works in Africa. The poem presents a cultural clash between traditional African values and Western modernization through the voice of Lawino, a rural Acholi woman whose husband Ocol has adopted Western ways. p'Bitek's other notable works include 'Song of Ocol' (1970), 'Song of a Prisoner' (1971), and 'Song of Malaya' (1971). He also wrote novels such as 'White Teeth' and scholarly works on African religions and oral literature. p'Bitek served as Director of the National Theatre and National Cultural Centre in Uganda and taught at universities in Nairobi and Makerere."),
        
        ("Ngugi wa Thiong'o Major Works", "Ngugi wa Thiong'o, born James Ngugi in 1938 in Limuru, Kenya, is one of Africa's most celebrated authors. His early novels written in English include 'Weep Not, Child' (1964), his first novel exploring the Mau Mau rebellion through a child's eyes; 'The River Between' (1965), set in the Gikuyu communities during early colonial contact; and 'A Grain of Wheat' (1967), examining Kenya's independence. His most politically charged work, 'Petals of Blood' (1977), critiques post-independence Kenya's neocolonialism and led to his imprisonment without trial in 1977. While detained, he wrote 'Devil on the Cross' on toilet paper, his first novel in Gikuyu language. After his release and exile, Ngugi renounced English and committed to writing in African languages. His later works include 'Matigari' (1986), 'Wizard of the Crow' (2006), and the memoir trilogy 'Dreams in a Time of War,' 'In the House of the Interpreter,' and 'Birth of a Dream Weaver.' He has been a perennial Nobel Prize candidate."),
        
        ("Abdulrazak Gurnah Nobel Prize", "Abdulrazak Gurnah was awarded the Nobel Prize in Literature in 2021 'for his uncompromising and compassionate penetration of the effects of colonialism and the fate of the refugee in the gulf between cultures and continents.' Born in Zanzibar in 1948, Gurnah arrived in England as a refugee in the late 1960s following the Zanzibar Revolution. He became a professor at the University of Kent, where he taught postcolonial literature. His novels include 'Memory of Departure' (1987), 'Pilgrim's Way' (1988), 'Dottie' (1990), 'Paradise' (1994, shortlisted for the Booker Prize), 'Admiring Silence' (1996), 'By the Sea' (2001, longlisted for the Booker), 'Desertion' (2005), 'The Last Gift' (2011), 'Gravel Heart' (2017), and 'Afterlives' (2020). His work explores themes of displacement, memory, identity, and the lasting impact of colonialism on individuals and communities. The Swedish Academy noted his 'emotional compass' and praised his ability to avoid cultural stereotypes while illuminating the devastating effects of colonialism on East African societies."),
        
        ("Barbara Kimenye Children's Literature", "Barbara Kimenye (1929-2012) was a pioneering Ugandan children's author and journalist, best known for her 'Moses' series of children's books. Born in England to a Ugandan Baganda father and English mother, she grew up in Uganda and became one of the first African writers to create widely popular children's literature. Her Moses books, starting with 'Moses' (1968), feature a mischievous schoolboy and were among the first children's books by an African author to gain international recognition. The series includes over a dozen titles such as 'Moses and Mildred,' 'Moses in Trouble,' and 'Moses and the School Farm.' Kimenye's straightforward prose and relatable characters made her books accessible to young African readers while avoiding the condescension often found in colonial-era children's literature. She also wrote for adults, including the collection 'Kalasanda' (1965) and 'Kalasanda Revisited' (1966), humorous stories about village life. She worked as a journalist for various publications and contributed to establishing a distinctly African voice in children's literature."),
        
        ("Grace Ogot Literary Contributions", "Grace Ogot (1930-2015) was a pioneering Kenyan author, nurse, and politician, widely regarded as the first major female voice in East African literature. Born Grace Emily Akinyi in Nyanza Province, she trained as a nurse in Uganda and England. Her short story 'The Year of Sacrifice' appeared in Black Orpheus in 1963, and her first novel 'The Promised Land' (1966) explored migration and Luo cultural traditions. Her best-known work, 'The Strange Bride' (1989), examines traditional beliefs and practices. Ogot's writing often incorporated Luo oral traditions, folklore, and mythology, blending realism with supernatural elements. Her short story collection 'Land Without Thunder' (1968) established her reputation for powerful narratives rooted in African cosmology. She wrote in both English and Luo, and her work addressed women's experiences, cultural change, and the tension between tradition and modernity. Beyond literature, Ogot served as a member of Kenya's parliament, worked for the WHO and UNESCO, and was a founding member of the Writers Association of Kenya. Her pioneering role opened doors for subsequent generations of African women writers."),
        
        ("Meja Mwangi Urban Fiction", "Meja Mwangi (born 1948) is a Kenyan novelist known for his gritty portrayals of urban life and social realism. His breakthrough novel 'Kill Me Quick' (1973) depicts Nairobi's street children and urban poverty with unflinching honesty. 'Going Down River Road' (1976), considered his masterpiece, follows a day in the life of an alcoholic construction worker, exploring themes of economic exploitation and urban alienation in post-independence Kenya. 'The Cockroach Dance' (1979) continues his examination of marginalized urban dwellers. Mwangi also wrote about Kenya's independence struggle in 'Carcase for Hounds' (1974) and 'Taste of Death' (1975). His later works include 'The Last Plague' (2000), a thriller about bioterrorism, and 'The Big Chiefs' (2008). Mwangi's writing style is characterized by vivid descriptions, dark humor, and social critique. Unlike writers who focused on rural settings or political allegory, Mwangi brought attention to the harsh realities of urban poverty, unemployment, and the disillusionment of Kenya's working class. His novels won multiple awards including the Jomo Kenyatta Prize for Literature."),
        
        ("Binyavanga Wainaina Essay Writing", "Binyavanga Wainaina (1971-2019) was a Kenyan author, journalist, and activist whose satirical essay 'How to Write About Africa' (2005) became one of the most influential pieces of African literary criticism. Published in Granta, the essay mockingly outlined stereotypes Western writers use when depicting Africa: starving children, wise elders, exotic wildlife, corruption, and violence. His sharp wit exposed how Africa is often portrayed as a monolithic entity rather than a diverse continent with complex societies. Wainaina's memoir 'One Day I Will Write About This Place' (2011) won the Windham-Campbell Prize and chronicled his childhood in Kenya and journey to becoming a writer. He won the Caine Prize for African Writing in 2002 for his story 'Discovering Home.' As founding editor of the literary magazine Kwani?, Wainaina provided a platform for emerging African writers and promoted new African voices. In 2014, he publicly came out as gay, becoming one of the most prominent LGBTQ+ activists in Africa. His work challenged Western narratives about Africa while also critiquing African societies, always with incisive humor and profound insight."),
        
        ("Yolande Mukagasana Rwanda Genocide", "Yolande Mukagasana (born 1954) is a Rwandan author and humanitarian known for her powerful testimonials about the 1994 Rwandan genocide. Her autobiographical work 'La Mort ne veut pas de moi' (Death Does Not Want Me, 1997), written with Patrick May, describes her harrowing survival during the genocide in which she lost her husband and two children. The book provides a personal account of the hundred days of horror and her subsequent life as a survivor. Her second book 'N'aie pas peur de savoir' (Don't Be Afraid to Know, 1999) continues her testimony and advocacy for genocide awareness. Mukagasana has dedicated her life to bearing witness, traveling internationally to speak about the genocide and working to prevent future atrocities. She founded the Association Muyira to support genocide survivors and orphans. Her writing is characterized by its emotional honesty, refusing to sanitize the brutal reality of genocide while also emphasizing the resilience of survivors. Though her work is painful to read, it serves as crucial historical documentation and a moral imperative to remember and learn from Rwanda's tragedy."),
        
        ("Penina Muhando Swahili Theater", "Penina Mlama (formerly Penina Muhando, born 1948) is a Tanzanian playwright, academic, and cultural activist, widely recognized as Tanzania's leading playwright writing in Swahili. Her plays address social issues including gender inequality, corruption, and cultural change. 'Hatia' (Guilt, 1972), her first play, explored justice and morality. 'Heshima Yangu' (My Respect, 1974) examined women's rights and dignity. 'Pambo' (1975) critiqued superficial modernization and cultural alienation. 'Lina Ubani' (There is Medicine, 1984) addressed women's oppression and the need for social transformation. Muhando's work is notable for incorporating traditional African performance elements, including song, dance, and audience participation, challenging Western theatrical conventions. She has been instrumental in developing Swahili-language theater as a vehicle for social commentary and cultural preservation. As a professor at the University of Dar es Salaam, she trained generations of playwrights and theater practitioners. Her scholarly work on African theater, oral traditions, and performance has been influential internationally. Muhando received numerous awards including the Noma Award for Publishing in Africa and has been recognized as a pioneer in African women's theater."),
        
        ("East African Literature Themes", "East African literature encompasses diverse voices from Kenya, Uganda, Tanzania, Rwanda, and neighboring countries, united by common historical experiences and thematic concerns. Major themes include the impact of colonialism and the struggle for independence, prominently featured in works by Ngugi wa Thiong'o and Okot p'Bitek. The disillusionment of post-independence societies appears in novels by Meja Mwangi and Abdulrazak Gurnah, critiquing neocolonialism and corruption. Cultural identity and the tension between tradition and modernity run through works from Barbara Kimenye to Grace Ogot, often exploring how Western education and values clash with indigenous cultures. Language politics is crucial, with debates over writing in English versus indigenous languages, championed by Ngugi's shift to Gikuyu. Women's voices have grown increasingly prominent, with writers like Grace Ogot, Penina Muhando, and Margaret Ogola addressing gender inequality and women's experiences. The 1994 Rwandan genocide produced a body of testimonial literature, including works by Yolande Mukagasana and Boubacar Boris Diop. Contemporary themes include urbanization, migration, diaspora experiences, and globalization, explored by writers like Yvonne Adhiambo Owuor and Abdulrazak Gurnah. East African literature often blends oral traditions with written forms, incorporating folklore, proverbs, and storytelling techniques."),
        
        ("Ugandan Literature Post-Independence", "Ugandan literature flourished following independence in 1962, though political turmoil under Idi Amin (1971-1979) and subsequent conflicts significantly impacted literary production. Okot p'Bitek's 'Song of Lawino' (1966) set the tone for post-independence critique, challenging both colonialism and uncritical adoption of Western values. During Amin's brutal regime, many writers fled into exile, including writers like Okello Oculi and Taban lo Liyong. The return to relative stability in the 1980s saw renewed literary activity. Moses Isegawa's 'Abyssinian Chronicles' (1998), written in Dutch exile, provided a panoramic view of Uganda's turbulent post-independence history. Jennifer Makumbi's 'Kintu' (2014) wove together centuries of Ugandan history through multiple generations of one family. Contemporary Ugandan literature addresses dictatorship, conflict, and recovery, while also exploring contemporary urban life, sexuality, and identity. Kakwenza Rukirabasaija's satirical novels have challenged current government policies, resulting in persecution. Women writers like Doreen Baingana and Beatrice Lamwaka have brought new perspectives to Ugandan literature. The literary scene includes poetry, with writers like Susan Kiguli and Beverley Nambozo Nsengiyunva gaining recognition, and a growing number of young writers publishing through digital platforms and small presses."),
    ]
    
    for title, text in synthetic_docs:
        idx = len(corpus_files)
        name = f"doc_{idx}_synthetic_{_sanitize_filename(title)}.txt"
        path = os.path.join('data', 'corpus', name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        corpus_files.append({'url': f"synthetic:///{title}", 'filename': name, 'text': text})
    
    # Continue adding more if still needed
    while len(corpus_files) < 50:
        # Split existing synthetic docs
        for doc in list(corpus_files):
            if len(corpus_files) >= 50:
                break
            if len(doc['text']) > 2000:
                mid = len(doc['text']) // 2
                chunk1 = doc['text'][:mid]
                chunk2 = doc['text'][mid:]
                for chunk_text in [chunk1, chunk2]:
                    idx = len(corpus_files)
                    name = f"doc_{idx}_synth_split.txt"
                    path = os.path.join('data', 'corpus', name)
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(chunk_text)
                    corpus_files.append({'url': doc['url'], 'filename': name, 'text': chunk_text})
                    if len(corpus_files) >= 50:
                        break
        break
    
    print(f"After synthetic additions: {len(corpus_files)} corpus files.")

if len(corpus_files) == 0:
    print("ERROR: No documents in corpus. Cannot proceed.")
    exit(1)

print(f"Total corpus files: {len(corpus_files)}")

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

# Track which documents have been used for diversity
doc_usage_count = defaultdict(int)

# IMPROVED: Better prompts for retrieval-dependent questions
def get_improved_prompt(q_type, context):
    """Generate prompts that encourage specific, retrieval-dependent questions"""
    base_instructions = {
        'multiple_choice': """Generate ONE multiple choice question with 4 options about SPECIFIC FACTS from this text about East African literature (Uganda, Kenya, Tanzania, Rwanda). The question should require knowledge of exact details like dates, book titles, awards, or specific events that are NOT common knowledge. Include the correct answer.
Format: {"question": "...", "answers": ["correct answer", "option 2", "option 3", "option 4"]}""",
        
        'factoid': """Generate ONE factoid question requiring a SHORT SPECIFIC ANSWER (name, date, title, place, or brief phrase) about East African literature from Uganda, Kenya, Tanzania, or Rwanda. Ask about details like specific book titles, publication years, awards, places, or exact names that require retrieval to answer.
Format: {"question": "...", "answers": ["answer1", "answer2 (if applicable)"]}""",
        
        'list': """Generate ONE question asking for a LIST of specific items from this text about East African literature (Uganda, Kenya, Tanzania, Rwanda). Ask for things like: multiple book titles by an author, list of awards, list of themes, or list of works. The answer should contain 3-5 specific items.
Format: {"question": "...", "answers": ["item1", "item2", "item3", ...]}"""
    }
    
    return f"""{base_instructions[q_type]}

Context from East African literature:
{context}

Return ONLY valid JSON, no markdown or extra text."""

for doc in tqdm(corpus_files, desc="Generating QA"):
    context = doc['text'][:2000]
    q_type = random.choice(question_types)
    prompt = get_improved_prompt(q_type, context)

    response = client.generate(model="llama3.1:latest", prompt=prompt)
    
    def extract_json_text(text: str):
        if not text:
            return None
        s = text.strip()
        if s.startswith('```'):
            parts = s.split('```')
            for part in parts:
                part = part.strip()
                if part and part[0] in '[{':
                    s = part
                    break
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r'(\[.*\])', s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r'(\{.*\})', s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
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
        out = []
        for a in ans_list or []:
            if isinstance(a, str):
                out.append(a)
            elif isinstance(a, dict):
                for k in ('text', 'answer', 'value'):
                    if k in a and isinstance(a[k], str):
                        out.append(a[k])
                        break
                else:
                    out.append(json.dumps(a, ensure_ascii=False))
            else:
                out.append(str(a))
        return out

    resp_text = getattr(response, 'response', None)
    data = extract_json_text(resp_text)
    if not data:
        preview = (resp_text or '')[:400]
        print(f"Error parsing response: no JSON found. Preview: {preview!r}")
        continue
    
    if isinstance(data, dict):
        data = [data]
    
    for item in data:
        try:
            q_text = item.get('question', '').strip()
            ans_list = normalize_answers(item.get('answers', []))
            if not q_text or not ans_list:
                continue
            questions.append(f"{q_text}\t{q_type}")
            answers.append('\t'.join(ans_list))
            evidence.append(f"{doc['url']}\t{doc['filename']}")
            doc_usage_count[doc['filename']] += 1
        except Exception as e:
            print(f"Skipping malformed item: {e}")

# -------------------------------------------------------------------
#  Step 4: Enforce quotas (20 per question type) - FIXED INDENTATION
# -------------------------------------------------------------------

categorized = defaultdict(list)
for q, a, e in zip(questions, answers, evidence):
    _, q_type = q.split('\t')
    categorized[q_type].append((q, a, e))

final_qs, final_as, final_es = [], [], []

# Process each question type to get at least 20
for q_type in question_types:
    subset = categorized[q_type][:20]
    if len(subset) < 20:
        print(f"Only {len(subset)} {q_type} found; regenerating more...")
        for _ in range(20 - len(subset)):
            # Prefer less-used documents for diversity
            sorted_docs = sorted(corpus_files, key=lambda d: doc_usage_count[d['filename']])
            cf = sorted_docs[random.randint(0, min(10, len(sorted_docs)-1))]
            
            context = cf['text'][:2000]
            prompt = get_improved_prompt(q_type, context)
            response = client.generate(model="llama3.1:latest", prompt=prompt)
            resp_text = getattr(response, 'response', None)
            item = None
            try:
                item = extract_json_text(resp_text)
            except Exception:
                item = None
            if not item:
                continue
            if isinstance(item, list) and len(item) > 0:
                item = item[0]
            if not isinstance(item, dict):
                continue
            q_text = item.get('question', '').strip()
            ans_list = normalize_answers(item.get('answers', []))
            if not q_text or not ans_list:
                continue
            subset.append((
                f"{q_text}\t{q_type}",
                '\t'.join(ans_list),
                f"{cf['url']}\t{cf['filename']}"
            ))
            doc_usage_count[cf['filename']] += 1
    
    for q, a, e in subset:
        final_qs.append(q)
        final_as.append(a)
        final_es.append(e)

# FIXED: This section is now OUTSIDE the question type loop
# -------------------------------------------------------------------
# Ensure we have at least 100 QA pairs in total
# -------------------------------------------------------------------
total_target = 100
attempts = 0
while len(final_qs) < total_target and attempts < 500:
    attempts += 1
    q_type = random.choice(question_types)
    
    # Prefer less-used documents for diversity
    sorted_docs = sorted(corpus_files, key=lambda d: doc_usage_count[d['filename']])
    cf = sorted_docs[random.randint(0, min(10, len(sorted_docs)-1))]
    
    context = cf['text'][:2000]
    prompt = get_improved_prompt(q_type, context)
    
    got = None
    for _ in range(2):  # Try twice
        resp = client.generate(model="llama3.1:latest", prompt=prompt)
        txt = getattr(resp, 'response', None)
        try:
            got = extract_json_text(txt)
        except Exception:
            got = None
        if got:
            break
    
    if not got:
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
    doc_usage_count[cf['filename']] += 1

print(f"\nFinal QA pairs: {len(final_qs)} (target {total_target})")
print(f"Documents used: {len([c for c in doc_usage_count.values() if c > 0])} out of {len(corpus_files)}")
print(f"Question type distribution:")
for qt in question_types:
    count = sum(1 for q in final_qs if q.endswith(f'\t{qt}'))
    print(f"  {qt}: {count}")

# -------------------------------------------------------------------
#  Step 5: Write TSV files in exact required format
# -------------------------------------------------------------------
# Format verification:
# - question.tsv: <question>\t<question_type> (one per line)
# - answer.tsv: <answer1>\t<answer2>\t... (tab-separated, one line per question)
# - evidence.tsv: <url>\t<filename> (one per line)

os.makedirs('data', exist_ok=True)

# Write question.tsv: each line = "question\ttype"
with open('data/question.tsv', 'w', encoding='utf-8') as fq:
    for q in final_qs:
        # q is already formatted as "question\ttype"
        fq.write(q + '\n')

# Write answer.tsv: each line = "answer1\tanswer2\t..." (tab-separated answers)
with open('data/answer.tsv', 'w', encoding='utf-8') as fa:
    for a in final_as:
        # a is already formatted as "ans1\tans2\t..."
        fa.write(a + '\n')

# Write evidence.tsv: each line = "url\tfilename"
with open('data/evidence.tsv', 'w', encoding='utf-8') as fe:
    for e in final_es:
        # e is already formatted as "url\tfilename"
        fe.write(e + '\n')

print("\n✓ Dataset creation complete!")
print(f"✓ Files saved in ./data/")
print(f"✓ Question types: {', '.join(question_types)}")
print(f"✓ Total QA pairs: {len(final_qs)}")
print(f"✓ Total corpus documents: {len(corpus_files)}")

# Format verification examples
print("\n--- Format Examples ---")
if final_qs:
    print(f"Question format: {final_qs[0][:80]}...")
if final_as:
    print(f"Answer format: {final_as[0][:80]}...")
if final_es:
    print(f"Evidence format: {final_es[0][:80]}...")