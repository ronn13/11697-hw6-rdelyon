# Question Answering with RAG for East African Literature

This project implements and evaluates a specialized Question Answering (QA) system focused on the domain of East African Literature. The core objective is to compare the performance of baseline Large Language Models (LLMs) against Retrieval-Augmented Generation (RAG) systems.

The goal is to demonstrate RAG's effectiveness in mitigating LLM hallucination and improving answer fidelity by grounding responses in a custom, verifiable knowledge corpus.

## Key Features

*   **Custom Domain Corpus**: A curated knowledge base covering authors, works, and themes in East African Literature.
*   **Grounded QA Dataset**: An automatically generated, verifiable evaluation dataset (~100 Q&A pairs) where every answer is guaranteed to be present in the corpus.
*   **Comparative RAG Evaluation**: Rigorous comparison of six system configurations, including two LLM baselines and four RAG variations.
*   **Dual Retrieval Strategy**: Implementation of both BM25 (Sparse) and FAISS (Dense) retrieval.
*   **Metrics**: Evaluation using Exact Match (EM) and Partial Match (PM) rates.

## Corpus Domain

The knowledge corpus is built from a collection of documents focused on prominent figures and concepts in East African literature. Key authors covered include:

*   **Ngũgĩ wa Thiong'o**: Kenyan author known for works like *A Grain of Wheat* and his advocacy for writing in African languages.
*   **Abdulrazak Gurnah**: 2021 Nobel Prize laureate whose novels explore colonialism and displacement.
*   **Grace Ogot**: A pioneering Kenyan female author who incorporated Luo oral traditions into her writing.
*   **Okot p'Bitek**: Ugandan poet famous for *Song of Lawino*, which explores the clash between tradition and modernity.
*   **Binyavanga Wainaina**: Kenyan author and activist, celebrated for his satirical essay *How to Write About Africa*.

The corpus also includes information on themes like post-colonial identity, social justice, and the role of language in culture.

## Configuration and Setup

This project uses Python 3.8+ and relies on several key tools.

### Prerequisites

*   **Python 3.8+**
*   **Ollama**: Required to run the local `llama3-2-11b-instruct` model. Ensure Ollama is installed and the model is pulled:
    ```bash
    ollama pull llama3-2-11b-instruct
    ```
*   **API Keys**:
    *   OpenAI API Key (for `gpt-4o-mini` and embeddings).

### Installation

1.  Clone the repository and navigate into the directory:
    ```bash
    git clone https://github.com/placeholder/qa-rag-project.git
    cd qa-rag-project
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `config.ini` file from the template and add your API key:
    ```ini
    [openai]
    embedding_model = azure/text-embedding-3-small
    embedding_api_key = 
    openai_api_key = 
    base_url = cmu ai gateway
    ```

## Project Pipeline

The project follows a three-stage pipeline:

1.  **Dataset Construction (`build_qa_dataset.py`)**:
    *   Fetches and augments documents to create the corpus.
    *   Chunks documents and indexes them for both BM25 and FAISS.
    *   Generates a grounded QA dataset (`question.tsv`, `answer.tsv`, `evidence.tsv`).
    ```bash
    python build_qa_dataset.py
    ```

2.  **QA System Execution (`qa_system.py`)**:
    *   Runs the evaluation dataset through all six defined configurations.
    *   Saves predictions to the `predictions/` directory.
    ```bash
    python qa_system.py
    ```

3.  **Evaluation (`eval_script.py`)**:
    *   Compares predictions against the gold answers and calculates performance metrics.
    ```bash
    python eval_script.py --predictions-dir predictions --answers-file data/answer.tsv
    ```

## Results and Discussion

The evaluation conclusively demonstrates the superiority of RAG over ungrounded baselines.

### Experimental Results Summary

| Generator | Retriever | Total Qs | EM Rate | PM Rate | Any Match Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OpenAI | Baseline (No Retrieval) | 100 | 28.0% | 18.0% | 46.0% |
| OpenAI | BM25 (Sparse) | 100 | 55.0% | 33.0% | 88.0% |
| OpenAI | FAISS (Dense) | 100 | 54.0% | 34.0% | 88.0% |
| Llama | Baseline (No Retrieval) | 100 | 29.0% | 17.0% | 46.0% |
| Llama | BM25 (Sparse) | 100 | 55.0% | 32.0% | 87.0% |
| Llama | **FAISS (Dense)** | 100 | 54.0% | **36.0%** | **90.0%** |

### Key Findings

*   **RAG Efficacy is Dominant**: All RAG configurations showed a massive performance leap, increasing the **Any Match Rate** from the baseline's 46% to between **87% and 90%**. This confirms RAG is essential for fact-checking and preventing hallucinations.
*   **Optimal Configuration**: The **Llama Dense RAG** configuration achieved the highest performance (90% Any Match Rate), suggesting the local Llama model excels at synthesizing complex answers when provided with semantically relevant context from FAISS.
*   **Retriever Similarity**: The performance difference between BM25 (Sparse) and FAISS (Dense) retrieval was marginal. This highlights that for a well-curated and chunked corpus, even simple keyword matching can be highly effective.
