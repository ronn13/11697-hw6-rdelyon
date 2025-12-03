import os
import argparse
import signal
import threading
from typing import List, Tuple, Optional
from functools import wraps

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

import configparser
config = configparser.ConfigParser()
# Read the configuration file
config.read('config.ini')

class DocumentStore:
    """Manages document loading and preprocessing with LangChain"""
    
    def __init__(self, doc_dir: str = "corpus"):
        self.doc_dir = doc_dir
        self.split_documents = []
        self.load_documents()
    
    def load_documents(self):
        """Load documents using LangChain loaders"""
        if not os.path.exists(self.doc_dir):
            print(f"Warning: Document directory {self.doc_dir} not found")
            return
        
        try:
            # Load all text files from directory
            loader = DirectoryLoader(
                self.doc_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()
            
            # Add metadata and split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            
            for doc in documents:
                filename = os.path.basename(doc.metadata.get('source', 'unknown'))
                doc.metadata['filename'] = filename
                # Split document and add to the list
                chunks = text_splitter.split_documents([doc])
                self.split_documents.extend(chunks)
            
            print(f"Loaded and split {len(documents)} documents into {len(self.split_documents)} chunks from {self.doc_dir}")
            
        except Exception as e:
            print(f"Error loading documents: {e}")
    
    def get_documents(self) -> List[Document]:
        return self.split_documents


class VectorStoreRetriever:
    """Retriever using LangChain's vector store with embeddings"""
    
    def __init__(self, documents: List[Document], embedding_model: str = config.get('openai', 'embedding_model')):
        self.documents = documents
        self.vectorstore = None
        self.retriever = None
        
        if documents:
            print(f"Creating embeddings with {embedding_model}...")
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model= config.get('openai', 'embedding_model'),
                api_key= config.get('openai', 'embedding_api_key'),
                base_url= config.get('openai', 'base_url')
            )
            
            # Create FAISS vector store
            self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            print("Vector store created successfully")
    
    def get_retriever(self):
        return self.retriever
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Retrieve documents with similarity scores"""
        if not self.vectorstore:
            return []
        
        # Using similarity_search_with_score to get scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        results = []
        for doc, score in docs_with_scores:
            filename = doc.metadata.get('filename', 'unknown')
            # Convert distance to similarity (FAISS returns L2 distance)
            similarity = 1 / (1 + score)
            results.append((filename, doc.page_content, similarity))
            
        
        return results


class LangChainBM25Retriever:
    """BM25 Retriever using LangChain"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.retriever = None
        
        if documents:
            print("Creating BM25 retriever...")
            self.retriever = BM25Retriever.from_documents(documents)
            self.retriever.k = 5
            print("BM25 retriever created successfully")
    
    def get_retriever(self):
        return self.retriever
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Retrieve documents (BM25 doesn't provide scores in LangChain)"""
        if not self.retriever:
            return []

        self.retriever.k = top_k
        docs = self.retriever.get_relevant_documents(query)
        
        results = []
        for i, doc in enumerate(docs):
            filename = doc.metadata.get('filename', 'unknown')
            # BM25 in langchain doesn't provide scores, use rank-based pseudo-score
            pseudo_score = 1.0 - (i * 0.1)
            results.append((filename, doc.page_content, pseudo_score))
        
        return results


def timeout_handler(timeout_seconds=120):
    """Decorator to add timeout to function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function call exceeded {timeout_seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        return wrapper
    return decorator


class QAGenerator:
    """Generator using LangChain with different LLMs"""
    
    def __init__(self, llm_type: str = "llama"):
        self.llm_type = llm_type
        
        # Initialize LLM
        if llm_type == "llama":
            print("Initializing LLM...")
            try:
                self.llm = ChatOpenAI(
                    model="llama3-2-11b-instruct",
                    api_key=config.get('openai', 'openai_api_key'),
                    base_url=config.get('openai', 'base_url'),
                    request_timeout=120,  # 120 second timeout
                    max_retries=2
                )
                print(f"  Model: llama3-2-11b-instruct")
                print(f"  Base URL: {config.get('openai', 'base_url')}")
            except TypeError:
                # Fallback if request_timeout is not supported
                self.llm = ChatOpenAI(
                    model="llama3-2-11b-instruct",
                    api_key=config.get('openai', 'openai_api_key'),
                    base_url=config.get('openai', 'base_url')
                )
                print(f"  Model: llama3-2-11b-instruct (no timeout)")
                print(f"  Base URL: {config.get('openai', 'base_url')}")
        else:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini-2024-07-18",
                    api_key=config.get('openai', 'openai_api_key'),
                    base_url=config.get('openai', 'base_url'),
                    request_timeout=120,  # 120 second timeout
                    max_retries=2
                )
                print(f"  Model: gpt-4o-mini-2024-07-18")
                print(f"  Base URL: {config.get('openai', 'base_url')}")
            except TypeError:
                # Fallback if request_timeout is not supported
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini-2024-07-18",
                    api_key=config.get('openai', 'openai_api_key'),
                    base_url=config.get('openai', 'base_url')
                )
                print(f"  Model: gpt-4o-mini-2024-07-18 (no timeout)")
                print(f"  Base URL: {config.get('openai', 'base_url')}")
        
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert in East African literature with deep knowledge of oral traditions, historical and contemporary authors, literary movements, poetry, novels, plays, folktales, post-colonial theory, linguistic context, and regional cultural influences from Kenya, Uganda, Tanzania, Rwanda, Burundi, Ethiopia, Somalia, South Sudan, and related diasporas.

Context Documents:
{context}

Question: {question}

Instructions:
- Answer the question based on your internal knowledge.
- Be concise and specific
- Provide only the answer (e.g., a year, a name, a title, or a tab-separated list).
- If you do not know the answer, say "Information not found".

Answer:"""
        )
        
        self.no_context_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an expert in East African literature with deep knowledge of oral traditions, historical and contemporary authors, literary movements, poetry, novels, plays, folktales, post-colonial theory, linguistic context, and regional cultural influences from Kenya, Uganda, Tanzania, Rwanda, Burundi, Ethiopia, Somalia, South Sudan, and related diasporas.

Question: {question}

Instructions:
- Answer based ONLY on the information in the context documents
- Be concise and specific.
- Provide only the answer (e.g., a year, a name, a title, or a tab-separated list).
- If the answer is not in the context, say "Information not found"
- Avoid overly generalized statements; focus on specifics tied to East Africa.

Answer:"""
        )
    
    def create_qa_chain(self, retriever):
        """Create a RetrievalQA chain"""
        if retriever is None:
            return None
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.rag_prompt}
        )
        
        return qa_chain
    
    def generate_with_retrieval(self, question: str, qa_chain) -> Tuple[str, Optional[str]]:
        """Generate answer using retrieval chain"""
        try:
            print(f"    Calling LLM API...", end='', flush=True)
            
            # Wrap the invoke call with timeout
            @timeout_handler(timeout_seconds=120)
            def call_qa_chain():
                return qa_chain.invoke({"query": question})
            
            try:
                result = call_qa_chain()
                print(" ✓", flush=True)
                answer = result['result'].strip()
                
                # Extract source information
                source_docs = result.get('source_documents', [])
                if source_docs:
                    doc_info = []
                    for doc in source_docs:
                        filename = doc.metadata.get('filename', 'unknown')
                        score = doc.metadata.get('score', 1.0) # Default to 1.0 if score not present
                        # For vector retrievers (like FAISS), score is distance, so convert to similarity
                        if hasattr(qa_chain.retriever, 'vectorstore'):
                            score = 1 / (1 + score)
                        doc_info.append(f"{filename}:{score:.4f}")
                    additional_info = "|".join(doc_info)
                else:
                    additional_info = None
                
                return answer, additional_info
            except TimeoutError as te:
                print(f" ✗ Timeout: {te}", flush=True)
                return "Error: Request timed out", None
        except Exception as e:
            print(f" ✗ Error: {e}", flush=True)
            return "Error generating response", None
    
    def generate_without_retrieval(self, question: str) -> str:
        """Generate answer without retrieval"""
        try:
            from langchain_core.output_parsers import StrOutputParser

            chain = self.no_context_prompt | self.llm | StrOutputParser()
            print(f"    Calling LLM API...", end='', flush=True)
            
            # Wrap the invoke call with timeout
            @timeout_handler(timeout_seconds=120)
            def call_llm():
                return chain.invoke({"question": question})
            
            try:
                answer = call_llm()
                print(" ✓", flush=True)
                return answer.strip()
            except TimeoutError as te:
                print(f" ✗ Timeout: {te}", flush=True)
                return "Error: Request timed out"
        except Exception as e:
            print(f" ✗ Error: {e}", flush=True)
            return "Error generating response"


class QASystem:
    """Main QA System using LangChain"""
    
    def __init__(self, retriever_wrapper, generator: QAGenerator, use_retrieval: bool = True):
        self.retriever_wrapper = retriever_wrapper
        self.generator = generator
        self.use_retrieval = use_retrieval
        self.qa_chain = None
        
        if use_retrieval and retriever_wrapper:
            retriever = retriever_wrapper.get_retriever()
            self.qa_chain = generator.create_qa_chain(retriever)
    
    def answer(self, question: str) -> Tuple[str, Optional[str]]:
        """Answer a question"""
        if self.use_retrieval and self.qa_chain:
            return self.generator.generate_with_retrieval(question, self.qa_chain)
        else:
            answer = self.generator.generate_without_retrieval(question)
            return answer, None


def load_questions(filepath: str) -> List[Tuple[str, str]]:
    """Load questions from TSV file"""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                questions.append((parts[0], parts[1]))
    return questions


def main():
    parser = argparse.ArgumentParser(description='QA System with LangChain RAG')
    parser.add_argument('--questions', type=str, default='data/question.tsv',
                        help='Path to questions TSV file')
    parser.add_argument('--documents', type=str, default='data/corpus',
                        help='Path to corpus directory')
    parser.add_argument('--output', type=str, default='output/prediction',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check if questions file exists
    if not os.path.exists(args.questions):
        print(f"Error: Questions file '{args.questions}' not found!")
        return
    
    # Load documents
    doc_store = DocumentStore(args.documents)
    documents = doc_store.get_documents()
    
    if len(documents) == 0:
        print(f"Warning: No documents found in '{args.documents}' directory!")
    
    # Load questions once
    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions}\n")
    
    # Define combinations to test
    retrievers = ['None', 'vector', 'bm25']
    generators = ['llama', 'openai']
    
    # Loop over all combinations
    for retriever_type in retrievers:
        for generator_type in generators:
            print(f"\n{'='*60}")
            print(f"Processing: Retriever={retriever_type}, Generator={generator_type}")
            print(f"{'='*60}\n")
            
            # Initialize retriever
            retriever_wrapper = None
            use_retrieval = retriever_type != 'None'
            
            if use_retrieval:
                print(f"Initializing {retriever_type} retriever...")
                if retriever_type == 'vector':
                    retriever_wrapper = VectorStoreRetriever(documents)
                elif retriever_type == 'bm25':
                    retriever_wrapper = LangChainBM25Retriever(documents)
            else:
                print("Running without retrieval (no RAG)")
            
            # Initialize generator
            generator = QAGenerator(llm_type=generator_type)
            
            # Test connection with a simple call
            print("Testing LLM connection...", end='', flush=True)
            try:
                test_chain = generator.no_context_prompt | generator.llm
                test_result = test_chain.invoke({"question": "Say 'test'"})
                print(" ✓ Connection successful\n", flush=True)
            except Exception as e:
                print(f" ✗ Connection test failed: {e}", flush=True)
                print("  Continuing anyway...\n", flush=True)
            
            # Create QA system
            qa_system = QASystem(retriever_wrapper, generator, use_retrieval)
            
            # Process questions
            output_file = os.path.join(args.output, f"{retriever_type}_{generator_type}.tsv")
            print(f"Processing {len(questions)} questions and writing to {output_file}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, (question, q_type) in enumerate(questions, 1):
                    print(f"  [{i}/{len(questions)}] Processing: {question[:60]}...", flush=True)
                    try:
                        answer, additional_info = qa_system.answer(question)
                        
                        # Clean answer for TSV format
                        answer = answer.replace('\n', ' ').replace('\t', ' ')
                        
                        if additional_info:
                            f.write(f"{answer}\t{additional_info}\n")
                        else:
                            f.write(f"{answer}\n")
                        f.flush()  # Ensure data is written immediately
                    except KeyboardInterrupt:
                        print("\n\nInterrupted by user. Exiting...")
                        raise
                    except Exception as e:
                        print(f"    ✗ Unexpected error: {e}", flush=True)
                        f.write("Error generating response\n")
                        f.flush()
            
            print(f"✓ Results saved to {output_file}\n")
    
    print(f"\n{'='*60}")
    print("All combinations completed!")
    print(f"Output files saved in: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
