import os
import argparse
from typing import List, Tuple, Optional

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
        self.documents = []
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
            self.documents = loader.load()
            
            # Add metadata with just filename
            for doc in self.documents:
                filename = os.path.basename(doc.metadata.get('source', 'unknown'))
                doc.metadata['filename'] = filename
            
            print(f"Loaded {len(self.documents)} documents from {self.doc_dir}")
            
        except Exception as e:
            print(f"Error loading documents: {e}")
    
    def get_documents(self) -> List[Document]:
        return self.documents


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
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
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
            self.retriever.k = 3
            print("BM25 retriever created successfully")
    
    def get_retriever(self):
        return self.retriever
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Retrieve documents (BM25 doesn't provide scores in LangChain)"""
        if not self.retriever:
            return []
        
        docs = self.retriever.get_relevant_documents(query)[:top_k]
        
        results = []
        for i, doc in enumerate(docs):
            filename = doc.metadata.get('filename', 'unknown')
            # BM25 doesn't provide scores, use rank-based pseudo-score
            pseudo_score = 1.0 - (i * 0.1)
            results.append((filename, doc.page_content, pseudo_score))
        
        return results


class QAGenerator:
    """Generator using LangChain with different LLMs"""
    
    def __init__(self, llm_type: str = "gemini"):
        self.llm_type = llm_type
        
        # Initialize LLM
        if llm_type == "gemini":
            print("Initializing LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model = "gemini-1.5-flash-002",
                api_key = config.get('openai', 'cmu_gemoni_key'),
                base_url=config.get('openai', 'base_url')    
            )
        else:
            self.llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            api_key=config.get('openai', 'openai_api_key'),
            base_url=config.get('openai', 'base_url')
        )
        
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert in East African literature with deep knowledge of oral traditions, historical and contemporary authors, literary movements, poetry, novels, plays, folktales, post-colonial theory, linguistic context, and regional cultural influences from Kenya, Uganda, Tanzania, Rwanda, Burundi, Ethiopia, Somalia, South Sudan, and related diasporas.

Context Documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the information in the context documents
- Be concise and specific
- For multiple choice questions, provide only the answer (year, name, or title)
- For list questions, provide items separated by tabs (no bullets or numbers)
- For factoid questions, provide a brief direct answer
- If the answer is not in the context, say "Information not found"
- For complex topics, break answers into sections (e.g., Themes, Context, Influence, Examples).
- Define key terms when they appear (e.g., “Ugandan literary renaissance”, “Apole line in Somali poetry”).
- Provide short reading recommendations when helpful.
- Use neutral, academic-but-accessible tone.
- Avoid overly generalized statements; focus on specifics tied to East Africa.

Answer:"""
        )
        
        self.no_context_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an expert in East African literature with deep knowledge of oral traditions, historical and contemporary authors, literary movements, poetry, novels, plays, folktales, post-colonial theory, linguistic context, and regional cultural influences from Kenya, Uganda, Tanzania, Rwanda, Burundi, Ethiopia, Somalia, South Sudan, and related diasporas.

Question: {question}

Instructions:
- Answer based ONLY on the information in the context documents
- Be concise and specific
- For multiple choice questions, provide only the answer (year, name, or title)
- For list questions, provide items separated by tabs (no bullets or numbers)
- For factoid questions, provide a brief direct answer
- If the answer is not in the context, say "Information not found"
- For complex topics, break answers into sections (e.g., Themes, Context, Influence, Examples).
- Define key terms when they appear (e.g., “Ugandan literary renaissance”, “Apole line in Somali poetry”).
- Provide short reading recommendations when helpful.
- Use neutral, academic-but-accessible tone.
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
            result = qa_chain.invoke({"query": question})
            answer = result['result'].strip()
            
            # Extract source information
            source_docs = result.get('source_documents', [])
            if source_docs:
                doc_info = []
                for doc in source_docs:
                    filename = doc.metadata.get('filename', 'unknown')
                    doc_info.append(f"{filename}:1.0000")
                additional_info = "|".join(doc_info)
            else:
                additional_info = None
            
            return answer, additional_info
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Error generating response", None
    
    def generate_without_retrieval(self, question: str) -> str:
        """Generate answer without retrieval"""
        try:
            from langchain_core.output_parsers import StrOutputParser

            chain = self.no_context_prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"question": question})
            return answer.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
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
    generators = ['gemini', 'openai']
    
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
            
            # Create QA system
            qa_system = QASystem(retriever_wrapper, generator, use_retrieval)
            
            # Process questions
            output_file = os.path.join(args.output, f"{retriever_type}_{generator_type}.tsv")
            print(f"Processing {len(questions)} questions and writing to {output_file}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, (question, q_type) in enumerate(questions, 1):
                    print(f"  [{i}/{len(questions)}] Processing: {question[:60]}...")
                    answer, additional_info = qa_system.answer(question)
                    
                    # Clean answer for TSV format
                    answer = answer.replace('\n', ' ').replace('\t', ' ')
                    
                    if additional_info:
                        f.write(f"{answer}\t{additional_info}\n")
                    else:
                        f.write(f"{answer}\n")
            
            print(f"✓ Results saved to {output_file}\n")
    
    print(f"\n{'='*60}")
    print("All combinations completed!")
    print(f"Output files saved in: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
