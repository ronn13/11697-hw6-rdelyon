import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import List

from langchain_core.documents import Document

# Import the classes we want to test
from qa_system import (
    DocumentStore,
    VectorStoreRetriever,
    LangChainBM25Retriever,
    QAGenerator,
    QASystem,
    load_questions
)


class TestDocumentStore:
    """Test DocumentStore class"""
    
    def test_init_with_existing_directory(self):
        """Test DocumentStore initialization with existing directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_file1 = os.path.join(tmpdir, "test1.txt")
            test_file2 = os.path.join(tmpdir, "test2.txt")
            
            with open(test_file1, 'w', encoding='utf-8') as f:
                f.write("Test content 1")
            with open(test_file2, 'w', encoding='utf-8') as f:
                f.write("Test content 2")
            
            with patch('qa_system.DirectoryLoader') as mock_loader:
                mock_docs = [
                    Document(page_content="Test content 1", metadata={'source': test_file1}),
                    Document(page_content="Test content 2", metadata={'source': test_file2})
                ]
                mock_loader_instance = Mock()
                mock_loader_instance.load.return_value = mock_docs
                mock_loader.return_value = mock_loader_instance
                
                store = DocumentStore(doc_dir=tmpdir)
                assert len(store.documents) == 2
    
    def test_init_with_nonexistent_directory(self):
        """Test DocumentStore initialization with non-existent directory"""
        with patch('builtins.print'):
            store = DocumentStore(doc_dir="nonexistent_dir_12345")
            assert len(store.documents) == 0
    
    def test_get_documents(self):
        """Test get_documents method"""
        store = DocumentStore.__new__(DocumentStore)
        store.documents = [
            Document(page_content="Test 1", metadata={}),
            Document(page_content="Test 2", metadata={})
        ]
        docs = store.get_documents()
        assert len(docs) == 2
        assert isinstance(docs, list)


class TestVectorStoreRetriever:
    """Test VectorStoreRetriever class"""
    
    @patch('qa_system.OpenAIEmbeddings')
    @patch('qa_system.FAISS')
    def test_init_with_documents(self, mock_faiss, mock_embeddings):
        """Test VectorStoreRetriever initialization"""
        docs = [
            Document(page_content="Test content 1", metadata={'filename': 'test1.txt'}),
            Document(page_content="Test content 2", metadata={'filename': 'test2.txt'})
        ]
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vectorstore
        
        retriever = VectorStoreRetriever(documents=docs)
        
        assert retriever.vectorstore is not None
        assert retriever.retriever is not None
        mock_faiss.from_documents.assert_called_once()
    
    def test_init_without_documents(self):
        """Test VectorStoreRetriever initialization without documents"""
        retriever = VectorStoreRetriever(documents=[])
        assert retriever.vectorstore is None
        assert retriever.retriever is None
    
    @patch('qa_system.OpenAIEmbeddings')
    @patch('qa_system.FAISS')
    def test_retrieve(self, mock_faiss, mock_embeddings):
        """Test retrieve method"""
        docs = [
            Document(page_content="Test content 1", metadata={'filename': 'test1.txt'}),
            Document(page_content="Test content 2", metadata={'filename': 'test2.txt'})
        ]
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        # Mock similarity_search_with_score
        mock_doc1 = Document(page_content="Test content 1", metadata={'filename': 'test1.txt'})
        mock_doc2 = Document(page_content="Test content 2", metadata={'filename': 'test2.txt'})
        mock_vectorstore.similarity_search_with_score.return_value = [
            (mock_doc1, 0.1),
            (mock_doc2, 0.2)
        ]
        
        mock_faiss.from_documents.return_value = mock_vectorstore
        
        retriever = VectorStoreRetriever(documents=docs)
        results = retriever.retrieve("test query", top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == 'test1.txt'
        assert results[0][2] > 0  # similarity score should be positive
    
    def test_retrieve_without_vectorstore(self):
        """Test retrieve method when vectorstore is None"""
        retriever = VectorStoreRetriever(documents=[])
        results = retriever.retrieve("test query")
        assert results == []


class TestLangChainBM25Retriever:
    """Test LangChainBM25Retriever class"""
    
    @patch('qa_system.BM25Retriever')
    def test_init_with_documents(self, mock_bm25):
        """Test LangChainBM25Retriever initialization"""
        docs = [
            Document(page_content="Test content 1", metadata={'filename': 'test1.txt'}),
            Document(page_content="Test content 2", metadata={'filename': 'test2.txt'})
        ]
        
        mock_retriever_instance = Mock()
        mock_retriever_instance.k = 3
        mock_bm25.from_documents.return_value = mock_retriever_instance
        
        retriever = LangChainBM25Retriever(documents=docs)
        
        assert retriever.retriever is not None
        mock_bm25.from_documents.assert_called_once()
    
    def test_init_without_documents(self):
        """Test LangChainBM25Retriever initialization without documents"""
        retriever = LangChainBM25Retriever(documents=[])
        assert retriever.retriever is None
    
    @patch('qa_system.BM25Retriever')
    def test_retrieve(self, mock_bm25):
        """Test retrieve method"""
        docs = [
            Document(page_content="Test content 1", metadata={'filename': 'test1.txt'}),
            Document(page_content="Test content 2", metadata={'filename': 'test2.txt'})
        ]
        
        mock_retriever_instance = Mock()
        mock_retriever_instance.k = 3
        mock_doc1 = Document(page_content="Test content 1", metadata={'filename': 'test1.txt'})
        mock_doc2 = Document(page_content="Test content 2", metadata={'filename': 'test2.txt'})
        mock_retriever_instance.get_relevant_documents.return_value = [mock_doc1, mock_doc2]
        mock_bm25.from_documents.return_value = mock_retriever_instance
        
        retriever = LangChainBM25Retriever(documents=docs)
        results = retriever.retrieve("test query", top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == 'test1.txt'
        assert results[0][2] > 0  # pseudo score should be positive
    
    def test_retrieve_without_retriever(self):
        """Test retrieve method when retriever is None"""
        retriever = LangChainBM25Retriever(documents=[])
        results = retriever.retrieve("test query")
        assert results == []


class TestQAGenerator:
    """Test QAGenerator class"""
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    def test_init_gemini(self, mock_gemini):
        """Test QAGenerator initialization with Gemini"""
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        generator = QAGenerator(llm_type="gemini")
        
        assert generator.llm_type == "gemini"
        assert generator.llm is not None
        mock_gemini.assert_called_once()
    
    @patch('qa_system.ChatOpenAI')
    def test_init_openai(self, mock_openai):
        """Test QAGenerator initialization with OpenAI"""
        mock_llm = Mock()
        mock_openai.return_value = mock_llm
        
        generator = QAGenerator(llm_type="openai")
        
        assert generator.llm_type == "openai"
        assert generator.llm is not None
        mock_openai.assert_called_once()
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    @patch('qa_system.RetrievalQA')
    def test_create_qa_chain(self, mock_qa, mock_gemini):
        """Test create_qa_chain method"""
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_qa_chain = Mock()
        mock_qa.from_chain_type.return_value = mock_qa_chain
        
        generator = QAGenerator(llm_type="gemini")
        chain = generator.create_qa_chain(mock_retriever)
        
        assert chain is not None
        mock_qa.from_chain_type.assert_called_once()
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    def test_create_qa_chain_none_retriever(self, mock_gemini):
        """Test create_qa_chain with None retriever"""
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        generator = QAGenerator(llm_type="gemini")
        chain = generator.create_qa_chain(None)
        
        assert chain is None
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    @patch('qa_system.RetrievalQA')
    def test_generate_with_retrieval(self, mock_qa, mock_gemini):
        """Test generate_with_retrieval method"""
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        mock_qa_chain = Mock()
        mock_qa.from_chain_type.return_value = mock_qa_chain
        
        # Mock the invoke result
        mock_doc = Document(page_content="Test", metadata={'filename': 'test.txt'})
        mock_qa_chain.invoke.return_value = {
            'result': 'Test answer',
            'source_documents': [mock_doc]
        }
        
        generator = QAGenerator(llm_type="gemini")
        chain = generator.create_qa_chain(Mock())
        
        answer, info = generator.generate_with_retrieval("test question", chain)
        
        assert answer == "Test answer"
        assert info is not None
        assert "test.txt" in info
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    def test_generate_without_retrieval(self, mock_gemini):
        """Test generate_without_retrieval method"""
        # Mock the entire chain construction by patching StrOutputParser
        # and the pipe operations
        with patch('langchain_core.output_parsers.StrOutputParser') as mock_str_parser:
            # Create a mock chain that will be returned
            mock_final_chain = MagicMock()
            mock_final_chain.invoke.return_value = "Test answer"
            
            # Mock StrOutputParser instance
            mock_str_parser_instance = MagicMock()
            mock_str_parser.return_value = mock_str_parser_instance
            
            # Mock the pipe operator chain construction
            # The chain is: prompt | llm | StrOutputParser()
            # First: prompt | llm creates an intermediate chain
            # Then: intermediate_chain | StrOutputParser() creates final chain
            mock_intermediate_chain = MagicMock()
            mock_intermediate_chain.__or__.return_value = mock_final_chain
            
            mock_llm = MagicMock()
            mock_gemini.return_value = mock_llm
            
            generator = QAGenerator(llm_type="gemini")
            # Mock the prompt to support pipe operations: prompt | llm
            generator.no_context_prompt = MagicMock()
            generator.no_context_prompt.__or__.return_value = mock_intermediate_chain
            
            answer = generator.generate_without_retrieval("test question")
            
            assert answer == "Test answer"
            mock_final_chain.invoke.assert_called_once()


class TestQASystem:
    """Test QASystem class"""
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    @patch('qa_system.RetrievalQA')
    def test_init_with_retrieval(self, mock_qa, mock_gemini):
        """Test QASystem initialization with retrieval"""
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        mock_retriever_wrapper = Mock()
        mock_retriever = Mock()
        mock_retriever_wrapper.get_retriever.return_value = mock_retriever
        
        mock_qa_chain = Mock()
        mock_qa.from_chain_type.return_value = mock_qa_chain
        
        generator = QAGenerator(llm_type="gemini")
        qa_system = QASystem(mock_retriever_wrapper, generator, use_retrieval=True)
        
        assert qa_system.use_retrieval is True
        assert qa_system.qa_chain is not None
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    def test_init_without_retrieval(self, mock_gemini):
        """Test QASystem initialization without retrieval"""
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        generator = QAGenerator(llm_type="gemini")
        qa_system = QASystem(None, generator, use_retrieval=False)
        
        assert qa_system.use_retrieval is False
        assert qa_system.qa_chain is None
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    @patch('qa_system.RetrievalQA')
    def test_answer_with_retrieval(self, mock_qa, mock_gemini):
        """Test answer method with retrieval"""
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        mock_retriever_wrapper = Mock()
        mock_retriever = Mock()
        mock_retriever_wrapper.get_retriever.return_value = mock_retriever
        
        mock_qa_chain = Mock()
        mock_qa.from_chain_type.return_value = mock_qa_chain
        
        mock_doc = Document(page_content="Test", metadata={'filename': 'test.txt'})
        mock_qa_chain.invoke.return_value = {
            'result': 'Test answer',
            'source_documents': [mock_doc]
        }
        
        generator = QAGenerator(llm_type="gemini")
        qa_system = QASystem(mock_retriever_wrapper, generator, use_retrieval=True)
        
        answer, info = qa_system.answer("test question")
        
        assert answer == "Test answer"
        assert info is not None
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    def test_answer_without_retrieval(self, mock_gemini):
        """Test answer method without retrieval"""
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        generator = QAGenerator(llm_type="gemini")
        qa_system = QASystem(None, generator, use_retrieval=False)
        
        with patch.object(generator, 'generate_without_retrieval', return_value="Test answer"):
            answer, info = qa_system.answer("test question")
            
            assert answer == "Test answer"
            assert info is None


class TestLoadQuestions:
    """Test load_questions function"""
    
    def test_load_questions_valid_file(self):
        """Test loading questions from a valid TSV file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv', encoding='utf-8') as f:
            f.write("Question 1\ttype1\n")
            f.write("Question 2\ttype2\n")
            f.write("Question 3\ttype3\n")
            temp_path = f.name
        
        try:
            questions = load_questions(temp_path)
            assert len(questions) == 3
            assert questions[0] == ("Question 1", "type1")
            assert questions[1] == ("Question 2", "type2")
            assert questions[2] == ("Question 3", "type3")
        finally:
            os.unlink(temp_path)
    
    def test_load_questions_empty_file(self):
        """Test loading questions from an empty file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv', encoding='utf-8') as f:
            temp_path = f.name
        
        try:
            questions = load_questions(temp_path)
            assert len(questions) == 0
        finally:
            os.unlink(temp_path)
    
    def test_load_questions_invalid_format(self):
        """Test loading questions with invalid format (single column)"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv', encoding='utf-8') as f:
            f.write("Question 1\n")
            f.write("Question 2\ttype2\n")
            temp_path = f.name
        
        try:
            questions = load_questions(temp_path)
            # Should only load valid lines
            assert len(questions) == 1
            assert questions[0] == ("Question 2", "type2")
        finally:
            os.unlink(temp_path)


class TestIntegration:
    """Integration tests"""
    
    @patch('qa_system.ChatGoogleGenerativeAI')
    @patch('qa_system.OpenAIEmbeddings')
    @patch('qa_system.FAISS')
    @patch('qa_system.DirectoryLoader')
    def test_full_pipeline_vector_gemini(self, mock_loader, mock_faiss, mock_embeddings, mock_gemini):
        """Test full pipeline with vector retriever and Gemini"""
        # Setup mocks
        mock_docs = [
            Document(page_content="Test content", metadata={'source': 'test.txt', 'filename': 'test.txt'})
        ]
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = mock_docs
        mock_loader.return_value = mock_loader_instance
        
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_faiss.from_documents.return_value = mock_vectorstore
        
        mock_llm = Mock()
        mock_gemini.return_value = mock_llm
        
        # Create system
        doc_store = DocumentStore(doc_dir="test_dir")
        documents = doc_store.get_documents()
        
        retriever_wrapper = VectorStoreRetriever(documents)
        generator = QAGenerator(llm_type="gemini")
        qa_system = QASystem(retriever_wrapper, generator, use_retrieval=True)
        
        # Verify components are initialized
        assert qa_system.retriever_wrapper is not None
        assert qa_system.generator is not None
        assert qa_system.use_retrieval is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

