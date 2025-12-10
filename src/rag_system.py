"""
RAG System Implementation: A Complete Teaching Framework
INFO 7390 - Advanced Data Science and Architecture

This implementation provides a fully functional RAG system with:
- Document processing and chunking
- Embedding generation
- Vector storage and retrieval
- LLM integration for answer generation
- Evaluation metrics and visualization
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# NOTE: In production, install these packages:
# pip install sentence-transformers chromadb openai tiktoken

@dataclass
class Document:
    """Represents a source document with metadata"""
    text: str
    metadata: Dict[str, any]
    doc_id: str

@dataclass
class Chunk:
    """Represents a text chunk with embedding"""
    text: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, any]
    chunk_id: str
    doc_id: str

@dataclass
class RetrievalResult:
    """Represents retrieved chunks with similarity scores"""
    chunks: List[Chunk]
    scores: List[float]
    query: str


class DocumentProcessor:
    """
    Handles document loading and chunking strategies
    
    Teaching Point: Different chunking strategies affect retrieval quality
    - Fixed-size: Simple but may break context
    - Sentence-based: Preserves meaning but variable size
    - Semantic: Groups related sentences (advanced)
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, texts: List[str], metadatas: List[Dict] = None) -> List[Document]:
        """
        Load documents from text and metadata
        
        Teaching Point: Metadata is crucial for filtering and attribution
        """
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        documents = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            doc = Document(
                text=text,
                metadata=metadata,
                doc_id=f"doc_{i}"
            )
            documents.append(doc)
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split documents into overlapping chunks
        
        Teaching Point: Overlap helps preserve context at boundaries
        Strategy: Sliding window with configurable size and overlap
        """
        chunks = []
        chunk_counter = 0
        
        for doc in documents:
            text = doc.text
            start = 0
            
            while start < len(text):
                # Define chunk boundaries
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                # Create chunk with inherited metadata
                chunk = Chunk(
                    text=chunk_text,
                    embedding=None,  # Will be added later
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_counter,
                        "start_char": start,
                        "end_char": end
                    },
                    chunk_id=f"chunk_{chunk_counter}",
                    doc_id=doc.doc_id
                )
                chunks.append(chunk)
                
                # Move window with overlap
                start += self.chunk_size - self.chunk_overlap
                chunk_counter += 1
        
        return chunks


class EmbeddingGenerator:
    """
    Generates vector embeddings for text chunks
    
    Teaching Point: Embeddings capture semantic meaning in vector space
    Similar texts have similar vectors (measured by cosine similarity)
    """
    
    def __init__(self, model_name: str = "mock"):
        """
        In production, use: 'sentence-transformers/all-MiniLM-L6-v2'
        For teaching, we'll use mock embeddings
        """
        self.model_name = model_name
        self.embedding_dim = 384  # Standard dimension for MiniLM
        
        # For demonstration, we'll create deterministic mock embeddings
        # In production: from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Teaching Point: Same text always produces same embedding (deterministic)
        We use a simple hash-based mock for teaching purposes
        """
        # Mock embedding based on text hash (for demonstration)
        # In production: return self.model.encode(text)
        
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        # Normalize to unit vector (important for cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate embeddings for all chunks
        
        Teaching Point: Batch processing is more efficient in production
        """
        embedded_chunks = []
        for chunk in chunks:
            embedding = self.embed_text(chunk.text)
            chunk.embedding = embedding
            embedded_chunks.append(chunk)
        
        return embedded_chunks


class VectorStore:
    """
    Stores and retrieves embeddings using similarity search
    
    Teaching Point: Vector databases enable fast semantic search
    Key operation: Find k-nearest neighbors in embedding space
    """
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add_chunks(self, chunks: List[Chunk]):
        """
        Add chunks to the vector store
        
        Teaching Point: We store both embeddings and original text/metadata
        """
        self.chunks.extend(chunks)
        
        # Stack embeddings into matrix for efficient similarity computation
        embeddings_list = [chunk.embedding for chunk in chunks]
        new_embeddings = np.vstack(embeddings_list)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> Tuple[List[Chunk], List[float]]:
        """
        Find k most similar chunks using cosine similarity
        
        Teaching Point: Cosine similarity measures angle between vectors
        Range: [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return [], []
        
        # Compute cosine similarity with all stored embeddings
        # Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        # Since vectors are normalized, this simplifies to dot product
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Retrieve corresponding chunks and scores
        top_chunks = [self.chunks[i] for i in top_k_indices]
        top_scores = [float(similarities[i]) for i in top_k_indices]
        
        return top_chunks, top_scores
    
    def get_stats(self) -> Dict:
        """Return statistics about the vector store"""
        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "unique_documents": len(set(chunk.doc_id for chunk in self.chunks))
        }


class RAGSystem:
    """
    Complete RAG system integrating all components
    
    Teaching Point: RAG = Retrieval + Augmentation + Generation
    This orchestrates the entire pipeline
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, top_k: int = 3):
        """
        Initialize RAG system components
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve per query
        """
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.top_k = top_k
        
    def ingest_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """
        Process and index documents into the RAG system
        
        Teaching Point: This is the "indexing" phase - done once per document
        Steps: Load -> Chunk -> Embed -> Store
        """
        print("📄 Loading documents...")
        documents = self.processor.load_documents(texts, metadatas)
        
        print(f"✂️  Chunking {len(documents)} documents...")
        chunks = self.processor.chunk_documents(documents)
        
        print(f"🧮 Generating embeddings for {len(chunks)} chunks...")
        embedded_chunks = self.embedder.embed_chunks(chunks)
        
        print("💾 Adding to vector store...")
        self.vector_store.add_chunks(embedded_chunks)
        
        print(f"✅ Indexed {len(documents)} documents into {len(chunks)} chunks")
        return self.vector_store.get_stats()
    
    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query
        
        Teaching Point: This is the "retrieval" phase
        Query goes through same embedding process as documents
        """
        # Embed the query using same model as documents
        query_embedding = self.embedder.embed_text(query)
        
        # Search for similar chunks
        chunks, scores = self.vector_store.similarity_search(query_embedding, self.top_k)
        
        return RetrievalResult(chunks=chunks, scores=scores, query=query)
    
    def generate_answer(self, query: str, retrieval_result: RetrievalResult) -> Dict:
        """
        Generate answer using retrieved context
        
        Teaching Point: This is the "generation" phase
        In production, this calls an LLM API (OpenAI, Anthropic, etc.)
        For teaching, we create a structured response
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, (chunk, score) in enumerate(zip(retrieval_result.chunks, retrieval_result.scores)):
            context_parts.append(
                f"[Source {i+1}] (Relevance: {score:.2f})\n{chunk.text}\n"
            )
        
        context = "\n".join(context_parts)
        
        # In production, you would call an LLM here:
        # prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        # response = openai.ChatCompletion.create(...)
        
        # For teaching demonstration:
        answer = f"""Based on the retrieved context, here's a synthesized answer to: "{query}"

The system retrieved {len(retrieval_result.chunks)} relevant passages with an average relevance score of {np.mean(retrieval_result.scores):.2f}.

[In production, this would be a natural language answer generated by an LLM using the retrieved context]"""
        
        return {
            "query": query,
            "answer": answer,
            "context": context,
            "num_chunks_used": len(retrieval_result.chunks),
            "retrieval_scores": retrieval_result.scores,
            "sources": [chunk.metadata for chunk in retrieval_result.chunks]
        }
    
    def query(self, question: str, return_context: bool = True) -> Dict:
        """
        End-to-end RAG query: retrieve + generate
        
        Teaching Point: This is what users interact with
        Input: question, Output: answer with sources
        """
        # Step 1: Retrieve relevant chunks
        retrieval_result = self.retrieve(question)
        
        # Step 2: Generate answer using retrieved context
        response = self.generate_answer(question, retrieval_result)
        
        if not return_context:
            response.pop('context', None)
        
        return response


class RAGEvaluator:
    """
    Evaluate RAG system performance
    
    Teaching Point: Evaluation is critical for improving RAG systems
    Metrics track both retrieval quality and generation quality
    """
    
    @staticmethod
    def calculate_retrieval_metrics(retrieved_chunks: List[Chunk], 
                                   relevant_doc_ids: List[str]) -> Dict[str, float]:
        """
        Calculate retrieval metrics
        
        Teaching Point: 
        - Precision: What fraction of retrieved chunks are relevant?
        - Recall: What fraction of relevant chunks were retrieved?
        - F1: Harmonic mean of precision and recall
        """
        retrieved_doc_ids = [chunk.doc_id for chunk in retrieved_chunks]
        
        # True positives: relevant docs that were retrieved
        true_positives = len(set(retrieved_doc_ids) & set(relevant_doc_ids))
        
        # Precision: TP / (TP + FP)
        precision = true_positives / len(retrieved_doc_ids) if retrieved_doc_ids else 0
        
        # Recall: TP / (TP + FN)
        recall = true_positives / len(relevant_doc_ids) if relevant_doc_ids else 0
        
        # F1 Score: Harmonic mean
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "num_retrieved": len(retrieved_doc_ids),
            "num_relevant": len(relevant_doc_ids)
        }
    
    @staticmethod
    def calculate_mrr(retrieved_chunks: List[Chunk], 
                     relevant_doc_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Teaching Point: MRR measures how quickly we find the first relevant result
        Formula: 1 / rank_of_first_relevant_item
        """
        retrieved_doc_ids = [chunk.doc_id for chunk in retrieved_chunks]
        
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank
        
        return 0.0  # No relevant documents found


# ============================================================================
# DEMONSTRATION AND TEACHING EXAMPLES
# ============================================================================

def demo_basic_rag():
    """
    Demonstration 1: Basic RAG workflow
    
    Teaching Objective: Show the complete pipeline in action
    """
    print("=" * 70)
    print("DEMO 1: Basic RAG Workflow")
    print("=" * 70)
    
    # Sample documents about Data Science
    documents = [
        "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by connecting them to external knowledge bases. This allows the model to retrieve relevant information before generating a response, leading to more accurate and up-to-date answers.",
        
        "The RAG pipeline consists of three main components: document processing and embedding, semantic retrieval using vector similarity, and answer generation using retrieved context. Each component plays a crucial role in the system's overall performance.",
        
        "Vector embeddings transform text into high-dimensional numerical representations that capture semantic meaning. Similar concepts are located close together in the embedding space, enabling semantic search capabilities.",
        
        "Common challenges in RAG systems include chunk size selection, handling context length limits, and ensuring retrieval quality. The choice of embedding model and chunking strategy significantly impacts system performance.",
        
        "Evaluation of RAG systems requires both retrieval metrics (precision, recall) and generation quality metrics (faithfulness, relevance). Comprehensive evaluation helps identify areas for improvement."
    ]
    
    metadatas = [
        {"source": "rag_intro.pdf", "page": 1},
        {"source": "rag_intro.pdf", "page": 2},
        {"source": "embeddings_guide.pdf", "page": 5},
        {"source": "rag_challenges.pdf", "page": 3},
        {"source": "evaluation_guide.pdf", "page": 1}
    ]
    
    # Initialize RAG system
    rag = RAGSystem(chunk_size=300, chunk_overlap=50, top_k=3)
    
    # Ingest documents
    stats = rag.ingest_documents(documents, metadatas)
    print(f"\n📊 Vector Store Stats: {stats}\n")
    
    # Query the system
    questions = [
        "What is RAG and why is it useful?",
        "How do you evaluate a RAG system?",
        "What are vector embeddings?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 70)
        
        result = rag.query(question, return_context=False)
        
        print(f"\n📊 Retrieved {result['num_chunks_used']} chunks")
        print(f"📈 Relevance scores: {[f'{s:.3f}' for s in result['retrieval_scores']]}")
        print(f"\n💡 Answer:\n{result['answer']}\n")
        print(f"📚 Sources: {result['sources']}\n")


def demo_chunking_strategies():
    """
    Demonstration 2: Impact of different chunking strategies
    
    Teaching Objective: Show how chunk size affects retrieval
    """
    print("=" * 70)
    print("DEMO 2: Chunking Strategy Comparison")
    print("=" * 70)
    
    sample_text = """Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future. The primary aim is to allow computers to learn automatically without human intervention or assistance and adjust actions accordingly."""
    
    chunk_sizes = [50, 100, 200]
    
    for size in chunk_sizes:
        processor = DocumentProcessor(chunk_size=size, chunk_overlap=10)
        documents = processor.load_documents([sample_text])
        chunks = processor.chunk_documents(documents)
        
        print(f"\n📏 Chunk size: {size} characters")
        print(f"   Number of chunks created: {len(chunks)}")
        print(f"   Average chunk length: {np.mean([len(c.text) for c in chunks]):.1f}")
        print(f"   First chunk preview: {chunks[0].text[:100]}...")


def demo_similarity_search():
    """
    Demonstration 3: Visualizing similarity scores
    
    Teaching Objective: Show how semantic similarity works
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Semantic Similarity Visualization")
    print("=" * 70)
    
    embedder = EmbeddingGenerator()
    
    # Create sample texts with varying similarity
    texts = [
        "Machine learning is a branch of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "The weather today is sunny and warm",
        "AI systems can learn from data without explicit programming"
    ]
    
    query = "What is artificial intelligence?"
    query_embedding = embedder.embed_text(query)
    
    print(f"\n🔍 Query: '{query}'\n")
    print("Similarity scores with different texts:")
    print("-" * 70)
    
    for i, text in enumerate(texts, 1):
        text_embedding = embedder.embed_text(text)
        similarity = np.dot(query_embedding, text_embedding)
        
        # Visual representation
        bar_length = int(similarity * 50)
        bar = "█" * bar_length
        
        print(f"{i}. [{similarity:.3f}] {bar}")
        print(f"   Text: {text}\n")


# Main execution
if __name__ == "__main__":
    print("\n🎓 RAG System Teaching Implementation")
    print("INFO 7390 - Advanced Data Science and Architecture\n")
    
    # Run all demonstrations
    demo_basic_rag()
    demo_chunking_strategies()
    demo_similarity_search()
    
    print("\n" + "=" * 70)
    print("✅ All demonstrations complete!")
    print("=" * 70)
    print("\n💡 Next steps for learners:")
    print("1. Modify chunk_size and observe effects on retrieval")
    print("2. Add your own documents and queries")
    print("3. Implement real embedding models (sentence-transformers)")
    print("4. Connect to an LLM API for actual answer generation")
    print("5. Add evaluation metrics to track improvements")
    print("\n🔗 See README.md for exercises and additional resources\n")