"""
RAG Starter Template - Build Your Own RAG System
INFO 7390 - Learning Exercise

TODO: Complete the missing functions marked with TODO comments
Follow the hints and refer to the main implementation for guidance

Learning Objectives:
1. Understand document chunking strategies
2. Implement similarity search
3. Build an end-to-end RAG pipeline
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    embedding: np.ndarray
    metadata: Dict
    chunk_id: str


class SimpleRAG:
    """
    A simplified RAG system for learning
    Complete the TODOs to make it functional!
    """
    
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size
        self.chunks: List[Chunk] = []
        self.embeddings = None
    
    def create_chunks(self, text: str) -> List[str]:
        """
        TODO: Split text into chunks of size self.chunk_size
        
        Hint: Use a sliding window approach
        - Start at position 0
        - Take chunk_size characters
        - Move forward by chunk_size characters
        - Repeat until end of text
        
        Example:
            text = "Hello world this is a test"
            chunk_size = 10
            Result: ["Hello worl", "d this is ", "a test"]
        """
        chunks = []
        
        # YOUR CODE HERE
        # Step 1: Initialize start position
        # Step 2: Loop while start < len(text)
        # Step 3: Extract chunk from start to start+chunk_size
        # Step 4: Add to chunks list
        # Step 5: Move start forward
        
        return chunks
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        TODO: Create a simple embedding for text
        
        Hint: For this exercise, use a deterministic hash-based approach
        - Use hash(text) to get a number
        - Create random vector with that number as seed
        - Normalize to unit length
        
        This simulates real embeddings while being reproducible
        """
        embedding_dim = 128
        
        # YOUR CODE HERE
        # Step 1: Get hash of text
        # Step 2: Set numpy random seed
        # Step 3: Generate random vector
        # Step 4: Normalize to unit length (divide by norm)
        
        return None  # Replace with your embedding
    
    def add_documents(self, texts: List[str]):
        """
        TODO: Process documents and add to system
        
        Hint: For each document:
        1. Split into chunks
        2. Create embedding for each chunk
        3. Store chunk with its embedding
        """
        
        # YOUR CODE HERE
        # For each text in texts:
        #   - Call create_chunks()
        #   - For each chunk:
        #       - Create embedding
        #       - Create Chunk object
        #       - Add to self.chunks
        
        # After processing all chunks:
        # Stack all embeddings into self.embeddings matrix
        
        pass
    
    def search(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        TODO: Find k most similar chunks to query
        
        Hint: 
        1. Create embedding for query
        2. Calculate cosine similarity with all stored embeddings
        3. Find top k highest similarities
        4. Return corresponding chunks and scores
        
        Cosine similarity formula:
            similarity = dot_product(query_emb, doc_emb)
            (assumes embeddings are normalized)
        """
        
        # YOUR CODE HERE
        # Step 1: Create query embedding
        # Step 2: Compute similarities (dot product with all embeddings)
        # Step 3: Find top k indices using np.argsort()
        # Step 4: Return chunks and scores
        
        return []
    
    def query(self, question: str) -> Dict:
        """
        TODO: Complete RAG query
        
        Hint:
        1. Search for relevant chunks
        2. Combine chunk texts into context
        3. Return formatted response
        """
        
        # YOUR CODE HERE
        # Step 1: Call search() to get relevant chunks
        # Step 2: Build context string from chunks
        # Step 3: Create response dictionary
        
        return {
            "question": question,
            "context": "",
            "num_chunks": 0,
            "scores": []
        }


# ============================================================================
# EXERCISES - Complete these to test your understanding
# ============================================================================

def exercise_1_chunking():
    """
    Exercise 1: Experiment with chunking
    
    Task: Create chunks of different sizes and observe the results
    Questions to explore:
    - What happens with very small chunks (e.g., 50 chars)?
    - What happens with very large chunks (e.g., 1000 chars)?
    - Which size works best for this text?
    """
    
    sample_text = """Retrieval-Augmented Generation combines retrieval systems with 
    generative language models. The retrieval component searches a knowledge base for 
    relevant information, while the generation component uses this context to produce 
    accurate responses. This approach addresses the limitation of language models having 
    outdated or incomplete training data."""
    
    rag = SimpleRAG(chunk_size=100)
    
    # TODO: Test different chunk sizes
    # Try: 50, 100, 200, 500
    # Print results and analyze
    
    print("Exercise 1: Chunking Strategies")
    print("=" * 50)
    
    # YOUR CODE HERE


def exercise_2_similarity():
    """
    Exercise 2: Understanding similarity
    
    Task: Calculate similarity between different text pairs
    Questions to explore:
    - Are similar texts close in embedding space?
    - Are unrelated texts far apart?
    - How does text length affect similarity?
    """
    
    texts = [
        "Machine learning is a type of artificial intelligence",
        "AI systems can learn from data",
        "The weather is nice today",
        "Deep learning uses neural networks"
    ]
    
    print("\nExercise 2: Similarity Analysis")
    print("=" * 50)
    
    # TODO: Create embeddings for each text
    # TODO: Calculate pairwise similarities
    # TODO: Print similarity matrix
    
    # YOUR CODE HERE


def exercise_3_end_to_end():
    """
    Exercise 3: Build complete RAG system
    
    Task: Create a working RAG system with your own data
    Requirements:
    - Add at least 3 documents
    - Query with 3 different questions
    - Analyze which chunks are retrieved for each query
    """
    
    print("\nExercise 3: End-to-End RAG")
    print("=" * 50)
    
    # TODO: Create your own documents about a topic you know
    # TODO: Initialize RAG system
    # TODO: Add documents
    # TODO: Query the system
    # TODO: Analyze results
    
    # YOUR CODE HERE


def exercise_4_evaluation():
    """
    Exercise 4: Evaluate retrieval quality
    
    Task: Implement basic evaluation metrics
    Calculate:
    - Precision: How many retrieved chunks are relevant?
    - Coverage: Did we retrieve all important information?
    """
    
    print("\nExercise 4: Evaluation Metrics")
    print("=" * 50)
    
    # TODO: Create test cases with known relevant documents
    # TODO: Retrieve chunks for test queries
    # TODO: Calculate precision
    # TODO: Compare different chunk sizes or k values
    
    # YOUR CODE HERE


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_implementation():
    """
    Test your implementation with sample data
    """
    print("\n🧪 Testing Your Implementation")
    print("=" * 50)
    
    # Test 1: Chunking
    rag = SimpleRAG(chunk_size=50)
    test_text = "This is a test. " * 10
    chunks = rag.create_chunks(test_text)
    
    print(f"✓ Chunking: Created {len(chunks)} chunks")
    assert len(chunks) > 0, "❌ Chunking failed: No chunks created"
    
    # Test 2: Embeddings
    embedding = rag.create_embedding("test text")
    print(f"✓ Embeddings: Shape {embedding.shape if embedding is not None else 'None'}")
    assert embedding is not None, "❌ Embedding failed: Returned None"
    
    # Test 3: Adding documents
    rag.add_documents(["Document 1", "Document 2"])
    print(f"✓ Added {len(rag.chunks)} chunks to system")
    
    # Test 4: Search
    results = rag.search("test query", k=2)
    print(f"✓ Search: Retrieved {len(results)} results")
    
    print("\n✅ All basic tests passed!")


if __name__ == "__main__":
    print("🎓 RAG Starter Template - Learning Exercises")
    print("INFO 7390 - Advanced Data Science\n")
    
    # Uncomment as you complete each exercise
    
    # exercise_1_chunking()
    # exercise_2_similarity()
    # exercise_3_end_to_end()
    # exercise_4_evaluation()
    
    # Run this to test your implementation
    # test_implementation()
    
    print("\n" + "=" * 50)
    print("📚 Learning Resources:")
    print("1. Main implementation: rag_system.py")
    print("2. Tutorial documentation: tutorial.md")
    print("3. Video walkthrough: [link in README]")
    print("=" * 50)