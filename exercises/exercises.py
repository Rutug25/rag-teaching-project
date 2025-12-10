"""
RAG System Exercises and Solutions
INFO 7390 - Advanced Data Science and Architecture

Complete these exercises to master RAG implementation!
Solutions are provided at the bottom of each exercise.
"""

import numpy as np
from typing import List, Dict, Tuple
import json
from collections import defaultdict

# ============================================================================
# EXERCISE 1: CHUNKING STRATEGIES ⭐
# Difficulty: Beginner | Time: 30-45 minutes
# ============================================================================

print("=" * 70)
print("EXERCISE 1: Comparing Chunking Strategies")
print("=" * 70)

"""
Objective: Understand how different chunking strategies affect retrieval

Tasks:
1. Implement three chunking functions: fixed-size, sentence-based, paragraph-based
2. Apply each to a sample document
3. Analyze the results (number of chunks, sizes, etc.)
4. Discuss which strategy is best for different use cases

Learning Goals:
- Understand chunking tradeoffs
- Learn to analyze text structure
- Make informed design decisions
"""

sample_document = """
Retrieval-Augmented Generation is a powerful technique in modern AI systems. 
It combines the strengths of retrieval systems with generative language models.

The process begins with document preparation. Documents are split into smaller 
chunks that can be efficiently stored and retrieved. Each chunk is then converted 
into a vector embedding that captures its semantic meaning.

When a user asks a question, the system converts the question into an embedding. 
It then searches for the most similar document chunks using vector similarity. 
Finally, these relevant chunks are provided as context to a language model, 
which generates a grounded response.

This approach offers several advantages. It allows models to access current 
information without retraining. It reduces hallucinations by grounding responses 
in actual documents. And it provides transparency through source attribution.
"""

# TODO 1: Implement fixed-size chunking
def chunk_fixed_size(text: str, size: int = 200, overlap: int = 50) -> List[str]:
    """
    Split text into fixed-size chunks with overlap
    
    Args:
        text: Input text to chunk
        size: Size of each chunk in characters
        overlap: Number of overlapping characters
        
    Returns:
        List of text chunks
    """
    # YOUR CODE HERE
    pass


# TODO 2: Implement sentence-based chunking
def chunk_by_sentences(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """
    Split text into chunks of N sentences
    
    Args:
        text: Input text to chunk
        sentences_per_chunk: Number of sentences per chunk
        
    Returns:
        List of text chunks
    """
    # YOUR CODE HERE
    # Hint: Use text.split('. ') as a simple sentence splitter
    pass


# TODO 3: Implement paragraph-based chunking
def chunk_by_paragraphs(text: str) -> List[str]:
    """
    Split text by paragraphs (double newlines)
    
    Args:
        text: Input text to chunk
        
    Returns:
        List of text chunks (one per paragraph)
    """
    # YOUR CODE HERE
    # Hint: Use text.split('\n\n')
    pass


# TODO 4: Compare strategies
def compare_chunking_strategies(text: str):
    """
    Apply all three strategies and compare results
    """
    print("\n📊 Chunking Strategy Comparison")
    print("-" * 70)
    
    # YOUR CODE HERE
    # Apply each function and print statistics:
    # - Number of chunks created
    # - Average chunk size
    # - Min/max chunk sizes
    # - Sample of first chunk
    pass


# ============================================================================
# SOLUTION 1
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTION 1")
print("=" * 70)

def chunk_fixed_size_solution(text: str, size: int = 200, overlap: int = 50) -> List[str]:
    """Split text into fixed-size chunks with overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        start += size - overlap
    
    return chunks


def chunk_by_sentences_solution(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """Split text into chunks of N sentences"""
    # Simple sentence splitting (in production, use spaCy or NLTK)
    sentences = text.replace('\n', ' ').split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    
    return chunks


def chunk_by_paragraphs_solution(text: str) -> List[str]:
    """Split text by paragraphs"""
    paragraphs = text.split('\n\n')
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks


def compare_chunking_strategies_solution(text: str):
    """Compare all three strategies"""
    strategies = {
        "Fixed-size (200 chars)": chunk_fixed_size_solution(text, 200, 50),
        "Sentence-based (3 sent)": chunk_by_sentences_solution(text, 3),
        "Paragraph-based": chunk_by_paragraphs_solution(text)
    }
    
    for name, chunks in strategies.items():
        sizes = [len(c) for c in chunks]
        print(f"\n{name}:")
        print(f"  Chunks created: {len(chunks)}")
        print(f"  Avg size: {np.mean(sizes):.1f} chars")
        print(f"  Min/Max: {min(sizes)} / {max(sizes)} chars")
        print(f"  First chunk preview: {chunks[0][:80]}...")


# Run solution
compare_chunking_strategies_solution(sample_document)


# ============================================================================
# EXERCISE 2: SEMANTIC SIMILARITY ⭐⭐
# Difficulty: Intermediate | Time: 45-60 minutes
# ============================================================================

print("\n\n" + "=" * 70)
print("EXERCISE 2: Understanding Semantic Similarity")
print("=" * 70)

"""
Objective: Implement and visualize semantic similarity

Tasks:
1. Create mock embeddings for text samples
2. Calculate cosine similarity between pairs
3. Find most similar texts for a query
4. Visualize similarity matrix

Learning Goals:
- Understand vector similarity
- Practice working with embeddings
- Learn to interpret similarity scores
"""

texts = [
    "RAG combines retrieval and generation",
    "Vector embeddings capture semantic meaning",
    "Machine learning models learn from data",
    "Semantic search finds similar documents",
    "The weather is sunny today",
    "Retrieval systems find relevant information"
]

# TODO 1: Create embeddings
def create_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Create a deterministic embedding for text
    
    Args:
        text: Input text
        dim: Embedding dimension
        
    Returns:
        Normalized embedding vector
    """
    # YOUR CODE HERE
    # Hint: Use hash(text) as seed for np.random
    # Then normalize the vector
    pass


# TODO 2: Calculate cosine similarity
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1, vec2: Input vectors (assumed normalized)
        
    Returns:
        Similarity score between -1 and 1
    """
    # YOUR CODE HERE
    # Hint: For normalized vectors, similarity = dot product
    pass


# TODO 3: Build similarity matrix
def build_similarity_matrix(texts: List[str]) -> np.ndarray:
    """
    Create pairwise similarity matrix
    
    Args:
        texts: List of text strings
        
    Returns:
        NxN similarity matrix
    """
    # YOUR CODE HERE
    pass


# TODO 4: Find most similar
def find_most_similar(query: str, texts: List[str], k: int = 3) -> List[Tuple[str, float]]:
    """
    Find k most similar texts to query
    
    Args:
        query: Query text
        texts: List of candidate texts
        k: Number of results
        
    Returns:
        List of (text, similarity) tuples
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# SOLUTION 2
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTION 2")
print("=" * 70)

def create_embedding_solution(text: str, dim: int = 128) -> np.ndarray:
    """Create deterministic embedding"""
    np.random.seed(hash(text) % (2**32))
    embedding = np.random.randn(dim)
    # Normalize to unit length
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def cosine_similarity_solution(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity"""
    return float(np.dot(vec1, vec2))


def build_similarity_matrix_solution(texts: List[str]) -> np.ndarray:
    """Build similarity matrix"""
    n = len(texts)
    embeddings = [create_embedding_solution(t) for t in texts]
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity_solution(embeddings[i], embeddings[j])
    
    return matrix


def find_most_similar_solution(query: str, texts: List[str], k: int = 3) -> List[Tuple[str, float]]:
    """Find most similar texts"""
    query_emb = create_embedding_solution(query)
    similarities = []
    
    for text in texts:
        text_emb = create_embedding_solution(text)
        sim = cosine_similarity_solution(query_emb, text_emb)
        similarities.append((text, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


# Run solution
print("\n🔍 Query: 'How does retrieval work?'")
query = "How does retrieval work?"
results = find_most_similar_solution(query, texts, k=3)

print("\nTop 3 most similar:")
for i, (text, score) in enumerate(results, 1):
    print(f"{i}. [{score:.3f}] {text}")

print("\n📊 Similarity Matrix:")
matrix = build_similarity_matrix_solution(texts[:4])  # First 4 for readability
print("     ", " ".join([f"T{i}" for i in range(1, 5)]))
for i, row in enumerate(matrix):
    print(f"T{i+1}:  ", " ".join([f"{val:.2f}" for val in row]))


# ============================================================================
# EXERCISE 3: END-TO-END RAG ⭐⭐⭐
# Difficulty: Advanced | Time: 1-2 hours
# ============================================================================

print("\n\n" + "=" * 70)
print("EXERCISE 3: Build Complete RAG System")
print("=" * 70)

"""
Objective: Integrate all components into working RAG system

Tasks:
1. Build document indexing pipeline
2. Implement query processing
3. Add answer generation (mock)
4. Test with multiple queries
5. Analyze retrieval quality

Learning Goals:
- Integrate multiple components
- Handle end-to-end workflow
- Practice system design
- Evaluate performance
"""

class MiniRAG:
    """
    Mini RAG system - complete this implementation!
    """
    
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = None
    
    # TODO 1: Implement indexing
    def index_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Index documents for retrieval
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        # YOUR CODE HERE
        # Steps:
        # 1. Chunk each document
        # 2. Create embeddings
        # 3. Store chunks with metadata
        pass
    
    # TODO 2: Implement retrieval
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve top-k relevant chunks
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            List of dicts with 'text', 'score', 'metadata'
        """
        # YOUR CODE HERE
        pass
    
    # TODO 3: Implement query
    def query(self, question: str) -> Dict:
        """
        Complete RAG query
        
        Args:
            question: User question
            
        Returns:
            Dict with 'question', 'answer', 'sources'
        """
        # YOUR CODE HERE
        # Steps:
        # 1. Retrieve relevant chunks
        # 2. Build context from chunks
        # 3. Generate answer (mock)
        # 4. Return formatted response
        pass


# ============================================================================
# SOLUTION 3
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTION 3")
print("=" * 70)

class MiniRAGSolution:
    """Complete Mini RAG implementation"""
    
    def __init__(self, chunk_size: int = 200):
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = None
    
    def index_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Index documents"""
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]
        
        self.chunks = []
        all_embeddings = []
        
        for doc, meta in zip(documents, metadatas):
            # Chunk document
            doc_chunks = chunk_fixed_size_solution(doc, self.chunk_size, 50)
            
            for i, chunk_text in enumerate(doc_chunks):
                # Create embedding
                embedding = create_embedding_solution(chunk_text)
                
                # Store chunk
                self.chunks.append({
                    'text': chunk_text,
                    'metadata': {**meta, 'chunk_idx': i}
                })
                all_embeddings.append(embedding)
        
        # Stack embeddings
        self.embeddings = np.vstack(all_embeddings)
        print(f"✅ Indexed {len(documents)} docs → {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant chunks"""
        if not self.chunks:
            return []
        
        # Embed query
        query_emb = create_embedding_solution(query)
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_emb)
        
        # Get top k
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_idx:
            results.append({
                'text': self.chunks[idx]['text'],
                'score': float(similarities[idx]),
                'metadata': self.chunks[idx]['metadata']
            })
        
        return results
    
    def query(self, question: str) -> Dict:
        """Complete RAG query"""
        # Retrieve relevant chunks
        retrieved = self.retrieve(question, k=3)
        
        # Build context
        context = "\n\n".join([
            f"[{i+1}] {r['text']}" 
            for i, r in enumerate(retrieved)
        ])
        
        # Mock answer generation
        answer = f"""Based on {len(retrieved)} retrieved chunks, here's the answer:

{context}

[In production, an LLM would synthesize this context into a natural answer]"""
        
        return {
            'question': question,
            'answer': answer,
            'sources': [r['metadata'] for r in retrieved],
            'retrieval_scores': [r['score'] for r in retrieved]
        }


# Demo solution
rag = MiniRAGSolution(chunk_size=200)

# Index documents
docs = [
    "RAG combines retrieval with language model generation. It enables models to access external knowledge.",
    "Vector embeddings transform text into numerical representations that capture semantic meaning.",
    "Chunking strategies affect retrieval quality. Common approaches include fixed-size and sentence-based.",
]
rag.index_documents(docs)

# Query
result = rag.query("How does RAG work?")
print(f"\n❓ Question: {result['question']}")
print(f"\n📊 Retrieved {len(result['sources'])} chunks")
print(f"📈 Scores: {[f'{s:.3f}' for s in result['retrieval_scores']]}")
print(f"\n💡 Answer:\n{result['answer'][:300]}...")


# ============================================================================
# EXERCISE 4: EVALUATION METRICS ⭐⭐⭐
# Difficulty: Advanced | Time: 1 hour
# ============================================================================

print("\n\n" + "=" * 70)
print("EXERCISE 4: Evaluation Framework")
print("=" * 70)

"""
Objective: Build evaluation framework for RAG systems

Tasks:
1. Implement precision and recall metrics
2. Calculate MRR (Mean Reciprocal Rank)
3. Create test dataset with ground truth
4. Evaluate system performance
5. Suggest improvements based on metrics

Learning Goals:
- Understand evaluation metrics
- Create test datasets
- Analyze system performance
- Make data-driven improvements
"""

# TODO: Implement evaluation metrics
def calculate_precision_at_k(retrieved_ids: List[str], 
                             relevant_ids: List[str]) -> float:
    """Calculate precision@k"""
    # YOUR CODE HERE
    pass


def calculate_recall_at_k(retrieved_ids: List[str], 
                          relevant_ids: List[str]) -> float:
    """Calculate recall@k"""
    # YOUR CODE HERE
    pass


def calculate_mrr(retrieved_ids: List[str], 
                 relevant_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank"""
    # YOUR CODE HERE
    pass


# ============================================================================
# SOLUTION 4
# ============================================================================

print("\n" + "=" * 70)
print("SOLUTION 4")
print("=" * 70)

def calculate_precision_at_k_solution(retrieved_ids: List[str], 
                                     relevant_ids: List[str]) -> float:
    """Calculate precision@k"""
    if not retrieved_ids:
        return 0.0
    
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    
    true_positives = len(retrieved_set & relevant_set)
    return true_positives / len(retrieved_set)


def calculate_recall_at_k_solution(retrieved_ids: List[str], 
                                   relevant_ids: List[str]) -> float:
    """Calculate recall@k"""
    if not relevant_ids:
        return 0.0
    
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    
    true_positives = len(retrieved_set & relevant_set)
    return true_positives / len(relevant_set)


def calculate_mrr_solution(retrieved_ids: List[str], 
                          relevant_ids: List[str]) -> float:
    """Calculate MRR"""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


# Demo evaluation
print("\n📊 Evaluation Example")
print("-" * 70)

# Simulated retrieval results
retrieved = ["doc_2", "doc_5", "doc_1", "doc_8"]
relevant = ["doc_1", "doc_2", "doc_3"]

precision = calculate_precision_at_k_solution(retrieved, relevant)
recall = calculate_recall_at_k_solution(retrieved, relevant)
mrr = calculate_mrr_solution(retrieved, relevant)

print(f"Retrieved: {retrieved}")
print(f"Relevant:  {relevant}")
print(f"\nMetrics:")
print(f"  Precision@4: {precision:.3f} ({precision*100:.1f}%)")
print(f"  Recall@4:    {recall:.3f} ({recall*100:.1f}%)")
print(f"  MRR:         {mrr:.3f}")

# Interpretation
print(f"\n💡 Interpretation:")
print(f"  - Precision: {int(precision*100)}% of retrieved docs are relevant")
print(f"  - Recall: Found {int(recall*100)}% of all relevant docs")
print(f"  - MRR: First relevant doc at rank {int(1/mrr) if mrr > 0 else 'N/A'}")


print("\n\n" + "=" * 70)
print("✅ ALL EXERCISES COMPLETE!")
print("=" * 70)
print("\n🎓 Congratulations! You've mastered RAG fundamentals.")
print("\n📚 Next Steps:")
print("  1. Implement real embedding models (sentence-transformers)")
print("  2. Connect to actual LLM APIs (OpenAI, Anthropic)")
print("  3. Build a web interface (Streamlit, Gradio)")
print("  4. Add advanced features (reranking, hybrid search)")
print("  5. Deploy to production (FastAPI, Docker)")
print("\n🔗 Check out the full tutorial for advanced topics!")
print("=" * 70)