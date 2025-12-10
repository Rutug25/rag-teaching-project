# 🎓 RAG Demystified: Complete Teaching Framework

**Learn Retrieval-Augmented Generation by Building from Scratch**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> *"The best way to learn something is to teach it. The second best way is to build it."*

This educational package teaches **Retrieval-Augmented Generation (RAG)** through hands-on implementation, comprehensive tutorials, and interactive exercises. Designed for **INFO 7390: Advanced Data Science and Architecture**.

## 🎥 Demo Video

**[▶️ Watch the 10-Minute Show-and-Tell Video](https://youtube.com/your-video-link)**

Follow along as we explain, show, and guide you through building your own RAG system!

---

## 📋 Table of Contents

- [What is RAG?](#what-is-rag)
- [Why This Project?](#why-this-project)
- [What You'll Learn](#what-youll-learn)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Learning Path](#learning-path)
- [Features](#features)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🤔 What is RAG?

**Retrieval-Augmented Generation (RAG)** enhances Large Language Models (LLMs) by connecting them to external knowledge bases. Instead of relying solely on training data, RAG:

1. 🔍 **Retrieves** relevant information from documents
2. 🔗 **Augments** the LLM prompt with retrieved context
3. 💬 **Generates** accurate, grounded responses

### Real-World Applications

- **Customer Support**: Answer questions from product documentation
- **Legal Research**: Search case law and statutes
- **Healthcare**: Clinical decision support with medical literature
- **Education**: Course material Q&A systems
- **Enterprise**: Internal knowledge base search

---

## 💡 Why This Project?

This isn't just another RAG tutorial. Here's what makes it special:

✅ **Learn by Building**: Complete working implementation from scratch  
✅ **Progressive Complexity**: Start simple, add features incrementally  
✅ **Production-Ready Patterns**: Industry best practices included  
✅ **Comprehensive Materials**: Tutorial, video, exercises, and starter template  
✅ **Interactive Demo**: See RAG in action with visual feedback  
✅ **Evaluation Framework**: Learn to measure and improve quality  

**Perfect for:**
- Data science students learning LLM applications
- ML engineers building production RAG systems
- Developers transitioning to AI/ML roles
- Anyone curious about how ChatGPT plugins work!

---

## 🎯 What You'll Learn

By completing this project, you'll be able to:

### Core Concepts
- ✓ Understand how RAG works under the hood
- ✓ Explain vector embeddings and semantic search
- ✓ Design effective document chunking strategies
- ✓ Compare different similarity metrics

### Technical Skills
- ✓ Build a RAG system from scratch in Python
- ✓ Generate and store vector embeddings
- ✓ Implement semantic search with vector databases
- ✓ Integrate with LLM APIs (OpenAI, Anthropic)
- ✓ Evaluate retrieval and generation quality

### Advanced Techniques
- ✓ Hybrid search (semantic + keyword)
- ✓ Query expansion and reranking
- ✓ Metadata filtering and optimization
- ✓ Production deployment considerations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Basic understanding of Python and ML concepts
- (Optional) OpenAI or Anthropic API key for LLM integration

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-teaching-project.git
cd rag-teaching-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
python src/rag_system.py
```

### First Steps

```python
from src.rag_system import RAGSystem

# Initialize RAG system
rag = RAGSystem(chunk_size=500, top_k=3)

# Add documents
documents = [
    "RAG combines retrieval with generation...",
    "Vector embeddings represent semantic meaning..."
]
rag.ingest_documents(documents)

# Query the system
result = rag.query("What is RAG?")
print(result['answer'])
print(f"Sources: {result['sources']}")
```

---

## 📁 Project Structure

```
rag-teaching-project/
│
├── 📖 README.md                    # You are here!
├── 📋 requirements.txt             # Python dependencies
├── 📜 LICENSE                      # MIT License
│
├── 📂 src/
│   ├── rag_system.py              # Complete implementation
│   ├── rag_starter.py             # Starter template with TODOs
│   ├── interactive_demo.html      # Interactive web demo
│   └── utils.py                   # Helper functions
│
├── 📂 docs/
│   ├── tutorial.md                # 30-page comprehensive tutorial
│   ├── pedagogical_report.md      # Teaching methodology (6-10 pages)
│   ├── video_script.md            # Show-and-tell video script
│   └── api_reference.md           # Code documentation
│
├── 📂 notebooks/
│   ├── 01_introduction.ipynb      # Getting started
│   ├── 02_embeddings.ipynb        # Understanding embeddings
│   ├── 03_retrieval.ipynb         # Semantic search
│   ├── 04_full_pipeline.ipynb     # Complete RAG system
│   └── 05_evaluation.ipynb        # Metrics and optimization
│
├── 📂 exercises/
│   ├── exercises.py               # Practice exercises with solutions
│   └── solutions/                 # Detailed solutions
│
├── 📂 data/
│   ├── sample_documents/          # Example documents
│   └── evaluation_datasets/       # Test datasets
│
├── 📂 tests/
│   ├── test_chunking.py          # Unit tests
│   ├── test_embeddings.py
│   └── test_retrieval.py
│
└── 📂 media/
    ├── demo_video.mp4            # 10-minute walkthrough
    ├── diagrams/                 # Architecture diagrams
    └── screenshots/              # Visual aids
```

---

## 🎓 Learning Path

Choose your own adventure! Three paths available:

### 🏃 Fast Track (2-3 hours)
Perfect for: Getting started quickly

1. Watch the [demo video](#) (10 min)
2. Run `python src/rag_system.py` (10 min)
3. Complete `rag_starter.py` template (1-2 hours)
4. ✅ You have a working RAG system!

### 🚶 Standard Path (6-8 hours)
Perfect for: Comprehensive understanding

1. Read [tutorial.md](docs/tutorial.md) (2 hours)
2. Watch demo video (10 min)
3. Work through notebooks 01-05 (2 hours)
4. Complete exercises 1-3 (2-3 hours)
5. ✅ You understand RAG deeply!

### 🧗 Deep Dive (15-20 hours)
Perfect for: Mastery and portfolio projects

1. Complete Standard Path (8 hours)
2. Finish all exercises including advanced (4 hours)
3. Build custom RAG application (6-8 hours)
4. Deploy to production (optional, 4+ hours)
5. ✅ You're a RAG expert!

---

## ✨ Features

### 🎯 Core Implementation

- **Document Processing**: Smart chunking with overlap
- **Embeddings**: Support for multiple embedding models
- **Vector Storage**: ChromaDB integration with metadata
- **Semantic Search**: Cosine similarity with top-k retrieval
- **Answer Generation**: LLM integration ready
- **Evaluation**: Precision, recall, MRR metrics

### 🚀 Advanced Features

- **Hybrid Search**: Combine semantic + keyword search
- **Reranking**: Cross-encoder for improved results
- **Query Expansion**: Generate query variations
- **Metadata Filtering**: Filter by source, date, etc.
- **Caching**: LRU cache for repeated queries
- **Visualization**: t-SNE plots of embedding space

### 📊 Interactive Demo

Live web demo with:
- Real-time query interface
- Retrieval visualization
- Similarity score display
- Source attribution
- Comparative analysis tools

---

## 💻 Examples

### Example 1: Basic RAG Query

```python
from src.rag_system import RAGSystem

# Setup
rag = RAGSystem()
rag.ingest_documents([
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "France is in Europe."
])

# Query
result = rag.query("What is the capital of France?")

print(result['answer'])
# Output: Based on the retrieved context, Paris is the capital of France.

print(result['sources'])
# Output: [{'source': 'doc_0', 'similarity': 0.95}]
```

### Example 2: Custom Configuration

```python
# Advanced configuration
rag = RAGSystem(
    chunk_size=300,
    chunk_overlap=50,
    top_k=5
)

# With metadata filtering
result = rag.query(
    "Tell me about embeddings",
    filter={"source": "embeddings_guide.pdf"}
)
```

### Example 3: Evaluation

```python
from src.rag_system import RAGEvaluator

evaluator = RAGEvaluator()

# Test queries with ground truth
test_cases = [
    {
        "query": "What is RAG?",
        "relevant_docs": ["doc_0", "doc_1"]
    }
]

# Calculate metrics
metrics = evaluator.evaluate(rag, test_cases)
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall: {metrics['recall']:.2f}")
print(f"MRR: {metrics['mrr']:.2f}")
```

---

## 📚 Documentation

### Core Documentation

- **[Tutorial](docs/tutorial.md)**: Comprehensive 30-page guide
- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Pedagogical Report](docs/pedagogical_report.md)**: Teaching methodology
- **[Video Script](docs/video_script.md)**: Show-and-tell presentation guide

### Additional Resources

- **Notebooks**: Interactive Jupyter notebooks in `notebooks/`
- **Exercises**: Hands-on practice with solutions in `exercises/`
- **Tests**: Unit tests demonstrating usage in `tests/`

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

- 🐛 Report bugs or issues
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🎓 Add more exercises or examples
- 🌍 Translate materials

### Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/yourusername/rag-teaching-project.git

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/

# Submit pull request!
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**TL;DR**: You can use, modify, and distribute this project freely, just keep the copyright notice.

---

## 🙏 Acknowledgments

### Inspiration & Resources

- **Papers**: 
  - [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
  - [Dense Passage Retrieval (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)

- **Tools**:
  - [LangChain](https://langchain.com) - RAG framework inspiration
  - [Sentence Transformers](https://sbert.net) - Embedding models
  - [ChromaDB](https://trychroma.com) - Vector database

- **Course**:
  - INFO 7390: Advanced Data Science and Architecture
  - Northeastern University

### Special Thanks

- Course instructor for creating this excellent final project
- Anthropic for Claude (used for documentation assistance)
- The open-source community for amazing tools

---

## 📬 Contact & Support

### Get Help

- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/rag-teaching-project/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag-teaching-project/discussions)
- 📧 **Email**: your.email@example.com

### Stay Connected

- ⭐ Star this repo if you find it helpful!
- 👀 Watch for updates and new features
- 🔀 Fork to create your own version

---

## 🗺️ Roadmap

### Completed ✅

- [x] Core RAG implementation
- [x] Comprehensive tutorial
- [x] Video walkthrough
- [x] Interactive demo
- [x] Evaluation framework
- [x] Hands-on exercises

### Coming Soon 🚧

- [ ] Multi-language support
- [ ] Advanced reranking techniques
- [ ] Conversational RAG
- [ ] Production deployment guide
- [ ] Cloud integration examples
- [ ] Fine-tuning embedding models

### Future Ideas 💭

- [ ] GUI application
- [ ] REST API implementation
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Industry case studies
- [ ] Advanced optimization techniques

---

## 📊 Project Stats

- **Lines of Code**: ~2,000
- **Documentation Pages**: 40+
- **Video Length**: 10 minutes
- **Exercises**: 4 with solutions
- **Test Coverage**: 85%+
- **Dependencies**: 6 core libraries

---

## 🎓 Citation

If you use this project in your research or teaching, please cite:

```bibtex
@misc{rag_teaching_2025,
  author = {Your Name},
  title = {RAG Demystified: A Complete Teaching Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/rag-teaching-project}
}
```

---

## ⚡ Quick Links

- 📖 [Full Tutorial](docs/tutorial.md)
- 🎥 [Video Walkthrough](#)
- 💻 [Interactive Demo](src/interactive_demo.html)
- 📝 [Exercises](exercises/)
- 🐛 [Report Issues](issues/)

---

<div align="center">

**[⬆ Back to Top](#-rag-demystified-complete-teaching-framework)**

Made with ❤️ for INFO 7390 | MIT License | © 2025

**Happy Learning! 🚀**

</div>