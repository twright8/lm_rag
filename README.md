# Anti-Corruption RAG System

A modular Retrieval-Augmented Generation (RAG) system specifically designed for anti-corruption document analysis. This system enables users to upload documents, process them to extract entities and relationships, and query the information using natural language.

## Features

- Upload documents (PDF, Word, TXT, CSV, XLSX)
- Sequential document processing with granular progress updates
- LLM-based entity and relationship extraction
- Hybrid search (BM25 + Vector) with optional reranking
- Local BM25 indices and Dockerized Qdrant vector storage
- Interactive network graph visualization
- Conversational query interface

## System Architecture

The system is composed of the following modules:

1. **Document Processing**:
   - Document loading with OCR support
   - Semantic chunking
   - LLM-based entity and relationship extraction
   - Indexing to BM25 and Qdrant

2. **Query System**:
   - Hybrid search combining BM25 and vector search
   - Optional reranking of results
   - Conversational LLM for generating answers

3. **UI**:
   - Document upload and progress monitoring
   - Data exploration (documents, entities, relationship graph)
   - Conversational interface for queries

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Tesseract OCR engine
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/anti-corruption-rag.git
   cd anti-corruption-rag
   ```

2. Start the Qdrant service:
   ```bash
   docker-compose up -d qdrant
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up NLTK resources:
   ```bash
   python setup_nltk.py
   ```

6. Ensure Tesseract is installed locally and in your PATH.

## Running the Application

Start the Streamlit application:

```bash
streamlit run src/ui/app.py --server.runOnSave true
```

The application will be available at `http://localhost:8501`.

## Configuration

The system can be configured through the `config.yaml` file. Key configuration options include:

- Document processing parameters (chunking)
- Model specifications
- Aphrodite (LLM) settings
- Qdrant connection settings
- Retrieval parameters
- UI customization

## Troubleshooting

### Common Issues and Solutions

1. **NLTK Resource Errors**
   - If you encounter errors related to missing NLTK resources like `punkt_tab`, run the setup script:
     ```bash
     python setup_nltk.py
     ```
   - For manual installation of specific resources:
     ```bash
     python -m nltk.downloader punkt punkt_tab stopwords
     ```

2. **Qdrant Connection Errors**
   - Ensure Qdrant is running:
     ```bash
     docker ps | grep qdrant
     ```
   - If not running, restart the container:
     ```bash
     docker-compose up -d qdrant
     ```
   - Check if the host and port in `config.yaml` match your Docker setup.

3. **Memory Errors**
   - If you encounter CUDA out-of-memory errors, try:
     - Reducing model size in `config.yaml` (use smaller models)
     - Increasing quantization (e.g., from `fp8` to `fp6`)
     - Processing fewer documents at a time
     - Running on a machine with more GPU memory

4. **Processing Hangs or Freezes**
   - Check logs in the `logs` directory for detailed error information
   - Reset the application by clicking "Clear All Data" in the sidebar
   - Restart the Streamlit application

## License

This project is licensed under the MIT License - see the LICENSE file for details.
