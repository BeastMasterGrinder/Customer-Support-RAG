# CloudSync RAG System

A Retrieval-Augmented Generation (RAG) system for CloudSync's customer support team, using LangChain and Google Gemini.

## Project Structure

```
.
├── data/
│   ├── product_docs.json
│   ├── support_tickets.json
│   └── test_queries.json
├── src/
│   ├── config/
│   │   └── config.py
│   ├── data_processing/
│   │   ├── document_loader.py
│   │   └── intelligent_splitter.py
│   ├── embeddings/
│   │   └── embedding_manager.py
│   ├── utils/
│   │   └── text_processing.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Features

- Intelligent document chunking that preserves:
  - Numbered lists
  - Step-by-step instructions
  - Related content sections
- Metadata preservation for both product docs and support tickets
- ChromaDB vector store integration
- Google Gemini embeddings

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
GOOGLE_API_KEY=your_api_key_here
```

if this does not work try to run 

```bash
export GOOGLE_API_KEY='your-actual-api-key'
```

## Running the System

To process documents and create the vector store:

```bash
python src/main.py
```

This will:
1. Load documents from JSON files
2. Apply intelligent chunking
3. Generate embeddings
4. Store in ChromaDB

## Directory Structure Explanation

- `config/`: Configuration settings and constants
- `data_processing/`: Document loading and chunking logic
- `embeddings/`: Vector store and embedding management
- `utils/`: Helper functions for text processing
- `main.py`: Main execution script

## Data Sources

- `product_docs.json`: Official product documentation
- `support_tickets.json`: Historical support tickets
- `test_queries.json`: Test queries with evaluation criteria 

Test Queries with their scores
![image](https://github.com/user-attachments/assets/9e9f91af-ca66-4fde-9adf-d5de46351e96)

![image](https://github.com/user-attachments/assets/b46fd564-1710-4cb4-9d98-524d77254e21)


As we can see that the relevant ticket is being found and a scroe is given to it according to thresholds we set in `retrieval_config.py`


Step 3:
Proper citations with Document titles and IDS
Specific sections when applicable and confidence indicators + handling edge cases

![image](https://github.com/user-attachments/assets/98949119-79b1-45d9-8e32-cc43aad3dd84)

![image](https://github.com/user-attachments/assets/a6b93986-cb12-4b3e-970a-02f1b0d49557)

![image](https://github.com/user-attachments/assets/0ad7151c-1ecd-4536-bd9d-b638e3e613d8)


Edge cases
![image](https://github.com/user-attachments/assets/6e6ea44a-531b-4280-9daa-c46f1a9d4c95)

# Chanllenge A. Version Aware

![image](https://github.com/user-attachments/assets/96f08d8b-262e-4ff2-98c3-c8454d1cb50d)

![image](https://github.com/user-attachments/assets/8d6f94a2-be4d-4c40-962f-5b6c31928403)

