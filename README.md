# LLM Notepad

An AI-powered second brain that lets you store and query memories using natural language.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Store a memory:
```bash
python main.py store "I learned that ChromaDB is a great vector database for embeddings"
```

### Query your memories:
```bash
python main.py query "What did I learn about databases?"
```

### Interactive mode:
```bash
python main.py interactive
```

### Raw memory search:
```bash
python main.py query "databases" --raw
```

## Features

- Automatic text chunking for large memories
- Vector similarity search using ChromaDB
- Natural language querying with OpenAI GPT
- Metadata tagging support
- Interactive chat mode