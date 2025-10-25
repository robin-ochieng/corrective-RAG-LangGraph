# Corrective RAG with LangGraph

A corrective Retrieval-Augmented Generation (RAG) system built with LangGraph that implements self-correction mechanisms to improve response quality.

## Features

- **Corrective RAG**: Implements error detection and correction mechanisms
- **LangGraph Integration**: Uses LangGraph for complex workflow orchestration
- **Modular Architecture**: Clean, extensible design with separate components
- **Vector Search**: Utilizes Chroma for efficient vector storage and retrieval
- **Web Search Fallback**: Integrates Tavily for web search when local knowledge is insufficient

## Prerequisites

- Python 3.10 or higher (but less than 3.13)
- Poetry for dependency management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/robin-ochieng/corrective-RAG-LangGraph.git
cd corrective-RAG-LangGraph
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Project Structure

```
corrective-rag-langgraph/
├── src/
│   └── corrective_rag/
│       ├── __init__.py
│       ├── agents/          # LangGraph agents
│       ├── chains/          # LangChain chains
│       ├── retrievers/      # Retrieval components
│       ├── tools/           # Custom tools
│       └── utils/           # Utility functions
├── tests/                   # Test files
├── docs/                    # Documentation
├── pyproject.toml          # Poetry configuration
└── README.md               # This file
```

## Usage

```python
from corrective_rag import CorrectiveRAG

# Initialize the corrective RAG system
rag = CorrectiveRAG()

# Process a query
response = rag.query("Your question here")
print(response)
```

## Development

### Code Formatting

This project uses Black and isort for code formatting:

```bash
poetry run black .
poetry run isort .
```

### Testing

Run tests with pytest:

```bash
poetry run pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and formatting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.