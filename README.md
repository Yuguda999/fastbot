# Fastbot

Fastbot is an asynchronous chat and content processing service built with FastAPI. It provides context-aware replies by integrating various generative AI models (such as OpenAI, Gemini, and Deepseek) with a knowledgebase powered by Qdrant. Additionally, Fastbot includes web scraping capabilities to process and embed content for later retrieval.

## Features

- **Chat Interface**: Start new chat threads (`/new` endpoint) and continue conversations (`/continue` endpoint).
- **Multiple AI Providers**: Supports OpenAI ChatCompletion, Gemini, and Deepseek for generating replies.
- **Knowledgebase Integration**: Retrieves relevant context from a Qdrant-powered knowledgebase to enrich conversations.
- **Caching**: Uses Redis for caching chat history and knowledgebase search results.
- **Content Scraping**: Scrape URLs and process content, including text extraction and embedding generation.
- **Asynchronous Operations**: Fully asynchronous endpoints using FastAPI and async database sessions.

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL (for async database with asyncpg)
- Redis (for caching)
- Qdrant (for knowledgebase vector search)
- Required API keys for:
    - OpenAI
    - Gemini
    - OpenRouter (if needed)
    - Hugging Face (for embeddings)

### Installation

1. Clone the repository:

     ```
     git clone https://your.repo.url/fastbot.git
     cd fastbot
     ```

2. Create a virtual environment and install dependencies:

     ```
     python -m venv venv
     source venv/bin/activate
     pip install -r requirements.txt
     ```

3. Set up your environment variables. Create a `.env` file similar to:

     ```
     OPENAI_API_KEY=your_openai_api_key
     GEMINI_API_KEY=your_gemini_api_key
     OPENROUTER_API_KEY=your_openrouter_api_key
     MODEL_PROVIDER=gemini  # or openai, deepseek depending on your configuration
     DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/fastbot_db
     QDRANT_HOST=localhost
     QDRANT_PORT=6333
     HUGGINGFACE_TOKEN=your_huggingface_token
     ```

4. Run database migrations if applicable.

### Running the Application

Start the FastAPI server:

```
uvicorn app.main:app --reload
```

Access the API documentation at: `http://127.0.0.1:8000/docs`

## Project Structure

```
fastbot/
├── app/
│   ├── routers/
│   │   ├── chat.py         # Chat endpoints and business logic
│   │   └── scrape.py       # Web scraping endpoints
│   ├── utils/
│   │   ├── embedding.py    # Embedding and vector generation
│   │   └── qdrant_utils.py # Qdrant integration for knowledge retrieval
│   └── models/             # Database models definitions
├── .env                  # Environment configurations (not in version control)
├── .gitignore            # Ignored files and directories
└── README.md             # Project documentation
```

## Usage

### Chat Endpoints

- **Start New Chat**: POST `/new`
    - Creates a new chat thread with an initial message.
    - Automatically generates a title and retrieves supporting context from the knowledgebase.

- **Continue Chat**: POST `/continue`
    - Continues an existing chat thread using preserved conversation history and additional context retrieval.

### Scraping Endpoint

- **Scrape URLs**: POST `/scrape`
    - Launches background tasks to scrape and process content from provided URLs.

## License

Specify your license information here.

## Contributing

Contributions are welcome! Please follow standard GitHub practices for issues and pull requests.

## Acknowledgements

- FastAPI
- Asyncpg
- Redis
- Qdrant
- OpenAI, Gemini, and Deepseek for model integrations
- Hugging Face Hub
- Other dependencies and open source projects
