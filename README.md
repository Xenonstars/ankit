# Personal Work Agent 🤖

An intelligent AI agent that analyzes data, learns from your unstructured work information, and provides assistance based on learned knowledge. This agent serves as your personal work assistant, helping you organize, understand, and retrieve information from your work data.

## Features ✨

- **🧠 Intelligent Data Analysis**: Automatically analyzes various data types (text, CSV, JSON, files)
- **📚 Learning Capability**: Trains on unstructured work data and builds a knowledge base
- **💬 Natural Language Queries**: Ask questions in natural language and get intelligent responses
- **🔍 Semantic Search**: Find relevant information using semantic similarity
- **📊 Performance Tracking**: Monitor agent performance and confidence metrics
- **🌐 REST API**: Complete API for integration with other tools
- **📱 Web Interface**: User-friendly web interface for easy interaction
- **💾 Persistent Memory**: Remembers interactions and continuously improves
- **🔄 Batch Processing**: Handle multiple files and large datasets
- **📈 Analytics Dashboard**: Track usage patterns and performance metrics

## Quick Start 🚀

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd personal-work-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file based on the example:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
AGENT_NAME=MyWorkAssistant
```

### 3. Run the Agent

```bash
# Start the server
python main.py
```

The agent will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Usage Examples 📖

### Basic Python Usage

```python
import asyncio
from agent import PersonalWorkAgent, Config

async def main():
    # Initialize the agent
    config = Config()
    agent = PersonalWorkAgent(config)
    
    # Train with work data
    work_data = """
    Meeting Notes - Project Alpha
    - Discussed Q1 milestones
    - John to complete API design by Friday
    - Sarah working on frontend mockups
    """
    
    await agent.train_on_data(work_data, {"type": "meeting"})
    
    # Query the agent
    response = await agent.query("What is John working on?")
    print(f"Answer: {response.response}")
    print(f"Confidence: {response.confidence}")

asyncio.run(main())
```

### API Usage

```python
import requests

# Train the agent
requests.post("http://localhost:8000/train", json={
    "content": "Project deadline: March 30th. Team: Alice, Bob, Charlie.",
    "metadata": {"project": "Alpha", "type": "deadline"}
})

# Query the agent
response = requests.post("http://localhost:8000/query", json={
    "question": "When is the project deadline?"
})

print(response.json()["response"])
```

### File Upload Training

```python
import requests

# Upload and train from a file
with open("meeting_notes.txt", "rb") as f:
    response = requests.post(
        "http://localhost:8000/train/file",
        files={"file": f},
        data={"metadata": '{"type": "meeting", "date": "2024-03-15"}'}
    )

print(response.json())
```

## API Endpoints 🔌

### Core Operations

- **POST /query** - Ask questions to the agent
- **POST /train** - Train with text content
- **POST /train/file** - Train with file uploads
- **POST /train/batch** - Batch train with multiple files
- **POST /analyze** - Analyze data and get insights

### Memory & Analytics

- **GET /stats** - Get agent statistics
- **GET /memory/recent** - Get recent interactions
- **GET /memory/search** - Search interaction history
- **GET /memory/metrics** - Get performance metrics
- **POST /feedback** - Add feedback for interactions

### Management

- **GET /health** - Health check
- **GET /config** - Get configuration
- **POST /agent/reset** - Reset knowledge base

## Data Types Supported 📊

The agent can analyze and learn from various data types:

- **Text**: Meeting notes, emails, documents, reports
- **CSV/TSV**: Spreadsheets, data tables, metrics
- **JSON**: Structured data, API responses, configurations
- **Files**: .txt, .md, .csv, .json, .py, .js, .html
- **Numeric**: Numbers, arrays, statistics
- **Mixed**: Any combination of the above

## Architecture 🏗️

```
personal-work-agent/
├── agent/
│   ├── core/           # Main agent logic
│   ├── data/           # Data analysis modules
│   ├── training/       # Learning and training
│   ├── api/            # REST API endpoints
│   ├── utils/          # Utilities and configuration
│   └── web/            # Web interface (optional)
├── examples/           # Usage examples
├── data/               # Data storage
├── tests/              # Test suites
└── main.py             # Entry point
```

## Configuration ⚙️

Key configuration options in `.env`:

```env
# Required
OPENAI_API_KEY=your_key_here

# Agent Settings
AGENT_NAME=WorkAssistant
MAX_MEMORY_SIZE=1000
EMBEDDING_MODEL=all-MiniLM-L6-v2

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Storage
DATABASE_URL=sqlite:///./agent_data.db
VECTOR_DB_PATH=./data/chroma_db
DATA_DIRECTORY=./data

# Chunking
MAX_CHUNK_TOKENS=500
CHUNK_OVERLAP=50
```

## Advanced Features 🔧

### Custom Data Processing

```python
# Process structured data
project_data = {
    "metrics": {"velocity": 3.2, "quality": 0.89},
    "team": ["Alice", "Bob", "Charlie"]
}

analysis = await agent.analyze_data(project_data, "json")
print(analysis["insights"])
```

### Batch Training

```python
# Train from multiple files
files = ["meeting1.txt", "meeting2.txt", "tasks.csv"]
for file_path in files:
    await agent.trainer.process_file_content(
        file_path, 
        {"source": "weekly_sync"}
    )
```

### Memory Management

```python
# Get interaction history
recent = await agent.memory_manager.get_recent_interactions(10)

# Search past conversations
results = await agent.memory_manager.search_interactions("project alpha")

# Add feedback
await agent.memory_manager.add_feedback(interaction_id, 5, "Very helpful!")
```

## Development 👨‍💻

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run with auto-reload
python main.py
```

### Running Examples

```bash
# Basic usage example
python examples/basic_usage.py

# API client example (requires running server)
python examples/api_client.py
```

## Deployment 🚀

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Production Considerations

- Set `DEBUG=False` in production
- Use a proper database (PostgreSQL) instead of SQLite
- Configure proper CORS settings
- Set up proper logging and monitoring
- Use a reverse proxy (nginx) for serving
- Consider using Redis for caching

## Troubleshooting 🔧

### Common Issues

1. **OpenAI API Key Missing**
   ```
   Error: OpenAI API key is required but not provided
   Solution: Add OPENAI_API_KEY to your .env file
   ```

2. **Module Import Errors**
   ```
   Solution: Install dependencies with pip install -r requirements.txt
   ```

3. **Permission Errors**
   ```
   Solution: Ensure the agent has write permissions to the data directory
   ```

4. **Memory Issues**
   ```
   Solution: Adjust MAX_MEMORY_SIZE and MAX_CHUNK_TOKENS in configuration
   ```

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Support 💬

- Check the [Issues](../../issues) for common problems
- Review the [API Documentation](http://localhost:8000/docs) when running
- See the `examples/` directory for usage patterns

---

**Built with ❤️ for productivity and knowledge management**