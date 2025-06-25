# Snowflake MCP OpenAI Integration

A powerful integration between Snowflake's MCP (Model Context Protocol) server and OpenAI's SDK, enabling intelligent data analysis and reasoning capabilities for Snowflake environments.

## 🚀 Features

- **Seamless Integration**: Connect OpenAI agents directly to Snowflake via MCP
- **Intelligent Reasoning**: Advanced reasoning capabilities with iterative query refinement
- **Cortex AI Integration**: Leverage Snowflake's Cortex AI services for data analysis
- **Configurable Agents**: Both basic and advanced reasoning agents available
- **Environment-based Configuration**: Secure credential management
- **Comprehensive Tooling**: Support for Cortex Search, Analyst, and Complete services

## 📋 Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Snowflake account with appropriate permissions
- Snowflake Personal Access Token (PAT)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/snowflake-mcp-openai.git
   cd snowflake-mcp-openai
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here

# Snowflake Configuration
SNOWFLAKE_ACCOUNT_IDENTIFIER=your-snowflake-account-identifier
SNOWFLAKE_USERNAME=your-snowflake-username
SNOWFLAKE_PAT=your-snowflake-personal-access-token
```

### Snowflake Tools Configuration

Update `src/config/tools_config.yaml` to match your Snowflake environment:

```yaml
cortex_complete:
  default_model: "snowflake-llama-3.3-70b"
  description: "LLM service that can answer general questions"

search_services:
  - service_name: "your_search_service"
    description: "Search service that indexes your documents"
    database_name: "YOUR_DATABASE"
    schema_name: "YOUR_SCHEMA"

analyst_services:
  - service_name: "your_analyst_service"
    semantic_model: "@YOUR_DATABASE.YOUR_SCHEMA.SEMANTIC_MODELS/your_model.yaml"
    description: "Analyst service for your data"
```

## 🎯 Usage

### Basic Agent

Run the basic agent for simple queries:

```bash
python src/agents/basic_agent.py
```

### Advanced Reasoning Agent

Run the advanced agent with intelligent reasoning capabilities:

```bash
python src/agents/reasoning_agent.py
```

### Programmatic Usage

```python
import asyncio
from src.agents.basic_agent import create_basic_agent
from src.agents.reasoning_agent import SmartReasoningAgent

async def main():
    # Create basic agent
    agent = await create_basic_agent()
    
    # Run a query
    result = await agent.run("What are our top performing products?")
    print(result)

# Or use the reasoning agent
async def advanced_example():
    base_agent = await create_basic_agent()
    reasoning_agent = SmartReasoningAgent(base_agent)
    
    result = await reasoning_agent.reason_and_execute(
        "Why are sales declining in Q4?"
    )
    print(result['final_result'])
```

## 🏗️ Project Structure

```
snowflake-mcp-openai/
├── src/
│   ├── agents/
│   │   ├── basic_agent.py          # Simple agent implementation
│   │   └── reasoning_agent.py      # Advanced reasoning agent
│   ├── config/
│   │   ├── mcp_config.json         # MCP server configuration
│   │   └── tools_config.yaml       # Snowflake tools configuration
│   └── examples/
│       ├── basic_queries.py        # Basic usage examples
│       └── advanced_queries.py     # Advanced reasoning examples
├── tests/
│   ├── test_basic_agent.py
│   └── test_reasoning_agent.py
├── docs/
│   └── api.md                      # API documentation
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🔧 Available Tools

The integration provides access to several Snowflake Cortex AI tools:

### Cortex Search
- **Purpose**: Search through indexed documents and contracts
- **Usage**: Full-text search across your document collections

### Cortex Analyst
- **Purpose**: Intelligent data analysis and query interpretation
- **Usage**: Natural language queries against your data models

### Cortex Complete
- **Purpose**: LLM-powered text completion and generation
- **Usage**: Generate insights and summaries from your data

## 🧠 Reasoning Agent Features

The advanced reasoning agent provides:

- **Iterative Query Refinement**: Automatically refines queries based on results
- **Quality Assessment**: Evaluates response quality and confidence
- **Adaptive Stopping**: Stops when sufficient information is gathered
- **Caching**: Avoids redundant queries
- **Cortex Integration**: Leverages Cortex Analyst for query interpretation

## 📊 Example Queries

```python
# Sales analysis
"Why are sales in Germany outperforming other regions?"

# Customer insights
"What are the key factors driving customer churn?"

# Financial analysis
"Show me the top revenue drivers for Q4"

# Document search
"Find all contracts mentioning pricing changes"

# Trend analysis
"What are the emerging trends in our product categories?"
```

## 🧪 Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_basic_agent.py

# Run with coverage
pytest --cov=src
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🛡️ Security

- Never commit API keys or credentials to version control
- Use environment variables for all sensitive configuration
- Regularly rotate your Snowflake PAT tokens
- Follow Snowflake's security best practices

## 📞 Support

- Create an issue for bugs or feature requests
- Check the [documentation](docs/) for detailed API information
- Review the [examples](src/examples/) for common use cases

## 🔗 Related Projects

- [Snowflake MCP Server](https://github.com/Snowflake-Labs/mcp)
- [OpenAI SDK](https://github.com/openai/openai-python)
- [Agents Framework](https://github.com/openai/openai-python)

## 📈 Roadmap

- [ ] Add support for more Snowflake Cortex services
- [ ] Implement streaming responses for large queries
- [ ] Add visualization capabilities
- [ ] Create web interface
- [ ] Add Docker support
- [ ] Implement query optimization
- [ ] Add monitoring and logging 