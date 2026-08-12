# LangChain SDK Examples

This repository contains small, hands-on examples for learning the core
building blocks of LangChain and LangGraph agents with OpenAI models.

The examples cover chat models, message formats, tools, structured output, and
short-term memory.

## Project Structure

```text
.
|-- core_components/
|   |-- agents.ipynb    # Agents, structured output, and memory examples
|   |-- models.ipynb    # Chat model calls, streaming, batching, and parsing
|   |-- messages.py     # LangChain message types and multimodal messages
|   |-- stm.py          # Short-term memory with LangGraph checkpointers
|   `-- tools.py        # Tool schemas, runtime context, and tool-backed agents
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.11 or later
- An OpenAI API key
- JupyterLab or Jupyter Notebook for the `.ipynb` examples

## Setup

1. Clone the repository and enter it:

   ```bash
   git clone <repository-url>
   cd langchain-sdk
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt jupyterlab
   ```

4. Create a local environment file:

   ```bash
   mkdir env
   ```

   Create `env/.env` and add:

   ```env
   OPENAI_API_KEY=your-api-key
   ```

   Keep this file private. Do not commit API keys to version control.

## Run the Examples

### Notebooks

The notebooks load `../env/.env`, so start Jupyter from the
`core_components` directory:

```bash
cd core_components
jupyter lab
```

Then open:

- `agents.ipynb`
- `models.ipynb`

Run the cells in order.

### Python Scripts

The Python scripts load `./env/.env`, so run them from the repository root:

```bash
python core_components/messages.py
python core_components/tools.py
python core_components/stm.py
```

## What You Will Learn

- Initialize chat models with `langchain.chat_models.init_chat_model`
- Use `SystemMessage`, `HumanMessage`, `AIMessage`, and `ToolMessage`
- Send plain text, structured message lists, and multimodal content
- Define Pydantic schemas for structured output
- Build agents with `langchain.agents.create_agent`
- Bind tools and inspect model-generated tool calls
- Pass runtime context into tools
- Store and retrieve long-term memory with `InMemoryStore`
- Add short-term memory with LangGraph checkpointers
- Trim, delete, and summarize message history with middleware

## Dependencies

- `langchain` for agent and model abstractions
- `langchain-openai` for OpenAI model integration
- `langchain-community` for community integrations
- `langgraph` for agent state, checkpointing, and memory workflows
- `langgraph-checkpoint-postgres` and `psycopg-binary` for Postgres-backed
  checkpoint experiments
- `pylate` for late-interaction embedding experiments
- `python-dotenv` for loading local environment variables

## Troubleshooting

- **API key is missing:** confirm that `env/.env` exists and contains
  `OPENAI_API_KEY`.
- **Notebook cannot find the API key:** start Jupyter from `core_components`.
- **Script cannot find the API key:** run scripts from the repository root.
- **Model access error:** replace the model name in the example with a model
  available to your OpenAI account.
- **Import error:** activate the same virtual environment where dependencies
  were installed, then restart the Python process or notebook kernel.

## Notes

The examples are designed for learning and experimentation. Some snippets use
in-memory stores or mock tools, so they are not production-ready without adding
persistent storage, error handling, and real external integrations.

OpenAI API usage may incur charges according to your account and model access.

## License

No license has been added yet. Unless one is provided, the repository source
code remains under the copyright holder's default rights.
