# LangChain SDK Examples

A small collection of hands-on examples for learning how to build agents with
LangChain. The current example creates an OpenAI-backed agent and uses a
Pydantic model to return validated, structured output.

## What is included

```text
.
├── core_components/
│   └── agents.ipynb     # Structured-output agent example
├── requirements.txt
└── README.md
```

The `agents.ipynb` notebook demonstrates how to:

- Load environment variables with `python-dotenv`
- Create an agent with `langchain.agents.create_agent`
- Define a response schema with Pydantic
- Invoke the agent and access its validated `structured_response`

## Requirements

- Python 3.11 or later
- An OpenAI API key
- JupyterLab or Jupyter Notebook

## Setup

1. Clone the repository and enter it:

   ```bash
   git clone <repository-url>
   cd langchain-sdk
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the project dependencies and JupyterLab:

   ```bash
   python -m pip install -r requirements.txt jupyterlab
   ```

4. Create the environment file expected by the notebook:

   ```bash
   mkdir -p env
   printf 'OPENAI_API_KEY=your-api-key\n' > env/.env
   ```

   Keep this file private and do not commit API keys to version control.

## Run the example

The notebook loads `../env/.env`, so start Jupyter from the
`core_components` directory:

```bash
cd core_components
jupyter lab agents.ipynb
```

Run the cells in order. The agent is configured to return an `Answer` object
with these fields:

```python
class Answer(BaseModel):
    summary: str
    confidence: float
```

The final expression displays the validated response:

```python
result["structured_response"]
```

OpenAI API usage may incur charges according to your account and model access.

## Dependencies

- `langchain` for agent creation and orchestration
- `langchain-openai` for OpenAI model integration
- `langchain-community` for community integrations
- `pylate` for late-interaction embedding experiments
- `python-dotenv` for loading local environment variables

## Troubleshooting

- **API key is `None`:** confirm that `env/.env` exists and that Jupyter was
  started from `core_components`.
- **Model access error:** replace the model name in the notebook with an OpenAI
  model available to your account.
- **Import error:** activate the same virtual environment in which the
  requirements were installed, then restart the notebook kernel.

## License

No license has been added yet. Unless one is provided, the repository's source
code remains under the copyright holder's default rights.
