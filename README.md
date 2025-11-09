# MathMind

A collaborative AI-powered mathematical proof system that uses multi-agent architecture to generate, verify, and prove mathematical conjectures in Number Theory and Algebraic Topology.

## Overview

MathMind is an advanced research tool that combines semantic search, symbolic reasoning, and iterative proof verification to explore mathematical conjectures. The system employs specialized AI agents that work together to:

1. **Generate Conjectures**: Analyze mathematical queries and propose precise conjectures
2. **Design Experiments**: Create computational tests to validate conjectures
3. **Verify Proofs**: Iteratively refine and verify mathematical proofs
4. **Synthesize Results**: Produce comprehensive research reports

## Features

- **Multi-Domain Support**: Number Theory and Algebraic Topology agents
- **Iterative Verification**: Up to configurable iterations of proof refinement
- **Real-time Visualization**: Interactive Streamlit interface showing agent conversations
- **Evidence Retrieval**: Semantic and symbol-based search through mathematical papers
- **Export Capabilities**: Download results in JSON or Markdown format
- **Comprehensive Reporting**: Track theorems (proven) vs conjectures (unproven)

## Architecture

The system implements a workflow:

```
User Query → Domain Expert Agent → Conjectures →
Reduce to Lemmas → Experimenter → Verifier (iterative) → Results
```

### Agents

1. **Number Theory Agent**: Specializes in analytic number theory, modular forms, L-functions, and Ramanujan tau functions
2. **Algebraic Topology Agent**: Focuses on homology, homotopy theory, spectral sequences, and K-theory
3. **Experimenter Agent**: Designs computational experiments (SageMath, PARI/GP, Python)
4. **Verifier Agent**: Rigorously checks proofs and identifies logical gaps
5. **Coordinator Agent**: Orchestrates the workflow and synthesizes results

## Installation

### Prerequisites

- Python >= 3.10, < 3.14
- Poetry (recommended) or pip
- OpenAI API key

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd MathMind

# Install dependencies
poetry install

# Run commands with poetry (recommended)
poetry run streamlit run streamlit_app.py

# Activate the virtual environment
poetry env activate
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd MathMind

# Install dependencies
pip install -r requirements_streamlit.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OTEL_SDK_DISABLED=true
```

## Usage

### Streamlit Web Interface

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

The interface provides:
- Text input for mathematical queries
- Agent type selection (Number Theory or Algebraic Topology)
- Configurable verification iterations
- Real-time agent conversation display
- Expandable result sections for theorems and conjectures
- Export functionality (JSON/Markdown)

### Command Line

Run the proof system directly:

```bash
# Number Theory
python main.py number_theory

# Algebraic Topology
python main.py algebraic_topology
```

### Example Queries

**Number Theory:**
- "distribution of zeros of Ramanujan tau(n) modulo small primes"
- "density of primes p where tau(p) ≡ 0 (mod p)"
- "asymptotic behavior of sum of tau(n) for n up to X"

**Algebraic Topology:**
- "homology groups of product spaces"
- "homotopy groups of spheres"
- "characteristic classes of vector bundles"
- "K-theory of topological spaces"

## Project Structure

```
MathMind/
├── main.py                      # Core proof system implementation
├── streamlit_app.py            # Web interface
├── agents.py                   # Agent definitions and prompts
├── retriever.py                # Evidence retrieval functions
├── ingest.py                   # Paper indexing system
├── requirements_streamlit.txt  # Python dependencies
├── pyproject.toml             # Poetry configuration
├── .env                       # Environment variables (create this)
├── papers/                    # Mathematical papers directory
└── index/                     # FAISS indices and metadata
```

## Key Components

### ProofSystem Class

The main orchestrator that processes queries:

```python
from main import ProofSystem

# Initialize with agent type
proof_system = ProofSystem(
    max_iterations=5,
    agent_type="number_theory"
)

# Process a query
results = proof_system.process_query(
    "distribution of zeros of Ramanujan tau(n) modulo small primes"
)
```

### Retrieval Tools

Two retrieval methods for gathering evidence:

1. **Semantic Retrieve**: Uses embeddings to find semantically similar content
2. **Symbol Retrieve**: Extracts and searches for mathematical symbols

### Iterative Verification

The `IterativeProofVerifier` class implements a conversation loop:
1. Verifier checks the proof
2. Returns verdict: VALID, INVALID, or INCOMPLETE
3. Experimenter refines based on feedback
4. Repeat until proven or max iterations reached

## Output Format

Results include:

```json
{
  "query": "user query",
  "timestamp": "ISO timestamp",
  "summary": {
    "total_conjectures": 5,
    "theorems_proven": 2,
    "conjectures_remaining": 3
  },
  "theorems": [/* proven statements */],
  "remaining_conjectures": [/* unproven statements */],
  "final_output": "synthesis report"
}
```

## Configuration

Adjust system parameters in the code:

- **max_iterations**: Number of verification iterations (default: 5)
- **temperature**: LLM creativity (default: 0.7)
- **llm model**: GPT-4 variants (gpt-4o, gpt-4o-mini)
- **retrieval k**: Number of evidence chunks to retrieve

## Development

### Adding New Agent Types

1. Create agent in `MathAgents` class ([main.py](main.py))
2. Add specialized backstory and tools
3. Create corresponding experimenter agent
4. Update `ProofSystem._create_agents()` to include new type

### Customizing Prompts

Edit agent backstories and task descriptions in [main.py](main.py) to adjust agent behavior.

### Extending Retrieval

Modify `retrieve_evidence` tool in [main.py](main.py:38-91) to add new retrieval strategies.

## Technical Details

- **Framework**: CrewAI for multi-agent orchestration
- **LLM**: OpenAI GPT-4 (configurable)
- **Embeddings**: OpenAI embeddings or sentence-transformers
- **Vector Store**: FAISS for efficient similarity search
- **UI**: Streamlit with real-time updates
- **Visualization**: Plotly for interactive charts

## Limitations

- Requires OpenAI API access (can be modified for other LLMs)
- Computational experiments are simulated (not executed)
- Formal verification requires external theorem provers (Lean/Coq/Isabelle)
- Limited to mathematical domains with pre-indexed papers

## Future Enhancements

- Integration with symbolic math engines (SageMath, SymPy)
- Formal proof generation for theorem provers
- Automated execution of computational experiments
- Support for additional mathematical domains
- Real-time collaborative features

## Contributing

Contributions are welcome! Areas for improvement:

- Additional mathematical domains
- Better proof verification heuristics
- Integration with formal verification tools
- Enhanced visualization of proof structures
- Performance optimizations

## License

[Specify your license here]

## Citation

If you use MathMind in your research, please cite:

```bibtex
@software{mathind2025,
  title={MathMind: Collaborative AI Mathematical Proof System},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/MathMind}
}
```

## Acknowledgments

- Built with [CrewAI](https://github.com/joaomdmoura/crewAI)
- Powered by OpenAI GPT-4
- Uses FAISS for vector similarity search

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: [your contact information]

---

**MathMind**: Advancing mathematical research through collaborative AI agents
