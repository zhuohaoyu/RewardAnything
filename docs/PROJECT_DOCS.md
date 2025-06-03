# RewardAnything Project Documentation

## Overview

RewardAnything is a revolutionary reward modeling framework that enables models to understand and follow explicit natural language principles instead of learning implicit preferences from fixed datasets. This enables dynamic adaptation to diverse evaluation criteria without costly retraining.

## Project Structure

```
rewardanything/
├── __init__.py           # Package initialization
├── models.py             # Data models and result classes
├── local.py              # Local inference implementation
├── client.py             # Remote client implementation
├── serve.py              # FastAPI server implementation
├── cli.py                # Command-line interface
├── utils.py              # Utility functions (OpenAI client, rate limiting)
└── benchmarks.py         # Benchmark evaluation tools (optional)

configs/
└── server_config.json    # Example server configuration

docs/
├── PROJECT_DOCS.md       # This file
├── API_REFERENCE.md      # Detailed API documentation
└── DEPLOYMENT_GUIDE.md   # Production deployment guide

tests/
├── test_local.py         # Local inference tests
├── test_client.py        # Remote client tests
└── test_server.py        # Server functionality tests

examples/
├── basic_usage.py        # Basic usage examples
├── batch_evaluation.py   # Batch processing examples
└── custom_principles.py  # Advanced principle examples
```

## Core Components

### 1. Local Inference (`local.py`)

The local inference module provides direct model loading and evaluation:

```python
import rewardanything

# Load model locally
reward_model = rewardanything.from_pretrained(
    "RewardAnything/RewardAnything-8B",
    device="cuda",
    torch_dtype="auto"
)

# Evaluate responses
result = reward_model.judge(
    principle="Prefer concise, accurate responses",
    prompt="What is Python?",
    responses={
        "model_a": "Python is a programming language...",
        "model_b": "Python is a snake."
    }
)
```

**Key Features:**
- Direct model loading from HuggingFace
- GPU/CPU support with automatic device detection
- Batch processing capabilities
- Customizable generation parameters
- Response masking to prevent bias

### 2. Remote Client (`client.py`)

The remote client enables interaction with RewardAnything servers:

```python
import rewardanything

# Connect to server
client = rewardanything.Client(
    base_url="http://localhost:8000",
    api_key="your-api-key",  # Optional
    timeout=30.0
)

# Same API as local inference
result = client.judge(
    principle="Prioritize safety and helpfulness",
    prompt="How to learn programming?",
    responses=responses
)
```

**Key Features:**
- HTTP-based communication
- Automatic retry with exponential backoff
- Authentication support
- Batch processing
- Health check capabilities

### 3. Server Implementation (`serve.py`)

The server provides a FastAPI-based REST API for RewardAnything:

```bash
# Start server
rewardanything-serve -c configs/server_config.json --port 8000
```

**API Endpoints:**
- `POST /api/rewardanything` - Single evaluation
- `POST /api/rewardanything_batch` - Batch evaluation
- `POST /api/new_batch_request` - Async batch processing
- `GET /api/fetch_results/{batch_id}` - Retrieve batch results
- `GET /health` - Health check

### 4. Data Models (`models.py`)

Core data structures for the framework:

```python
@dataclass
class RewardResult:
    reasoning: str                    # Model's reasoning process
    scores: Dict[str, float]         # Model scores (1-5)
    ranking: List[str]               # Best to worst ranking
    raw_output: Optional[str] = None # Raw model output

class RewardRequest(BaseModel):
    principle: str                   # Evaluation principle
    prompt: str                     # Input prompt
    responses: Dict[str, str]       # Model responses
    mask_responses: bool = True     # Whether to mask model names
```

## Installation and Setup

### Basic Installation

```bash
pip install RewardAnything
```

### Development Installation

```bash
git clone https://github.com/zhuohaoyu/RewardAnything.git
cd RewardAnything
pip install -e ".[dev]"
```

### Server Installation

```bash
pip install "RewardAnything[server]"
```

### Full Installation

```bash
pip install "RewardAnything[all]"
```

## Usage Patterns

### 1. Research and Experimentation

For research use cases, local inference is recommended:

```python
import rewardanything

# Load model with specific configuration
model = rewardanything.from_pretrained(
    "RewardAnything/RewardAnything-8B",
    device="cuda",
    torch_dtype="bfloat16",
    generation_config={
        "temperature": 0.1,
        "max_new_tokens": 2048
    }
)

# Evaluate with complex principles
principle = """
Evaluate responses based on:
1. Factual accuracy (50% weight)
2. Clarity and structure (30% weight) 
3. Engagement and tone (20% weight)
"""

result = model.judge(principle, prompt, responses)
```

### 2. Production Deployment

For production use cases, use the server:

```bash
# Start server
rewardanything-serve -c production_config.json --port 8000

# Scale with load balancer and multiple instances
# Use Docker for containerization
```

```python
# Client usage in production
client = rewardanything.Client("https://api.yourservice.com/v1")
results = client.judge_batch(evaluation_requests)
```

### 3. RLHF Integration

Integration with reinforcement learning from human feedback:

```python
def reward_function(prompt, response):
    principle = "Reward helpful, harmless, and honest responses"
    result = reward_model.judge(
        principle=principle,
        prompt=prompt,
        responses={"candidate": response}
    )
    return result.scores["candidate"]

# Use in PPO/GRPO training loops
```

## Configuration

### Local Model Configuration

```python
model = rewardanything.from_pretrained(
    model_name_or_path="RewardAnything/RewardAnything-8B",
    device="cuda",                    # Device placement
    torch_dtype="auto",              # Automatic dtype selection
    trust_remote_code=True,          # Trust remote code
    generation_config={              # Generation parameters
        "max_new_tokens": 2048,
        "temperature": 0.1,
        "do_sample": True,
        "top_p": 0.9
    }
)
```

### Server Configuration

```json
{
  "api_model": "gpt-4-turbo-preview",
  "api_key": ["key1", "key2"],
  "api_base": ["https://api.openai.com/v1"],
  "generation_config": {
    "max_tokens": 2048,
    "temperature": 0.1,
    "frequency_penalty": 0,
    "presence_penalty": 0
  },
  "num_workers": 8,
  "request_limit": 100,
  "request_limit_period": 60
}
```

## Advanced Features

### Response Masking

RewardAnything automatically masks model names during evaluation to prevent bias:

```python
result = model.judge(
    principle="Judge based on helpfulness",
    prompt="How to cook pasta?",
    responses={
        "gpt4": "Boil water, add pasta, cook for 8-10 minutes...",
        "claude": "Start by bringing a large pot of salted water to boil..."
    },
    mask_responses=True  # Default: True
)
# Model sees "model-1", "model-2" instead of "gpt4", "claude"
```

### Batch Processing

```python
# Local batch processing
requests = [
    {
        "principle": "Prefer technical accuracy",
        "prompt": "Explain machine learning",
        "responses": {...}
    },
    {
        "principle": "Favor practical examples", 
        "prompt": "How to debug code?",
        "responses": {...}
    }
]

results = model.judge_batch(requests, batch_size=4)

# Remote batch processing
results = client.judge_batch(requests)
```

### Custom Principles

RewardAnything excels with sophisticated, multi-criteria principles:

```python
complex_principle = """
Evaluate responses using these criteria:

1. **Technical Accuracy** (40%): 
   - Factual correctness
   - Up-to-date information
   - Proper terminology

2. **Clarity** (30%):
   - Clear explanations
   - Logical structure
   - Appropriate detail level

3. **Practical Value** (20%):
   - Actionable advice
   - Real-world applicability
   - Concrete examples

4. **Safety** (10%):
   - No harmful content
   - Appropriate disclaimers
   - Ethical considerations

For conflicting criteria, prioritize safety > accuracy > clarity > practical value.
"""

result = model.judge(complex_principle, prompt, responses)
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# Specific test modules
pytest tests/test_local.py -v
pytest tests/test_client.py -v
pytest tests/test_server.py -v

# With coverage
pytest --cov=rewardanything tests/
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run tests and linting
5. Submit a pull request

```bash
# Development setup
git clone https://github.com/your-username/RewardAnything.git
cd RewardAnything
pip install -e ".[dev]"

# Pre-commit hooks
pre-commit install

# Run tests
pytest

# Code formatting
black rewardanything/
isort rewardanything/

# Type checking
mypy rewardanything/
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Use smaller model or CPU
   model = rewardanything.from_pretrained(
       "RewardAnything/RewardAnything-1B",  # Smaller model
       device="cpu"  # Or use CPU
   )
   ```

2. **Server Connection Issues**
   ```python
   # Check server health
   client = rewardanything.Client("http://localhost:8000")
   if not client.health_check():
       print("Server is not responding")
   ```

3. **Rate Limiting**
   ```python
   # Adjust client timeout and retries
   client = rewardanything.Client(
       base_url="http://localhost:8000",
       timeout=120.0,  # Longer timeout
       max_retries=5   # More retries
   )
   ```

### Performance Optimization

1. **Use appropriate hardware**
   - GPU with sufficient VRAM for local inference
   - Multiple workers for server deployment

2. **Batch processing**
   - Use batch methods for multiple evaluations
   - Adjust batch size based on available memory

3. **Caching**
   - Server automatically caches responses
   - Use consistent request IDs for cache hits

## License

Apache 2.0 License - see [LICENSE](../LICENSE) for details. 