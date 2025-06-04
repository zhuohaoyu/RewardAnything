# RewardAnything: Generalizable Principle-Following Reward Models

Traditional reward models learn **implicit preferences** from fixed datasets, leading to static judgments that struggle with the **nuanced and multifaceted nature of human values**.
We believe that, much like Large Language Models follow diverse instructions, reward models must be able to understand and follow **explicitly specified principles**.

**RewardAnything** embodies this new paradigm. Our models are designed to interpret natural language principles at inference time, enabling **dynamic adaptation** to a wide array of evaluation criteria **without costly retraining**. This approach shifts from fitting a single preference distribution to achieving true principle-following generalization.

## 🌟 Key Features

- 🧠 **Principle-Following**: Directly interprets and applies reward criteria specified in natural language
- 🔄 **Dynamic Adaptability**: Generalizes to new, unseen principles at inference time without retraining
- 💰 **Resource Efficient**: Eliminates costly cycles of collecting preference data and retraining RMs
- 📊 **State-of-the-Art Performance**: Achieves SOTA on RM-Bench and excels on our RABench benchmark
- 🧩 **Easy Integration**: Works seamlessly with existing RLHF pipelines (PPO, GRPO)
- 🔍 **Interpretable**: Provides transparent reasoning for evaluation decisions

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

RewardAnything offers three flexible deployment options to fit your workflow:

## 1. 🏠 Local Inference (Recommended for Quick Testing)

**Best for**: Quick experimentation, small-scale evaluation, research

**Pros**: Simple setup, no external dependencies
**Cons**: Requires local GPU, slower for batch processing

```python
import rewardanything

# Load model locally (similar to HuggingFace)
reward_model = rewardanything.from_pretrained(
    "<path-to-model-weights>",  # Model path/name
    device="cuda",                        # Device placement
    torch_dtype="auto"                   # Automatic dtype selection
)

# Define your evaluation principle
principle = "I prefer clear, concise and helpful responses over long and detailed ones."

# Your evaluation data
prompt = "How do I learn Python programming effectively?"
responses = {
    "response_a": "Start with Python.org's tutorial, practice daily with small projects, and join r/learnpython for help. Focus on fundamentals first.",
    "response_b": "Here's a comprehensive approach: 1) Start with Python basics including variables, data types, operators, control structures like if-statements, for-loops, while-loops, and functions, 2) Practice with small projects like calculators, text games, and data manipulation scripts, 3) Use interactive platforms like Codecademy, Python.org's official tutorial, edX courses, Coursera specializations, and YouTube channels, 4) Join communities like r/learnpython, Stack Overflow, Python Discord servers, and local meetups for support and networking, 5) Build progressively complex projects including web scrapers, APIs, data analysis tools, and web applications, 6) Read books like 'Automate the Boring Stuff', 'Python Crash Course', and 'Effective Python', 7) Dedicate 1-2 hours daily for consistent progress and track your learning journey.",
    "response_c": "Learn Python by coding."
}

# Get comprehensive evaluation
result = reward_model.judge(
    principle=principle,
    prompt=prompt, 
    responses=responses
)

print(f"Scores: {result.scores}")
print(f"Best to worst: {result.ranking}")
print(f"Reasoning: {result.reasoning}")
```

## 2. 🚀 vLLM Deployment (Recommended for Production & RL Training)

**Best for**: High-throughput batch inference, RLHF training, production workloads

**Pros**: Fast batch processing, optimized inference, scalable
**Cons**: Requires vLLM setup

### Step 1: Setup vLLM Server

First, install and start a vLLM server. See the [vLLM quickstart guide](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server) for detailed instructions:

```bash
# Install vLLM
pip install vllm

# Start vLLM server with RewardAnything model
vllm serve <path-to-model-weights> \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --tensor-parallel-size 1
```

### Step 2: Configure RewardAnything Server

Create a config file `config.json`:

```json
{
  "api_key": ["dummy-key-for-vllm"],
  "api_model": "<path-to-model-weights>",
  "api_base": ["http://localhost:8000/v1"],
  "api_timeout": 120.0,
  "generation_config": {
    "temperature": 0.0,
    "max_tokens": 4096
  },
  "num_workers": 8,
  "request_limit": 500,
  "request_limit_period": 60
}
```

### Step 3: Start RewardAnything Server

```bash
# Start the RewardAnything API server
rewardanything serve -c config.json --port 8001
```

### Step 4: Use in Your Code

```python
import rewardanything

# Connect to the RewardAnything server
client = rewardanything.Client("http://localhost:8001")

# Process batch requests efficiently
requests = [
    {
        "principle": "Prefer clear, concise and helpful responses over long and detailed ones.",
        "prompt": "How to learn programming?",
        "responses": {
            "assistant_a": "Start with Python, practice daily, build projects.",
            "assistant_b": "Read books and hope for the best.",
            "assistant_c": "Start with Python.org's tutorial, practice daily with small projects, and join r/learnpython for help. Focus on fundamentals first."
        }
    },
    # ... more requests
]

results = client.judge_batch(requests)
for result in results:
    print(f"Winner: {result.ranking[0]}")
```

## 3. 🔧 Direct HuggingFace Integration

**Best for**: Custom workflows, advanced users, integration with existing HF pipelines

**Pros**: Full control, custom processing
**Cons**: Manual parsing required

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from rewardanything.processing import prepare_chat_messages, parse_rewardanything_output

# Load model and tokenizer directly
model = AutoModelForCausalLM.from_pretrained(
    "<path-to-model-weights>",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("<path-to-model-weights>")

# Prepare evaluation data
principle = "Judge responses based on helpfulness and accuracy"
prompt = "What is the capital of France?"
responses = {
    "model_a": "Paris is the capital of France.",
    "model_b": "I think it might be Lyon or Paris."
}

# Prepare chat messages (handles masking automatically)
messages, masked2real = prepare_chat_messages(principle, prompt, responses)

# Format with chat template
formatted_input = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Generate response
inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode output
generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

# Parse structured results (handles JSON parsing robustly)
result = parse_rewardanything_output(output_text, masked2real)

print(f"Raw output: {output_text}")
print(f"Parsed scores: {result.scores}")
print(f"Ranking: {result.ranking}")
print(f"Reasoning: {result.reasoning}")
```

## 📊 When to Use Each Method

| Use Case | Method | Why |
|----------|--------|-----|
| Quick testing | Local Inference | Simplest setup |
| Research & development | Local Inference | Full control, easy debugging |
| RLHF training | vLLM Deployment | High throughput, optimized for batches |
| Production evaluation | vLLM Deployment | Scalable, reliable |
| Large-scale evaluation | vLLM Deployment | Best performance |
| Custom integration | Direct HuggingFace | Maximum flexibility |


## 🔬 Advanced Usage

### Custom Principles

RewardAnything excels with sophisticated, multi-criteria principles:

```python
complex_principle = """
Evaluate responses using these criteria:
1. **Technical Accuracy** (40%): Factual correctness and up-to-date information
2. **Clarity** (30%): Clear explanations and logical structure  
3. **Practical Value** (20%): Actionable advice and real-world applicability
4. **Safety** (10%): No harmful content, appropriate disclaimers

For conflicting criteria, prioritize: safety > accuracy > clarity > practical value.
"""

result = reward_model.judge(complex_principle, prompt, responses)
```

### Integration with RLHF

```python
# Example: Use in PPO training loop
def reward_function(principle, prompt, response):
    result = reward_model.judge(
        principle=principle,
        prompt=prompt,
        responses={"generated": response, "reference": "baseline response"}
    )
    return result.scores["generated"]

# Use in your RLHF training
rewards = [reward_function(principle, prompt, resp) for resp in generated_responses]
```

### Response Masking

RewardAnything automatically masks model names to prevent bias:

```python
result = reward_model.judge(
    principle="Judge based on helpfulness", 
    prompt="How to cook pasta?",
    responses={
        "gpt4": "Boil water, add pasta...",
        "claude": "Start by bringing water to boil..."
    },
    mask_responses=True  # Default: True, model sees "model-1", "model-2"
)
```
