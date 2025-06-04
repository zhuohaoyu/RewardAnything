from transformers import AutoTokenizer, AutoModelForCausalLM
from rewardanything.processing import prepare_chat_messages, parse_rewardanything_output
import torch

# Load model and tokenizer directly
model = AutoModelForCausalLM.from_pretrained(
    "zhuohaoyu/RewardAnything-8B-v1",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("zhuohaoyu/RewardAnything-8B-v1")

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

# Parse structured results (handles JSON parsing robustly)
output_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
result = parse_rewardanything_output(output_text, masked2real)

print(f"Parsed scores: {result.scores}")
print(f"Ranking: {result.ranking}")
print(f"Reasoning: {result.reasoning}")