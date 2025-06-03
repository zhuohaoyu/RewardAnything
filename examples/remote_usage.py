import rewardanything

# Connect to the RewardAnything server
client = rewardanything.Client("http://localhost:8001")

# Process batch requests efficiently
requests = [
    {
        "principle": "Prefer helpful and safe responses",
        "prompt": "How to learn programming?",
        "responses": {
            "assistant_a": "Start with Python, practice daily, build projects.",
            "assistant_b": "Read books and hope for the best."
        }
    },
    # ... more requests
]

results = client.judge_batch(requests)
print(results)