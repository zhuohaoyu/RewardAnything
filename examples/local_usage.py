import rewardanything

# Load model locally (similar to HuggingFace)
reward_model = rewardanything.from_pretrained(
    "zhuohaoyu/RewardAnything-8B-v1",  # Model path/name
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