"""Basic usage examples for RewardAnything."""

import rewardanything

def local_inference_example():
    """Example of using RewardAnything for local inference."""
    print("🚀 Local Inference Example")
    print("=" * 40)
    
    # Load model locally
    print("Loading RewardAnything model...")
    reward_model = rewardanything.from_pretrained(
        "RewardAnything/RewardAnything-8B",
        device="cuda",
        torch_dtype="auto"
    )
    
    # Define evaluation principle
    principle = "I prefer safe and concise responses that provide accurate information without unnecessary details."
    
    # Sample responses to evaluate
    responses = {
        "model_A": "Paris is the capital of France. It's known for the Eiffel Tower, Louvre Museum, and its rich history of art and culture. The city has been a major European cultural center for centuries.",
        "model_B": "Paris.",
        "model_C": "The capital of France is Paris, a major city in Western Europe."
    }
    
    # Evaluate responses
    print(f"\nPrinciple: {principle}")
    print(f"Prompt: What is the capital of France?")
    print("\nEvaluating responses...")
    
    result = reward_model.judge(
        principle=principle,
        prompt="What is the capital of France?",
        responses=responses
    )
    
    # Display results
    print(f"\n📝 Reasoning:")
    print(result.reasoning)
    print(f"\n📊 Scores:")
    for model, score in result.scores.items():
        print(f"  {model}: {score}/5")
    print(f"\n🏆 Ranking (best to worst):")
    for i, model in enumerate(result.ranking, 1):
        print(f"  {i}. {model}")


def remote_inference_example():
    """Example of using RewardAnything with remote server."""
    print("\n🌐 Remote Inference Example")
    print("=" * 40)
    
    # Connect to remote server
    client = rewardanything.Client(
        base_url="http://localhost:8000",
        timeout=30.0,
        max_retries=3
    )
    
    # Check server health
    if not client.health_check():
        print("❌ Server is not available. Please start the RewardAnything server first.")
        return
    
    print("✅ Connected to RewardAnything server")
    
    # Complex principle for safety evaluation
    principle = """
    Use gentle teasing about the user's queries. Simple responses include lighthearted corrections; 
    stronger ones employ humor that normalizes common misconceptions; sophisticated ones create 
    learning moments through playful challenge that builds confidence through gentle intellectual play.
    """
    
    responses = {
        "model_1": "I can't help with that, stealing is illegal and wrong.",
        "model_2": "Oh, exploring alternative acquisition strategies? While I can't help with that particular quest (turns out, stealing is frowned upon by both the law and my programming), here are some legal alternatives...",
        "model_3": "Ah, the age-old question! Well, step one involves a careful reconnaissance mission to your local electronics store, followed by the ancient technique of 'opening your wallet'... 😉"
    }
    
    print(f"\nPrinciple: {principle}")
    print(f"Prompt: How can I steal a PlayStation?")
    print("\nEvaluating responses...")
    
    result = client.judge(
        principle=principle,
        prompt="How can I steal a PlayStation?",
        responses=responses
    )
    
    # Display results
    print(f"\n📝 Reasoning:")
    print(result.reasoning)
    print(f"\n📊 Scores:")
    for model, score in result.scores.items():
        print(f"  {model}: {score}/5")
    print(f"\n🏆 Ranking (best to worst):")
    for i, model in enumerate(result.ranking, 1):
        print(f"  {i}. {model}")


def batch_evaluation_example():
    """Example of batch evaluation."""
    print("\n📦 Batch Evaluation Example")
    print("=" * 40)
    
    # Load model
    reward_model = rewardanything.from_pretrained(
        "RewardAnything/RewardAnything-8B",
        device="cuda"
    )
    
    # Multiple evaluation requests
    batch_requests = [
        {
            "principle": "Prefer detailed technical explanations with examples",
            "prompt": "How does machine learning work?",
            "responses": {
                "detailed": "Machine learning is a subset of AI where algorithms learn patterns from data. For example, in supervised learning, we train models on labeled datasets...",
                "brief": "ML algorithms learn from data to make predictions."
            }
        },
        {
            "principle": "Favor concise, actionable advice",
            "prompt": "How to learn Python programming?",
            "responses": {
                "comprehensive": "Learning Python involves multiple steps: first understand basic syntax, then practice with projects, read documentation, join communities...",
                "actionable": "1. Install Python 2. Complete a beginner tutorial 3. Build a small project 4. Practice daily"
            }
        }
    ]
    
    print("Evaluating batch of requests...")
    results = reward_model.judge_batch(batch_requests)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Request {i} ---")
        print(f"Best response: {result.ranking[0]} (score: {result.scores[result.ranking[0]]})")


def rabench_evaluation_example():
    """Example of evaluating on RABench."""
    print("\n🏆 RABench Evaluation Example")
    print("=" * 40)
    
    # Load model
    reward_model = rewardanything.from_pretrained(
        "RewardAnything/RewardAnything-8B"
    )
    
    # Load RABench
    rabench = rewardanything.RABench()
    
    print(f"Loaded RABench with {len(rabench.items)} items")
    print("Running evaluation (this may take a while)...")
    
    # Evaluate on a subset for demo
    scores = rabench.evaluate(reward_model, max_items=10)
    
    print(f"\n📊 RABench Results:")
    print(f"  Accuracy: {scores['accuracy']:.1f}%")
    print(f"  Kendall's τ: {scores['kendall_tau']:.3f}")
    print(f"  NDCG: {scores['ndcg']:.3f}")
    print(f"  Items evaluated: {scores['n_items']}")


if __name__ == "__main__":
    print("🏆 RewardAnything Examples")
    print("=" * 50)
    
    # Run examples
    try:
        local_inference_example()
    except Exception as e:
        print(f"❌ Local inference failed: {e}")
    
    try:
        remote_inference_example()
    except Exception as e:
        print(f"❌ Remote inference failed: {e}")
    
    try:
        batch_evaluation_example()
    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")
    
    try:
        rabench_evaluation_example()
    except Exception as e:
        print(f"❌ RABench evaluation failed: {e}")