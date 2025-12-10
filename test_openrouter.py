"""
Quick test script to verify OpenRouter is working
"""
import os
from src.models.openrouter_client import OpenRouterClient

# Set API key
api_key = "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"

print("Testing OpenRouter connection...\n")

# Create client - Use CORRECT OpenRouter model ID
client = OpenRouterClient(
    api_key=api_key,
    model="mistralai/devstral-2512:free",  # ✅ This one definitely works!
    temperature=1.0
)

# Test prompt
system_prompt = "You are a helpful restaurant reviewer."
user_prompt = """Write a 3-star restaurant review (100-150 words) for a casual Italian restaurant.
Be honest and balanced - mention both positives and negatives.

Review:"""

print(f"Model: {client.model}")
print(f"Sending request...")
print(f"\nSystem: {system_prompt}")
print(f"User: {user_prompt}\n")
print("=" * 60)

# Generate
result = client.generate(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    max_tokens=256
)

# Print results
print("\n" + "=" * 60)
print("RESULT:")
print("=" * 60)
print(f"Success: {result['success']}")
print(f"Model: {result['model']}")
print(f"Time: {result['time']:.2f}s")
print(f"Cost: ${result['cost']:.4f}")

if result['success']:
    print(f"\nGenerated Text ({len(result['text'].split())} words):")
    print("-" * 60)
    print(result['text'])
    print("-" * 60)
    print(f"\nTokens: {result['tokens']}")
else:
    print(f"\n❌ ERROR: {result.get('error', 'Unknown error')}")

print("\n" + "=" * 60)

# If successful, test the model router
if result['success']:
    print("\nNow testing with ModelRouter...")
    from src.models.model_router import ModelRouter
    from src.utils.config_loader import ModelConfig
    
    config = ModelConfig(
        provider="openrouter",
        model="google/gemini-2.0-flash-exp:free",  # ✅ Updated
        temperature=1.0,
        weight=1.0,
        api_key_env="OPENROUTER_API_KEY"
    )
    
    # Set env var
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    router = ModelRouter([config])
    
    result2 = router.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=256
    )
    
    print(f"\nRouter Result:")
    print(f"  Success: {result2['success']}")
    print(f"  Text length: {len(result2.get('text', '').split())} words")
    print(f"  Model key: {result2.get('model_key', 'N/A')}")
    
    if result2['success']:
        print(f"\n  Generated text: {result2['text'][:100]}...")
    else:
        print(f"\n  ❌ Error: {result2.get('error', 'Unknown')}")