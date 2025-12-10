# Synthetic Review Data Generator with Quality Guardrails

A production-grade synthetic data generator for service and tool reviews. This project uses multiple AI models with comprehensive quality checks, multi-agent orchestration, and an LLM-as-a-Judge evaluation system.

## What This Does

This system generates realistic, diverse product/service reviews using AI models. It's not just about generating text - it includes:

- Multi-agent framework built with LangGraph with specialized agents for different tasks
- LLM-as-a-Judge system using Gemini 2.0 Flash Exp to evaluate quality
- Support for multiple AI providers (OpenAI GPT-4, Google Gemini, and local models via Groq)
- High temperature sampling (1.0) to maximize diversity in generated content
- Random seed words using Python Faker to inject true randomness
- Comprehensive quality checks including statistical metrics, bias detection, realism validation, and LLM judge scoring
- Automatic rejection and regeneration if quality isn't good enough (up to 3 attempts)
- Detailed logging for every step of the process
- Real-time tracking of costs and performance across all models
- Quality reports in Markdown format with metrics and comparisons

## System Architecture

The system works through a multi-agent approach:

```
Orchestrator Agent manages everything
    |
    |--- Generator Agent (creates the reviews)
    |--- Quality Validators (check statistical metrics)
    |--- LLM Judge Agent (evaluates authenticity and quality)
```

Each review goes through this pipeline:

1. Generate a review using a specific persona, seed words, and high temperature settings
2. Run statistical checks on length, vocabulary, diversity, and sentiment
3. Have the LLM Judge evaluate it across 4 dimensions
4. Decide whether to accept it, regenerate it (up to 3 times), or reject it
5. Log everything for analysis

## Why I Made These Design Choices

**Using Gemini 2.0 Flash Exp as the Primary Judge**

I chose Gemini because it's fast (about 2 seconds per evaluation compared to 4+ seconds for GPT-4), costs less (free during preview vs $0.03 per 1K tokens), and provides comparable quality. It made sense for evaluating hundreds of reviews.

**Temperature Set to 1.0**

I wanted maximum diversity, so I cranked up the temperature. Real humans are unpredictable when writing reviews - some are verbose, some are terse, some are emotional, some are clinical. High temperature mimics that natural variability. The trade-off is slightly lower coherence sometimes, but the diversity gain is worth it.

**Python Faker for Seed Words**

LLMs struggle to generate truly random content. They tend to fall into patterns. By injecting 10 random seed words from Faker into each review prompt, I force the model to incorporate unexpected elements, which dramatically reduces repetition (measured by Self-BLEU scores).

**Multi-Model Strategy**

I split generation between OpenAI GPT-4 (50%) and Google Gemini (50%). This gives me diversity from different model "personalities" while optimizing costs. GPT-4 is great for creative, high-quality reviews. Gemini is fast and cost-effective for volume.

**Groq for Local Models (Due to Hardware Limitations)**

I was working with limited local computing power and memory, so running large models locally wasn't practical. Instead, I used Groq to access open source models like Llama and Mistral. Groq provides incredibly fast inference for open source models without needing powerful local hardware. This let me include open source models in the mix without infrastructure investment.

## Getting Started

**What You Need**

- Python 3.9 or higher
- OpenAI API key (optional if using only Gemini/Groq)
- Google Gemini API key (optional if using only OpenAI/Groq)
- Groq API key (optional if using only OpenAI/Gemini)

**Setup Steps**

```bash
# Clone the repository
git clone https://github.com/abdalrahmenyousifMohamed/Synthetic-Data-Generator-with-Quality-Guardrails.git
cd Synthetic-Data-Generator-with-Quality-Guardrails

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"

# Set up environment variables
cp .env.example .env
# Edit .env file and add your API keys:
# OPENAI_API_KEY=your_key_here
# GEMINI_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here

# Create directories for data and logs
mkdir -p data/real_reviews data/generated logs
```

## How to Use It

**Basic Commands**

```bash
# Generate 500 reviews with default settings
python generate.py

# Use a custom configuration file
python generate.py --config config.yaml --output reviews.jsonl

# Generate for a specific domain with custom sample count
python generate.py --domain "developer-tools" --samples 300

# Compare against real reviews
python generate.py --real-reviews data/real_reviews/reviews.jsonl
```

**Advanced Options**

```bash
# Skip the LLM judge to generate faster (uses only statistical checks)
python generate.py --skip-llm-judge

# Specify custom output locations
python generate.py \
  --output data/generated/reviews.jsonl \
  --report-output data/generated/report.md

# Run in benchmark mode for detailed model comparison
python generate.py --benchmark
```

**Available Command Line Options**

- config: Path to configuration file (default: config.yaml)
- output: Where to save generated reviews (default: data/generated/reviews.jsonl)
- domain: Override the domain from config (like "developer-tools" or "restaurants")
- samples: How many reviews to generate
- real-reviews: Path to real reviews for comparison (default: data/real_reviews/reviews.jsonl)
- skip-llm-judge: Skip LLM evaluation to speed things up
- report-output: Where to save the quality report (default: data/generated/quality_report.md)
- benchmark: Enable detailed model performance comparison

## Configuration File

Everything is controlled through config.yaml. Here are the main sections:

**Generation Settings**
```yaml
generation:
  target_samples: 500        # How many reviews to generate
  batch_size: 50             # How many to process at once
  temperature: 1.0           # Higher = more diverse (0.0-2.0)
  seed_word_count: 10        # Random words injected per review
  domain: "developer-tools"  # What domain/industry
```

**Model Configuration**
```yaml
models:
  - provider: "openai"
    model: "gpt-4"
    weight: 0.3              # 30% of samples use this model
    temperature: 1.0
    api_key_env: "OPENAI_API_KEY"
    
  - provider: "google"
    model: "gemini-2.0-flash-exp"
    weight: 0.4              # 40% of samples
    temperature: 1.0
    api_key_env: "GEMINI_API_KEY"
    
  - provider: "groq"
    model: "llama-3.1-70b-versatile"
    weight: 0.3              # 30% of samples
    temperature: 1.0
    api_key_env: "GROQ_API_KEY"
```

**LLM Judge Settings**
```yaml
llm_judge:
  primary_model: "gemini-2.0-flash-exp"  # Main evaluator
  secondary_model: "gpt-4"                # Backup if primary fails
  temperature: 0.2                        # Low for consistent evaluation
  enable_multi_judge: true                # Use consensus between judges
  parallel_execution: true                # Evaluate multiple at once
  evaluation_criteria:
    - authenticity       # Does it sound like a real person?
    - alignment          # Does sentiment match the rating?
    - expertise          # Does it show domain knowledge?
    - uniqueness         # Is it original, not generic?
```

**Quality Thresholds**
```yaml
quality_thresholds:
  # Statistical checks
  min_length: 50
  max_length: 500
  min_unique_words: 20
  max_self_bleu: 0.5              # Lower = more diverse
  min_sentiment_alignment: 0.7     # Sentiment must match rating
  
  # LLM Judge scores (0-1 scale)
  min_llm_judge_score: 0.7
  min_authenticity_score: 0.7
  min_alignment_score: 0.6
  min_expertise_score: 0.6
  min_uniqueness_score: 0.6
  
  # Process controls
  max_regeneration_attempts: 3     # How many times to retry bad reviews
```

**Personas**
```yaml
personas:
  - name: "enterprise_admin"
    characteristics: "Large company IT administrator"
    experience: "expert"
    focus: ["security", "integration", "scalability"]
    weight: 0.3              # 30% of reviews use this persona
```

## What You Get

**Generated Dataset (reviews.jsonl)**

Each review is saved as a JSON line with all metadata:

```jsonl
{
  "id": "rev_0001",
  "text": "After using this tool for 6 months in our enterprise environment...",
  "rating": 5,
  "persona": "enterprise_admin",
  "model": "openai:gpt-4",
  "seed_words": ["purple", "mountain", "database", ...],
  "generation_time": 3.2,
  "cost": 0.045,
  "statistical_scores": {
    "length": 187,
    "unique_words": 98,
    "self_bleu_diversity": 0.85,
    "sentiment_alignment": 0.92
  },
  "llm_judge_scores": {
    "overall_score": 0.87,
    "authenticity": 8.7,
    "alignment": 9.1,
    "expertise": 8.5,
    "uniqueness": 8.2
  },
  "final_decision": "PASS",
  "regeneration_count": 0
}
```

**Quality Report (quality_report.md)**

A comprehensive Markdown report that includes:
- Summary of how many samples were generated, time taken, and total cost
- Performance comparison between different models
- Diversity metrics across the dataset
- LLM judge evaluation results
- Distribution of ratings
- Comparison with real reviews if provided
- Analysis of rejections and why they failed
- Cost and performance breakdown
- Recommendations for improving future runs

**Detailed Logs**

- logs/generation/session.jsonl - Every generation attempt with full details
- logs/quality/session.jsonl - All quality check results
- logs/llm_judge/session.jsonl - LLM judge evaluations
- logs/pipeline/session_timestamp.log - Overall pipeline execution log

## Understanding the Quality Metrics

**Diversity Metrics**

Self-BLEU Diversity (target above 0.5): This measures how much vocabulary overlap exists between reviews. It's calculated as 1.0 minus the average BLEU score. Lower Self-BLEU means higher diversity, which is what we want.

Semantic Similarity (target below 0.7): Uses sentence embeddings to calculate average pairwise cosine similarity between reviews. Lower scores mean the reviews are talking about more diverse topics and using different phrasing.

Lexical Diversity: Measured through Type-Token Ratio (unique words divided by total words) and Moving Average TTR over 100-word windows. Higher numbers mean richer vocabulary usage.

**LLM Judge Criteria**

Each review is evaluated on a 0-10 scale across four dimensions:

Authenticity: Does it sound like a real person wrote it? Are there specific details and examples? Is it balanced with pros and cons? Does the emotional tone stay consistent? Does it avoid sounding templated or robotic?

Alignment: Does the sentiment actually match the star rating? A 5-star review shouldn't be full of complaints. The intensity level should be appropriate for the rating given.

Expertise: Does it demonstrate technical knowledge about the domain? Are the use cases realistic? Does it stay consistent with the persona's characteristics and experience level?

Uniqueness: Is the phrasing original compared to other reviews? Does it avoid generic statements like "great product" or "highly recommend"? Is the structure varied?

## Performance and Costs

**Expected Performance for 500 Samples**

When using a mix of models:
- Generation time: approximately 60-120 minutes (most of the tests were taking more time)
- Generation cost: around $45 (OpenAI ~$40, Gemini ~$5, Groq ~$0)
- LLM Judge cost: around $10 if using paid models, much less with Gemini
- Total cost: approximately $55
- Total time: about 60-120 minutes

**Cost Optimization Tips**

Use more Gemini: A 70/30 split (Gemini/GPT-4) reduces costs by about 40%

Skip LLM judge when testing: Using --skip-llm-judge makes it 60% faster and 40% cheaper, though you lose some quality assurance

Increase batch size: Processing in larger batches (100 instead of 50) is more efficient

Reduce regeneration attempts: Setting max_regeneration_attempts to 2 instead of 3 saves API calls

Use Groq for open source models: Since I had hardware limitations, Groq was essential for accessing Llama and Mistral models at high speed without local infrastructure

## Hardware and API Limits

**API Rate Limits**

OpenAI: 10,000 tokens per minute for GPT-4 on standard tier
Gemini: 15 requests per minute on free tier, much higher on paid
Groq: Very high throughput, great for open source models

**Recommendations**

For 500+ samples: Consider paid Gemini tier or split work across multiple API keys
For enterprise scale (1000+ samples): Implement proper request queuing and rate limiting
Memory requirements: Around 2GB RAM for embedding models and processing

**Token Limits**

GPT-4: 8K tokens per request, which is plenty for reviews
Gemini 2.0 Flash: 32K tokens per request
Groq models: Varies by model, typically 8K-32K
Most reviews use 200-500 tokens each

## Common Issues and Solutions

- when a review gets stuck in a rejection loop (like rev_0026 in the logs which failed 7+ times with the same sentiment-alignment issue), the system just keeps trying the same approach
```bash
source .env
echo $OPENAI_API_KEY  # Should print your key
```

**"NLTK data not found"**
Download the required data:
```bash
python -c "import nltk; nltk.download('punkt')"
```

**"Rate limit exceeded"**
- Reduce batch_size in your config
- Add delays between requests
- Upgrade your API tier or use multiple keys

**"LLM judge evaluation failed"**
- Check your Gemini API quota
- Verify your API key has access to gemini-2.0-flash-exp
- Try using the --skip-llm-judge flag temporarily

**Groq-specific issues**
- Most Groq errors are transient, retry usually works
- Check your Groq API key is valid
- Verify the model name matches Groq's available models

## Trade-offs I Made

**Quality vs Speed**
I chose comprehensive evaluation with the LLM judge, which makes things 2-3x slower but reduces bad reviews by about 40%. To mitigate this, I enabled parallel evaluation and added the --skip-llm-judge flag for when you just need speed.

**Cost vs Quality**
Mixing expensive GPT-4 with cheaper Gemini and free Groq models increases cost by about 50% but improves quality by roughly 15%. The weights are configurable, so you can adjust based on your budget.

**Diversity vs Realism**
Using high temperature (1.0) plus random seed words maximizes diversity but sometimes produces slightly less coherent reviews. The LLM judge filters out anything that crosses the line into unrealistic.

**Statistical vs LLM Evaluation**
I included both approaches even though it's slower and more expensive. The reason is that each method catches different issues. Statistical checks find repetition and formatting problems. The LLM judge catches subtle issues with tone, expertise, and authenticity that numbers can't measure.

**Local vs Cloud Models**
I wanted to support local models but didn't have the hardware. Groq solved this by providing fast inference for open source models like Llama and Mistral without requiring me to have a powerful GPU setup locally.

## Example of What Gets Generated

Here's a sample review that passed all quality checks:

```
I've been using this platform for our startup's development workflow for about 3 months now. 
As an intermediate-level developer managing a small team, I was initially drawn to the 
competitive pricing and minimal setup time.

The documentation is excellent - clear examples, comprehensive API references, and active 
community forums. Integration with our existing tools (GitHub, Slack) was surprisingly 
smooth. The API quality is solid, with consistent responses and good error handling.

However, I did encounter a few hiccups. The initial learning curve was steeper than expected, 
especially around some of the advanced features. Support response times were acceptable but 
not exceptional - usually 24-48 hours for non-critical issues.

Overall, for the cost and feature set, I'm satisfied with the decision. It's not perfect, 
but it solves our core problems efficiently. Would recommend for startups and small teams 
looking for a balance of functionality and affordability.

Rating: 4/5 stars

[Statistical Scores: Diversity=0.89, Sentiment Alignment=0.94]
[LLM Judge: Authenticity=8.8, Alignment=9.2, Expertise=8.4, Uniqueness=8.6]
```


---

## Future Improvements

I have several ideas for making this better:

**TrustCall for JSON Patching**
Planning to integrate TrustCall (https://github.com/hinthornw/trustcall) for more reliable JSON handling when the LLM generates structured outputs. This will reduce parsing errors and make the evaluation process more robust.

**DSPY Integration**
Want to use DSPY for better prompt optimization. Instead of manually tuning prompts, DSPY can automatically optimize them based on the quality metrics we're already tracking.

**Improved Models for Bias Detection**
The current bias detection relies on either keyword matching or a basic 5-star sentiment model (nlptown/bert-base-multilingual-uncased-sentiment). While functional, there are better alternatives:

- **Upgrade to more advanced sentiment models**: Consider using models like `cardiffnlp/twitter-roberta-base-sentiment-latest` or `distilbert-base-uncased-finetuned-sst-2-english` which have better nuance detection and handle edge cases more reliably
- **Multi-aspect sentiment analysis**: Instead of just overall sentiment, detect sentiment about specific aspects (price, features, support, usability) to catch subtle misalignments where overall sentiment matches but specific complaints contradict the rating
- **Emotion detection models**: Add models like `j-hartmann/emotion-english-distilroberta-base` to detect when emotional intensity doesn't match the rating (e.g., 5-star review with anger/frustration signals)
- **Fine-tune on review data**: The current models aren't specifically trained on product/service reviews. Fine-tuning on domain-specific review data would significantly improve accuracy
- **Ensemble approach**: Combine multiple sentiment models and use voting or weighted averaging to reduce false positives that cause rejection loops
- **Demographic bias detection**: Add dedicated models to detect and flag potential demographic biases or stereotypes in generated content
- **Replace circuit breaker with smarter detection**: The current system has a circuit breaker that auto-passes after 3 failures, which is a hack. Better models would reduce the need for this workaround

**Bias Preprocessing Improvements**
Beyond just better models, the preprocessing pipeline needs work:

- **Early bias screening**: Check for potential bias issues before generation by analyzing persona characteristics and seed words
- **Real-time bias monitoring**: Track bias metrics across the entire dataset as it's being generated, not just per-review
- **Contextual bias rules**: Different domains have different bias concerns - developer tools reviews shouldn't be judged the same as restaurant reviews
- **Adaptive threshold learning**: Instead of hardcoded thresholds that get loosened on retries, learn optimal thresholds from successful reviews
- **Bias report generation**: Automatically generate bias analysis reports showing distribution of sentiments, rating patterns, and potential skews

**Better Internal Workflow for Edge Cases**
Right now, some edge cases (like extremely short reviews or reviews that fail all models) don't have great handling. Need to build out more sophisticated fallback logic and recovery mechanisms.

**Unified Architecture**
The current system has separate code paths for different providers. Planning to unify this into a single, cleaner architecture that makes it easier to add new providers and reduces code duplication.

**Hardware Constraints Context**
Since I was working in a limited environment with low computing power and memory, I had to use Groq for accessing open source models. In the future, I'd like to support both local inference (for those with powerful machines) and cloud inference (via Groq or similar) in a more seamless way.

**Intelligent Retry Strategy with Dynamic Adjustments**
Currently, when a review gets stuck in a rejection loop (like rev_0026 in the logs which failed 7+ times with the same sentiment-alignment issue), the system just keeps trying the same approach. This wastes time and API costs. I need to implement:

- **Detection of repetitive failure patterns**: If a review fails 2-3 times with the same issue, don't just retry blindly
- **Dynamic prompt adjustments**: Automatically modify the generation prompt based on what's failing (e.g., if sentiment alignment is the issue, add explicit instructions about matching tone to rating)
- **Persona switching**: If one persona keeps producing misaligned reviews, try a different persona for that rating tier
- **Temperature adjustment**: Lower the temperature slightly on retries to get more controlled output
- **Early termination**: After 3 identical failures, skip to the next review instead of burning through all retry attempts
- **Fallback templates**: For reviews that consistently fail, use a more structured template approach as a last resort
- **Root cause analysis**: After each failure, analyze why it failed and adjust the next attempt accordingly rather than just trying again with the same parameters

This would have saved the 5+ minutes wasted on rev_0026 where it kept failing sentiment alignment checks repeatedly. The system should be smart enough to recognize "this isn't working" and try a different strategy instead of doing the same thing over and over.
