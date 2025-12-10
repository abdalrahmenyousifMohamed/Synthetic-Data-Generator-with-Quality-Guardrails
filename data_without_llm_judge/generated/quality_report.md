# Synthetic Review Dataset Quality Report

Generated: 2025-12-09 19:16:49

## Generation Summary

- **Target Samples**: 300
- **Samples Generated**: 300
- **Samples Accepted**: 300 (100.0% success rate)
- **Samples Rejected**: 0
- **Total Regenerations**: 0
- **Generation Time**: 178.7 minutes
- **Total Cost**: $7.93
- **LLM Judge Evaluations**: 0

## Model Performance Comparison

| Model | Samples | Accepted | Accept Rate | Avg Quality | LLM Judge Score | Avg Time | Cost | Cost/Sample |
|-------|---------|----------|-------------|-------------|-----------------|----------|------|-------------|
| openai:gpt-4 | 236 | 236 | 100.0% | 0.80 | 1.00 | 20.42s | $7.93 | $0.034 |
| google:gemini-2.0-flash-exp | 64 | 64 | 100.0% | 0.80 | 1.00 | 4.63s | $0.00 | $0.000 |

## Diversity Metrics

- **Overall Diversity Score**: 0.530 ✗
- **Self-BLEU Diversity**: 0.729 (higher is better)
- **Semantic Diversity**: 0.480 (higher is better)
- **Vocabulary Size**: 10922 unique tokens
- **Type-Token Ratio**: 0.121
- **MATTR**: 0.802
- **Bigram Diversity**: 0.569
- **Trigram Diversity**: 0.864

### Semantic Similarity Analysis

- **Average Similarity**: 0.520 ✓
- **Max Similarity**: 0.828
- **Min Similarity**: 0.163

## LLM-as-a-Judge Evaluation Results

**PRIMARY JUDGE**: Gemini 2.0 Flash Exp (as specified in requirements)



## Rating Distribution

- **1-Star**: 16 (5.3%)
- **2-Star**: 37 (12.3%)
- **3-Star**: 50 (16.7%)
- **4-Star**: 106 (35.3%)
- **5-Star**: 91 (30.3%)



## Cost & Performance Analysis

### Generation Costs

- **openai:gpt-4**: $7.93 (236 samples) = $0.0336/sample
- **google:gemini-2.0-flash-exp**: $0.00 (64 samples) = $0.0000/sample

**Total Generation Cost**: $7.93

### Time Performance

- **Total Pipeline Time**: 178.7 minutes
- **Throughput**: 1.7 samples/minute
- **Average Time per Accepted Sample**: 35.7s

## Recommendations

1. **Quality Improvements**:
   - Current diversity score: 0.53
   - Consider increasing seed word variety

2. **Cost Optimization**:
   - Most cost-effective: google:gemini-2.0-flash-exp at $0.0000/sample

3. **Performance Optimization**:
   - Current throughput: 1.7 samples/minute
   - Consider parallel processing for faster generation


---

