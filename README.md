# SCARL: Self-Corrective Agentic Reinforcement Learning
A dual-policy reinforcement learning framework that enables AI agents to self-evaluate and self-correct during task execution.

## Overview
SCARL introduces a novel approach to building robust AI agents by combining three interacting components:

- **Primary Policy (πₚ)**: Handles standard task execution (dialogue, code generation, navigation)
- **Self-Evaluation Layer (R_meta)**: Generates internal quality signals based on uncertainty, constraint violations, and self-critique
- **Corrective Policy (πc)**: Takes over when quality drops below threshold, executing corrective actions like replanning or requesting clarification

## Key Features
- **Dual-Policy Architecture**: Automatic switching between primary execution and corrective modes
- **Meta-Reward System**: Internal quality assessment without external feedback
- **Multi-Task Support**: Validated across dialogue, summarization, QA, translation, and advanced conversation tasks
- **High Stability**: 100% task completion maintained across 500 training episodes
- **Efficient Complexity**: O(1) per-step execution with O(D²) space complexity

## How It Works

SCARL uses a composite reward function:

```
R_total = R_ext + λ * R_meta
```

Where:
- `R_ext`: External task success reward
- `R_meta`: Internal quality signal (uncertainty, constraint checks, self-critique)
- `λ`: Balancing coefficient

The agent switches policies dynamically:

```
π(s) = πₚ(s)  if R_meta ≥ θ_meta
       πc(s)  if R_meta < θ_meta
```

## Architecture Components

### 1. Unified State Encoder
- BERT-based text encoding (max 512 tokens)
- LSTM for memory processing (last 5 memories)
- Fixed-dimensional embeddings for diverse state types

### 2. Meta-Reward Generator
- 3-layer neural network predicting:
  - Performance scores
  - Correction probabilities
  - Quality estimates (coherence, accuracy, relevance)
  - Overall meta-reward signal

### 3. Corrective Policy Actions
- `REPLAN`: Restart strategy
- `SEARCH`: Query external knowledge
- `CRITIQUE`: Self-analyze previous action
- `ASK`: Request user clarification
- `REFINE`: Improve previous output

## Performance Results

| Task | Avg Reward | Correction Frequency | Episode Length | Stability |
|------|------------|---------------------|----------------|-----------|
| Advanced Dialogue | 5.94 | 91.7% | 12 steps | High |
| Dialogue QA | 3.97 | 86.3% | 8 steps | High |
| Text Summarization | 0.65 | 7.5% | 1.5 steps | Medium |
| Question Answering | 0.45 | 0% | 1 step | High |
| Translation | 0.53 | 0% | 1 step | High |

**Key Findings:**
- Complex multi-turn tasks benefit most from self-correction (91.7% correction rate)
- Simple single-step tasks require minimal correction
- Stable training with no performance collapses across 500 episodes
- Task-dependent correction patterns validate adaptive behavior

## Complexity Analysis

### Time Complexity
- **Per-step**: O(S + D² + A) ≈ O(1) with bounded dimensions
- **Training**: O(B × D²) per batch (B=32 batch size)
- Dominant factor: Meta-reward generator at O(D²)

### Space Complexity
- **BERT encoder**: O(V × D) where V ≈ 30,000 vocabulary
- **Neural networks**: O(D²) for parameters
- **Replay buffer**: O(N × D) for N=10,000 experiences

## Known Limitations

- **Reward Model Calibration**: Significant reward scale differences between tasks require task-specific tuning
- **External Dependencies**: Relies on HuggingFace model hub (5-retry mechanism included)
- **Weight Initialisation**: BERT classifier heads require careful initialisation for optimal performance
- **Single-Step Tasks**: Minimal benefit for tasks that don't require sequential reasoning

## Future Research Directions

- Principled meta-reward design through learned critique models
- Shared vs separate weight architectures for πₚ and πc
- Pre-training reward models on human preference datasets
- Adversarial meta-reward generator training
- Unified state representations for cross-task transfer learning