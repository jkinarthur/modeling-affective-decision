# AD-DAN Architecture

Comprehensive guide to the Affect–Decision Dual Alignment Network architecture.

## Overview

AD-DAN is a unified neural architecture with four key innovations:

1. **Dual-Objective Formulation**: Jointly optimize affective alignment and decision quality
2. **Cross-Modal Inconsistency Attention (CMIA)**: Explicit bidirectional coupling between modules
3. **Inconsistency-Aware Loss**: Provable upper bound on expected ADI
4. **Distribution-Aligned RL Reward**: Closes train/evaluation distribution gap

---

## Architecture Diagram

```
                    Input: Context + User Message
                                 |
                    Shared BART Context Encoder
                                 |
                  +---------------+---------------+
                  |                               |
      Parallel Affective               Decision Evaluation
      Alignment Module (PAA)           Module (DEM)
           (DEM dropout)                  (dropout)
           Emotion scoring               Binary scoring
         Output: S_aff ∈ [0,1]        Output: S_dec ∈ [0,1]
                  |                              |
                  +----------+-----+----------+--+
                             |
              Cross-Modal Inconsistency
                 Attention (CMIA)
              Bidirectional gradients
                             |
                    Inconsistency Loss
                  L_adi = max(0, S_aff - S_dec - λ)
                             |
                    Total Loss (SFT Stage)
                    L = L_aff + L_dec + L_adi
                             |
                   [PPO Critic Head] (RL only)
                     Value function
```

---

## Module Descriptions

### 1. Shared Context Encoder

**Model**: Microsoft DeBERTa-v3-base (183M parameters)

- Encodes dialogue context (system messages + user utterance)
- Disentangled attention mechanism for better long-range dependencies
- Frozen during SFT (optional: fine-tune last 2 layers with weight decay)

**Input**: Tokenized context (max length: 512 tokens)  
**Output**: Contextualized embeddings `H ∈ ℝ^{T × 768}`

### 2. Parallel Affective Alignment Module (PAA)

**Purpose**: Score how well the response validates/aligns with user's emotional state

**Architecture**:
```
    H (encoder output)
        |
    [Dropout: paa_dropout]
        |
    [Linear: 768 → 256]
    [ReLU]
    [Dropout: paa_dropout]
        |
    [Linear: 256 → num_emotions]
    [Softmax]
        |
    Fine-tuned Emotion Model (RoBERTa distilbase)
    Computes emotion alignment score S_aff ∈ [0,1]
```

**Loss**: Cross-entropy with label smoothing
```
L_aff = CrossEntropy(
    predicted_emotions,
    target_emotions,
    label_smoothing=emotion_label_smoothing
)
```

### 3. Decision Evaluation Module (DEM)

**Purpose**: Score whether the suggested action/decision is appropriate and safe

**Architecture**:
```
    H (encoder output)
        |
    [Dropout: dem_dropout]
        |
    [Linear: 768 → 256]
    [ReLU]
    [Dropout: dem_dropout]
        |
    [Linear: 256 → 1]
    [Sigmoid]
        |
    S_dec ∈ [0,1]  (binary decision quality)
```

**Loss**: Binary cross-entropy
```
L_dec = BCELoss(S_dec, decision_labels)
```

### 4. Cross-Modal Inconsistency Attention (CMIA)

**Purpose**: Create explicit bidirectional gradient coupling between PAA and DEM during backpropagation

**Forward Pass**:
```python
# Compute attention weights between module representations
R_aff = PAA(H)  # Affective representation
R_dec = DEM(H)  # Decision representation

# Cross-attention: learn how decision quality depends on affective output
alpha = softmax(W_cross @ R_aff)
R_dec_attended = alpha * R_dec

# Backward pass: gradients flow bidirectionally
loss.backward()  # ∇_R_aff receives gradient flow from L_dec
                 # ∇_R_dec receives gradient flow from L_aff (via CMIA)
```

**Implementation**: Attention mechanism in `src/models/modules.py`

### 5. Inconsistency-Aware Dual-Objective Loss

**Definition**:
```
L_adi(S_aff, S_dec, τ) = max(0, S_aff - S_dec - τ)
```

**Interpretation**:
- When `S_aff - S_dec > τ`: model is inconsistent (high affective, low decision)
- When `S_aff - S_dec ≤ τ`: model is consistent or safe (no penalty)
- τ = threshold (typically 0.5)

**Total SFT Loss**:
```
L_total = λ_aff * L_aff + λ_dec * L_dec + λ_adi * L_adi
```

Default weights: λ_aff = 1.0, λ_dec = 1.0, λ_adi = 0.5

**Theoretical Guarantee** (Theorem in paper):
```
E[ADIR] ≤ (1/N) * Σ E[L_adi] / τ
```

---

## RL Stage (PPO with Distribution-Aligned Reward)

### Policy & Value Network

- **Policy Actor**: Use SFT-trained model to generate text via greedy decoding
- **Value Critic**: Additional 2-layer MLP on top of encoder to estimate V(s)

### Composite Reward Function

**Key Innovation**: Distribution-aligned direct reward term

```python
def compute_reward(
    generated_text: str,
    original_text: str,
    model: ADDAN,
    tau: float = 0.50,
    config: RewardConfig = None
) -> float:
    """
    Composite reward combining:
    1. Empathetic quality (BLEU vs. references)
    2. Decision correctness (binary decision score)
    3. Inconsistency penalization
    4. Distribution-aligned direct reward (CRITICAL)
    """
    
    # Score model-generated text (exploration)
    S_aff_gen, S_dec_gen = model.score(generated_text)
    
    # Score original dataset text (distribution alignment)
    S_aff_orig, S_dec_orig = model.score(original_text)
    
    # Partial rewards
    r_empathy = bleu_score(generated_text, references)
    r_decision = S_dec_gen
    r_adi = -max(0, S_aff_gen - S_dec_gen - tau)
    
    # CRITICAL: Direct reward on original response
    r_direct = S_dec_orig  # or S_aff_orig, configurable
    
    # Composite reward
    reward = (
        config.reward_alpha * r_empathy +
        config.reward_beta * r_decision +
        config.reward_gamma_adi * r_adi +
        config.reward_direct_alpha * r_direct  # Distribution alignment
    )
    
    return reward
```

### Why Distribution Alignment Matters

**Problem without direct reward**:
- RL reward scores model-generated text (shifted distribution)
- Test evaluation scores original dataset text (different distribution)
- Leads to poor transfer: training ADIR ≠ test ADIR

**Solution**:
- Include both in the same reward computation
- Model learns to optimize consistent with evaluation metric
- Test ADIR becomes meaningful and non-trivial

### PPO Training Loop

```python
for epoch in range(num_rl_epochs):
    for batch in dataloader:
        # Generate exploration samples
        generated = actor.generate(batch.context)
        
        # Compute rewards
        rewards = [compute_reward(gen, orig) for gen, orig in zip(generated, batch.original)]
        
        # Compute advantages
        values = critic(batch.context)
        advantages = rewards - values.detach()
        returns = advantages + values
        
        # PPO loss: clip surrogate
        for ppo_epoch in range(ppo_epochs):
            logprobs_new = actor.log_prob(generated)
            ratio = exp(logprobs_new - logprobs_old)
            surr1 = ratio * advantages
            surr2 = clip(ratio, 1-eps, 1+eps) * advantages
            
            loss_policy = -min(surr1, surr2).mean()
            loss_value = mse_loss(values, returns)
            loss_kl = kl_divergence(actor, sft_model)
            
            loss = loss_policy + 0.5*loss_value + 0.01*loss_kl
            loss.backward()
            optimizer.step()
    
    # Save checkpoint per epoch
    save_checkpoint(actor, f"rl_epoch_{epoch:02d}")
```

---

## Configuration

All hyperparameters are defined in `ADDANConfig`:

```python
@dataclass
class ADDANConfig:
    # Encoder & Model
    encoder_model: str = "microsoft/deberta-v3-base"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    bart_model: str = "facebook/bart-base"  # RL stage only
    
    # Module dimensions
    hidden_dim: int = 768
    mlp_dim: int = 256
    num_emotions: int = 28
    
    # Dropout & Regularization
    paa_dropout: float = 0.1
    dem_dropout: float = 0.1
    cmia_dropout: float = 0.1
    weight_decay: float = 0.01
    emotion_label_smoothing: float = 0.1
    
    # Loss weights
    lambda_aff: float = 1.0
    lambda_dec: float = 1.0
    lambda_adi: float = 0.5
    
    # RL reward coefficients
    reward_alpha: float = 0.35          # Empathetic quality
    reward_beta: float = 0.85           # Decision correctness
    reward_gamma_adi: float = 0.80      # ADI penalty
    reward_adir_tau_bonus: float = 0.80 # Low ADIR bonus
    reward_tau_sharpness: float = 16.0  # Sharpness of tau boundary
    reward_margin_spread: float = 0.05  # Margin around tau
    reward_direct_alpha: float = 1.5    # Distribution-aligned term
    
    # Threshold
    tau: float = 0.50                   # ADI threshold
```

---

## Extension Points

### 1. Custom Reward Functions
Modify `compute_reward()` in `src/training/rl_trainer.py`:
```python
def compute_reward(self, ...):
    # Add custom metrics here
    r_custom = compute_something_novel(...)
    return reward + 0.1 * r_custom
```

### 2. Alternative Encoders
Update `ADDANConfig.encoder_model`:
```python
config.encoder_model = "roberta-large"  # ~355M params
config.encoder_model = "t5-base"         # ~220M params
```

### 3. Multi-Task Learning
Add auxiliary losses in the loss computation:
```python
L_total = (
    lambda_aff * L_aff +
    lambda_dec * L_dec +
    lambda_adi * L_adi +
    lambda_aux * L_auxiliary  # Custom task
)
```

---

## Performance Characteristics

**Single T4 GPU (16GB VRAM)**:
- Batch size: 8 (SFT), 8 (RL)
- SFT: ~2 hours (20 epochs)
- RL: ~5.5 hours (6 epochs, 5 PPO updates)
- Total: ~7.5 hours

**Memory Breakdown**:
- Model: ~1.2 GB (DeBERTa + heads)
- Batch: ~3.5 GB (8 samples × 512 tokens)
- Optimizer state: ~2.5 GB (Adam)
- Activations: ~7 GB (gradients)

---

## References

- DeBERTa: He et al., 2020. "[DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://openreview.net/pdf?id=XPLEEFVeF-)"
- PPO: Schulman et al., 2017. "[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)"
- BLEU: Papineni et al., 2002. "[BLEU: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040.pdf)"

---

**Last Updated**: April 28, 2026
