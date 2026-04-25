# ============================================================
# AD-DAN  –  Affective–Decision Inconsistency Detection
# Base: PyTorch 2.1.0 + CUDA 12.1 + cuDNN 8
# Compatible with: AWS p3/p3dn/g4dn/g5 instances
# ============================================================

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ---- system dependencies -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- working directory ---------------------------------------------
WORKDIR /app

# ---- Python dependencies (cached layer) ----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ---- source code ---------------------------------------------------
COPY . .

# ---- HuggingFace cache directory (mapped to volume in compose) -----
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# ---- data / checkpoint directories ---------------------------------
RUN mkdir -p /app/data /app/checkpoints

# ---- default command: show usage -----------------------------------
CMD ["python", "-c", "\
import textwrap; \
print(textwrap.dedent('''\n\
    AD-DAN container ready.\n\
\n\
    Typical workflow:\n\
      # 1. Build TAD-Bench 10K dataset\n\
      docker compose run --rm addan build\n\
\n\
      # 2. SFT training\n\
      docker compose run --rm addan train-sft\n\
\n\
      # 3. RL fine-tuning\n\
      docker compose run --rm addan train-rl\n\
\n\
      # 4. Evaluate\n\
      docker compose run --rm addan evaluate\n\
'''))"]
