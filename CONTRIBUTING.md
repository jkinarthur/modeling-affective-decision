# Contributing to AD-DAN

Thank you for your interest in contributing to the Affect–Decision Dual Alignment Network project!

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:
- Be respectful and professional
- Provide constructive feedback
- Focus on ideas, not personal attacks
- Help ensure a harassment-free experience for everyone

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the existing issues to avoid duplicates.

**When filing a bug report, please include:**
1. Clear, descriptive title
2. Exact reproduction steps
3. Expected behavior vs. actual behavior
4. Environment details: OS, Python version, PyTorch version, CUDA version
5. Relevant error messages or logs
6. If possible, a minimal reproducible example

### Suggesting Enhancements

**Enhancement suggestions should include:**
1. Use case and motivation
2. Proposed solution or design
3. Alternative approaches considered
4. Expected impact on performance/usability

### Pull Requests

We actively welcome pull requests! Here's the process:

1. **Fork** the repository and create a new branch: `git checkout -b feature/your-feature-name`
2. **Make your changes** with clear, descriptive commits
3. **Test thoroughly** — ensure your changes don't break existing functionality
4. **Follow code style** — use PEP 8 conventions, type hints, and docstrings
5. **Update documentation** — reflect changes in README, docstrings, and comments
6. **Submit a PR** with:
   - Clear title and description
   - Reference to any related issues (`Fixes #123`)
   - Brief summary of changes
   - Test results or coverage information

### Code Style Guide

We follow PEP 8 with the following additional conventions:

```python
# Type hints required for function signatures
def compute_reward(generated_text: str, original_text: str, tau: float = 0.50) -> float:
    """
    Compute composite reward for RL training.
    
    Args:
        generated_text: Model-generated response text
        original_text: Original dataset response for distribution alignment
        tau: Inconsistency threshold (default: 0.50)
    
    Returns:
        Scalar reward value
    """
    pass

# Docstrings: use Google-style format
# Line length: 100 characters max
# Imports: organize by standard library, third-party, local
```

### Testing

- Write tests for all new functionality in `tests/` directory
- Run tests locally before submitting PR: `pytest tests/`
- Ensure tests pass on both CPU and GPU (if applicable)
- Target >80% code coverage for new code

### Documentation

- Update [README.md](README.md) with usage examples for new features
- Add docstrings to all functions and classes
- Include inline comments for complex logic
- Update [ARCHITECTURE.md](ARCHITECTURE.md) if you modify model structure

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/jkinarthur/modeling-affective-decision.git
cd modeling-affective-decision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code quality checks
flake8 src/
mypy src/ --ignore-missing-imports
black src/ --check
pytest tests/
```

## Project Structure

```
modeling-affective-decision/
├── src/
│   ├── data/              # Dataset loading & preprocessing
│   ├── models/            # AD-DAN architecture
│   ├── training/          # SFT & RL training loops
│   └── utils/             # Evaluation metrics, helpers
├── train.py               # Entry point for training
├── evaluate.py            # Evaluation script
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Areas for Contribution

We're particularly interested in contributions to:

1. **Multi-GPU Training**: Distributed training support
2. **Alternative Encoders**: Support for other pretrained models (T5, LLaMA, etc.)
3. **Evaluation Metrics**: New ADI metrics or domain-specific variants
4. **Benchmark Expansion**: Additional dialogue datasets with ADI annotations
5. **Performance Optimization**: Memory efficiency, inference speedup
6. **Documentation**: Additional examples, tutorials, API reference

## Questions?

- **GitHub Issues**: Best for bugs and feature requests
- **Discussions**: For general questions or design feedback
- **Email**: Contact maintainers directly (see README)

---

Thank you for contributing to making conversational AI safer and more reliable!
