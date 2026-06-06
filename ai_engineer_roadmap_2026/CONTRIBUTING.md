<div align="center">

# 🤝 Contributing to AI Engineer Roadmap 2026

**Thank you for your interest in contributing!** Every contribution — from fixing a typo to adding a full notebook — helps thousands of learners on their AI engineering journey.

</div>

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflows](#contribution-workflows)
  - [Fixing Bugs or Typos](#fixing-bugs-or-typos)
  - [Adding a Learning Resource](#adding-a-learning-resource)
  - [Creating a Notebook](#creating-a-notebook)
  - [Building a Project](#building-a-project)
  - [Improving Documentation](#improving-documentation)
- [Code Style Guide](#code-style-guide)
- [Notebook Standards](#notebook-standards)
- [Pull Request Process](#pull-request-process)
- [Contributor Ladder](#contributor-ladder)
- [Recognition](#recognition)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior by opening an issue.

---

## How Can I Contribute?

| Contribution Type | Difficulty | Good First Issue? | Description |
|-------------------|:---------:|:-----------------:|-------------|
| 🐛 **Bug Fix / Typo** | ⭐ | ✅ | Fix errors in notebooks, docs, or code |
| 📚 **Add Resource** | ⭐ | ✅ | Submit a learning resource to RESOURCES.md |
| 🧪 **Add Tests** | ⭐⭐ | ✅ | Write unit tests for project code |
| 📝 **Improve Docs** | ⭐⭐ | ✅ | Enhance explanations, add diagrams, fix formatting |
| 🌍 **Translate** | ⭐⭐ | ✅ | Translate notebooks or docs to other languages |
| 📓 **Create Notebook** | ⭐⭐⭐ | | Add a new educational notebook |
| 💻 **Build Project** | ⭐⭐⭐⭐ | | Contribute a capstone project |
| 🏗️ **Infrastructure** | ⭐⭐⭐ | | Improve CI/CD, Docker, or tooling |

---

## Getting Started

### Find Something to Work On

1. **Check the [Issues](https://github.com/YOUR_USERNAME/ai-engineer-roadmap-2026/issues)** — look for `good first issue` and `help wanted` labels
2. **Read the [ROADMAP.md](ROADMAP.md)** — identify gaps in coverage
3. **Browse [RESOURCES.md](RESOURCES.md)** — find missing resources to add
4. **Propose something new** — open an issue to discuss before starting large contributions

### Before You Start

- **For small changes** (typos, formatting, broken links): Just submit a PR directly
- **For medium changes** (new resource, test additions): Open an issue first to confirm it's needed
- **For large changes** (new notebook, new project): Open an issue and get approval before investing significant time

---

## Development Setup

### Prerequisites

- Python 3.11+
- Git
- [Ollama](https://ollama.com/) (for Phase 4+ notebooks)

### Setup

```bash
# 1. Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ai-engineer-roadmap-2026.git
cd ai-engineer-roadmap-2026

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development dependencies
pip install ruff black isort mypy pytest nbval

# 5. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 6. Create a branch for your work
git checkout -b feature/your-feature-name
```

### Verify Your Setup

```bash
# Run linting
ruff check .

# Run formatting check
black --check .

# Run tests
pytest

# Validate notebooks (CPU-only subset)
python scripts/validate_notebooks.py --cpu-only
```

---

## Contribution Workflows

### Fixing Bugs or Typos

1. Fork → Clone → Branch
2. Make your fix
3. Submit a PR with a clear description of what you fixed

### Adding a Learning Resource

1. Verify the resource is:
   - **Free** or has a substantial free tier
   - **High quality** and actively maintained
   - **Not already listed** in [RESOURCES.md](RESOURCES.md)
2. Add the resource to the appropriate phase section in `RESOURCES.md`
3. Use the established table format:

```markdown
| [Resource Title](https://url.com) | 🎥 | ⭐ | Brief description of why this resource is valuable. |
```

4. Tag appropriately:
   - Use `⭐` (Essential) only for truly foundational resources
   - Use `📎` (Supplementary) for deeper explorations

### Creating a Notebook

New notebooks must follow our **Notebook Standards** (see below). The process:

1. **Open an issue** describing:
   - Topic and learning objectives
   - Which roadmap phase it belongs to
   - Why it's needed (gap analysis)
2. **Get approval** from a maintainer
3. **Create the notebook** following the template structure
4. **Ensure it runs top-to-bottom** without errors
5. **Submit a PR** with the notebook and updated phase README

#### Notebook Template

Every notebook must contain these 10 sections:

```python
# %% [markdown]
# # Notebook Title
# 
# ## 1. Overview
# Brief description of what this notebook covers.
#
# ## 2. Learning Objectives
# By the end of this notebook, you will be able to:
# - Objective 1
# - Objective 2
# - Objective 3

# %% [markdown]
# ## 3. Imports

# %%
import numpy as np
import torch
# ... all imports here with version comments

# %% [markdown]
# ## 4. Configuration

# %%
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ... all config here

# %% [markdown]
# ## 5. Theory
# Explain the concept with intuitive descriptions, diagrams, and LaTeX equations.

# %% [markdown]
# ## 6. Implementation
# Production-quality code with type hints, docstrings, and error handling.

# %% [markdown]
# ## 7. Evaluation
# Metrics, plots, analysis of results.

# %% [markdown]
# ## 8. Exercises
# Guided practice problems with clear instructions.

# %% [markdown]
# ## 9. Challenge Problems
# Open-ended, portfolio-worthy challenges.

# %% [markdown]
# ## 10. Further Reading
# - [Paper Title](https://arxiv.org/abs/...)
# - [Documentation](https://...)
```

### Building a Project

Projects are significant contributions. Follow this process:

1. **Open an issue** with your project proposal including:
   - Problem statement
   - Architecture overview
   - Tech stack
   - Estimated difficulty and time
2. **Get approval** from a maintainer
3. **Follow the project structure**:

```
projects/XX-project-name/
├── README.md               # Setup, usage, architecture
├── requirements.txt         # Pinned dependencies
├── architecture.md          # Design document
├── src/
│   ├── __init__.py
│   └── ...                 # Modular, typed Python
├── tests/
│   └── ...                 # pytest tests
└── assets/
    └── ...                 # Diagrams, screenshots
```

4. **Ensure all tests pass**: `pytest projects/XX-project-name/tests/`
5. **Submit a PR** with the project and updated PROJECTS.md

### Improving Documentation

- Documentation lives in `docs/` (MkDocs format)
- Follow the existing structure
- Include diagrams where helpful (Mermaid preferred)
- Test locally: `mkdocs serve`

---

## Code Style Guide

We use automated tools to maintain consistent code style:

| Tool | Purpose | Config |
|------|---------|--------|
| **Ruff** | Linting | `pyproject.toml` |
| **Black** | Formatting (88 char line width) | `pyproject.toml` |
| **isort** | Import sorting | `pyproject.toml` |
| **mypy** | Type checking | `pyproject.toml` |

### Key Rules

- **Type hints** on all function signatures
- **Docstrings** on all public functions and classes (Google style)
- **No wildcard imports** (`from module import *`)
- **Constants** in UPPER_SNAKE_CASE
- **Variables/functions** in lower_snake_case
- **Classes** in PascalCase

### Example

```python
def compute_cosine_similarity(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between a query and document embeddings.

    Args:
        query_embedding: A 1D array of shape (embedding_dim,).
        document_embeddings: A 2D array of shape (num_docs, embedding_dim).

    Returns:
        A 1D array of similarity scores, one per document.

    Raises:
        ValueError: If embedding dimensions don't match.
    """
    if query_embedding.shape[0] != document_embeddings.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query has {query_embedding.shape[0]}, "
            f"documents have {document_embeddings.shape[1]}"
        )

    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = document_embeddings / np.linalg.norm(
        document_embeddings, axis=1, keepdims=True
    )
    return doc_norms @ query_norm
```

---

## Notebook Standards

### Quality Requirements

| Requirement | Description |
|-------------|-------------|
| **Runs top-to-bottom** | No manual intervention needed. Cell execution order must be sequential. |
| **Reproducible** | Set random seeds. Pin dependency versions. Use public datasets. |
| **No toy examples** | Every notebook teaches a practical, real-world skill. |
| **No placeholders** | No `TODO`, no fake data, no fake metrics. |
| **Real datasets** | Use HuggingFace Hub, torchvision, UCI ML, or Kaggle datasets. |
| **Explained** | Every code block has a preceding markdown cell explaining *why*. |

### Beginner Notebooks (Phases 1–2)

- Avoid unnecessary abstractions
- Heavy inline comments
- Step-by-step explanations
- Visual outputs (plots, tables)

### Advanced Notebooks (Phases 4–6)

- Production-quality code patterns
- Type hints throughout
- Error handling and edge cases
- Performance considerations

---

## Pull Request Process

1. **Create your PR** against the `main` branch
2. **Fill out the PR template** completely
3. **Ensure CI passes**:
   - Linting (Ruff)
   - Formatting (Black)
   - Tests (pytest)
   - Notebook validation (if applicable)
4. **Request review** from a maintainer
5. **Address feedback** promptly
6. **Squash and merge** once approved

### PR Title Convention

Use conventional commit prefixes:

```
feat: Add RAG fundamentals notebook
fix: Correct import in supervised learning notebook
docs: Update Phase 4 resources
test: Add tests for semantic search project
chore: Update dependencies in requirements.txt
```

---

## Contributor Ladder

We value sustained contributions and offer a clear path for growth:

| Level | Role | How to Reach |
|:-----:|------|--------------|
| 🌱 | **Contributor** | Submit your first merged PR |
| 🌿 | **Regular Contributor** | 5+ merged PRs across different areas |
| 🌳 | **Reviewer** | Demonstrated expertise; invited to review PRs |
| 🏔️ | **Maintainer** | Consistent, high-quality contributions; invited by existing maintainers |

---

## Recognition

All contributors are recognized in our README's Contributors section. We use the [All Contributors](https://allcontributors.org/) specification to acknowledge every type of contribution — code, documentation, design, ideas, and more.

---

<div align="center">

**Thank you for helping build the most comprehensive AI Engineer learning resource on GitHub! ❤️**

**[← Back to README](README.md)**

</div>
