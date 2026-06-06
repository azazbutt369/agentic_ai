# Contributing to AI Engineer Roadmap 2026

First off, thank you for considering contributing.

This repository aims to become the most practical, comprehensive, and up-to-date roadmap for modern AI Engineering.

Whether you're fixing a typo, adding a notebook, contributing a project, or improving documentation, your contribution is valuable.

---

# Mission

Our goal is to help learners become production-ready AI Engineers through:

- Structured learning paths
- Hands-on projects
- Reproducible notebooks
- Production deployment examples
- Community-driven improvements

---

# Ways to Contribute

You do not need to be an expert to contribute.

## Documentation

Examples:

- Fix typos
- Improve explanations
- Add diagrams
- Update outdated resources
- Improve readability

---

## Learning Resources

Examples:

- Add books
- Add papers
- Add courses
- Add tutorials
- Add learning paths

---

## Notebooks

Examples:

- Mathematics notebooks
- Machine learning notebooks
- Deep learning notebooks
- RAG implementations
- Agent examples

Requirements:

- Reproducible
- Well documented
- Clear instructions
- No broken dependencies

---

## Projects

Examples:

- RAG applications
- AI agents
- LLM applications
- MLOps projects
- Multimodal applications

Requirements:

- README included
- Installation instructions
- Architecture diagram
- Example outputs
- License compatibility

---

## Research Summaries

We welcome:

- Paper breakdowns
- Architecture explanations
- Model comparisons
- Industry trend reports

Format:

```markdown
# Paper Title

## Summary

## Key Contributions

## Strengths

## Weaknesses

## Practical Applications
```

---

# Before You Start

Please check:

- Existing issues
- Existing pull requests
- Roadmap structure

Avoid duplicate contributions.

---

# Repository Structure

```text
foundations/
machine-learning/
deep-learning/
generative-ai/
rag/
agents/
llmops/
multimodal/
safety/
projects/
resources/
community/
```

Please place contributions in the appropriate directory.

---

# Development Setup

## Fork Repository

```bash
git clone https://github.com/<your-username>/ai-engineer-roadmap.git
```

---

## Create Branch

```bash
git checkout -b feature/my-feature
```

Examples:

```bash
feature/rag-project
feature/transformer-notebook
fix/typo-foundations
docs/update-resources
```

---

# Commit Guidelines

Use conventional commit messages.

Examples:

```text
feat: add graph rag project
feat: add transformer notebook

fix: correct typo in deep learning section

docs: update resource directory

refactor: reorganize project structure
```

Avoid:

```text
update
changes
fixed stuff
```

---

# Pull Request Requirements

Every pull request should include:

- Purpose of contribution
- Files modified
- Screenshots (if applicable)
- Testing information

Template:

```markdown
## Description

What does this PR do?

## Type

- Documentation
- Notebook
- Project
- Resource
- Bug Fix

## Checklist

- [ ] Tested locally
- [ ] Documentation updated
- [ ] No broken links
- [ ] Follows repository structure
```

---

# Notebook Standards

All notebooks should:

- Execute from start to finish
- Contain explanations
- Use reproducible seeds
- Include requirements

Recommended structure:

```text
01_imports
02_data
03_processing
04_training
05_evaluation
06_conclusion
```

---

# Python Standards

Preferred:

- Python 3.11+
- Type hints
- Black formatting
- Ruff linting

Example:

```python
def calculate_accuracy(
    predictions: list[int],
    labels: list[int]
) -> float:
    ...
```

---

# Documentation Standards

Use:

```markdown
# Title

## Overview

## Concepts

## Example

## Further Reading
```

Avoid large walls of text.

Use:

- Diagrams
- Tables
- Examples

---

# Project Standards

Each project should contain:

```text
project-name/

README.md
requirements.txt
src/
notebooks/
tests/
assets/
```

---

# Resource Submission Guidelines

Resources must be:

- Free or clearly labeled as paid
- Relevant to AI Engineering
- Actively maintained
- High quality

Preferred categories:

- Courses
- Books
- Papers
- GitHub Repositories
- Videos
- Communities

---

# Good First Issues

New contributors should start with:

- Documentation fixes
- Broken links
- Resource updates
- Diagram improvements

These issues are labeled:

```text
good-first-issue
```

---

# Contributor Recognition

Contributors may be featured in:

- README
- Contributors Page
- Release Notes

---

# Code of Conduct

By participating in this project, you agree to follow the Code of Conduct.

Please read:

CODE_OF_CONDUCT.md

---

# Questions?

Open a Discussion or Issue.

We are building this roadmap together.

Thank you for contributing.
