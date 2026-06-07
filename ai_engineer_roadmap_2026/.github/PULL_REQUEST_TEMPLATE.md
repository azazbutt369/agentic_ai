## Description

<!-- Provide a brief description of the changes in this PR. -->

### What does this PR do?

<!-- Explain the purpose of this change. Link to any related issues. -->

Fixes #<!-- issue number -->

### Type of Change

<!-- Check all that apply -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 📓 New notebook
- [ ] 💻 New project or project enhancement
- [ ] 📝 Documentation update
- [ ] 📚 New learning resource
- [ ] 🧪 Test addition or improvement
- [ ] 🏗️ Infrastructure / CI/CD / Docker
- [ ] 🔧 Refactoring (no functional changes)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)

---

## Checklist

### General

- [ ] My code follows the [code style guide](CONTRIBUTING.md#code-style-guide) of this project
- [ ] I have performed a self-review of my own changes
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings or errors

### For Notebooks

- [ ] The notebook runs **top-to-bottom** without errors
- [ ] The notebook follows the [10-section structure](CONTRIBUTING.md#notebook-template)
- [ ] All random seeds are set for reproducibility
- [ ] Only free, open-source tools are used (no paid API keys)
- [ ] Real datasets are used (no fake/placeholder data)
- [ ] Every code cell has a preceding markdown explanation

### For Projects

- [ ] All tests pass: `pytest projects/<project-name>/tests/`
- [ ] `requirements.txt` has pinned dependency versions
- [ ] README.md includes setup, usage, and architecture documentation
- [ ] Type hints are used on all function signatures
- [ ] Docstrings are present on all public functions (Google style)

### For Documentation

- [ ] I have updated relevant documentation
- [ ] MkDocs builds cleanly: `mkdocs build --strict`
- [ ] Links are valid and not broken

---

## Screenshots / Output

<!-- If applicable, add screenshots, plots, or output snippets to demonstrate the changes. -->

---

## Additional Notes

<!-- Any additional context for reviewers. -->
