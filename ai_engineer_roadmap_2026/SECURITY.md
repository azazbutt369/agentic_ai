<div align="center">

# 🔒 Security Policy

**We take the security of this project and its users seriously.**

</div>

---

## Supported Versions

| Version | Supported |
|:-------:|:---------:|
| `main` branch (latest) | ✅ |
| Previous releases | ❌ |

We only support the latest version on the `main` branch. If you find a vulnerability, please ensure you are testing against the latest commit.

---

## Reporting a Vulnerability

> ⚠️ **Please do NOT open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in this repository, we appreciate your responsible disclosure.

### How to Report

1. **GitHub Private Vulnerability Reporting (Preferred)**
   - Navigate to the **Security** tab of this repository
   - Click **"Report a vulnerability"**
   - Fill out the form with as much detail as possible

2. **Email**
   - If GitHub private reporting is unavailable, email the maintainers directly
   - Include `[SECURITY]` in the subject line

### What to Include

Please include the following in your report:

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Affected component** (notebook, project, script, configuration)
- **Potential impact** (what an attacker could achieve)
- **Suggested fix** (if you have one)

### Response Timeline

| Action | Timeframe |
|--------|-----------|
| Acknowledgment of report | Within **48 hours** |
| Initial assessment | Within **1 week** |
| Fix development & review | Within **2 weeks** |
| Public disclosure (after fix) | Within **30 days** |

We will keep you informed throughout the process and credit you in the fix (unless you prefer to remain anonymous).

---

## Security Considerations for This Repository

This repository contains **educational AI/ML code**. While it is designed for learning, the following security considerations apply:

### 🐳 Docker Configurations

- Docker files in `docker/` are configured for **local development and learning**
- Do **not** deploy these configurations to production without proper hardening
- Review exposed ports, volume mounts, and environment variables before deploying

### 🤖 LLM & AI Safety

- Notebooks in Phase 4 (Generative AI) demonstrate prompt engineering and agentic patterns
- Code execution tools in projects (e.g., AI Agent Framework) use **sandboxed environments**
- Always review LLM outputs before acting on them in any production context
- See [`notebooks/06-advanced-paradigms/03-ai-safety-security.ipynb`](notebooks/06-advanced-paradigms/03-ai-safety-security.ipynb) for AI safety best practices

### 🔑 API Keys & Secrets

- This repository is designed to work **without paid API keys** (Ollama, HuggingFace)
- **Never commit API keys, tokens, or secrets** to this repository
- If you fork this repo, ensure your `.gitignore` excludes sensitive files
- Use environment variables for any configuration that includes credentials

### 📦 Dependencies

- All dependencies are pinned in `requirements.txt` for reproducibility
- We regularly update dependencies to patch known vulnerabilities
- If you discover an outdated dependency with a known CVE, please report it

### 📓 Notebook Execution

- Notebooks may download datasets from the internet (HuggingFace, torchvision)
- Notebooks may install additional Python packages via `pip`
- Review notebook contents before executing, especially if you've received them from a third party
- Use virtual environments to isolate dependencies

---

## Security Best Practices for Contributors

When contributing to this repository, please follow these guidelines:

1. **No hardcoded secrets** — Use environment variables or `.env` files (gitignored)
2. **Pin dependencies** — Always pin exact versions in `requirements.txt`
3. **Sanitize inputs** — Validate and sanitize all user inputs in project code
4. **Sandbox execution** — Code execution tools must use sandboxed environments
5. **Review downloads** — Verify the source and integrity of downloaded models/datasets
6. **No executable binaries** — Do not commit compiled binaries or executables
7. **Use HTTPS** — All URLs in documentation and code should use HTTPS

---

## Acknowledgments

We gratefully acknowledge security researchers who help keep this project safe. Contributors who report valid security issues will be recognized in our [CHANGELOG.md](CHANGELOG.md) (unless they prefer anonymity).

---

<div align="center">

**[← Back to README](README.md)** · **[Contributing Guide →](CONTRIBUTING.md)**

</div>
