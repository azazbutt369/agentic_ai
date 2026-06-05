# Security Policy

## Supported Versions

The AI Engineer Roadmap is an educational and open-source project.

The following versions are currently supported with security updates and maintenance.

| Version            | Supported |
| ------------------ | --------- |
| Latest Main Branch | ✅         |
| Archived Releases  | ❌         |

Always use the latest version of the repository.

---

# Reporting a Vulnerability

If you discover a security vulnerability, please do not create a public issue immediately.

Instead:

1. Contact the maintainers privately.
2. Provide a detailed description.
3. Include reproduction steps if possible.
4. Include affected files or components.

Examples of security issues:

* Exposed API keys
* Vulnerable dependencies
* Unsafe Docker configurations
* Prompt injection vulnerabilities
* Agent security flaws
* Authentication weaknesses
* Sensitive data exposure
* Supply chain vulnerabilities

---

# What To Include

When reporting a vulnerability, please provide:

## Summary

A brief description of the issue.

---

## Impact

Explain:

* What can be exploited?
* Who is affected?
* What is the potential damage?

---

## Reproduction Steps

Provide:

1. Environment details
2. Commands executed
3. Expected behavior
4. Actual behavior

---

## Proof of Concept

If available:

* Code snippets
* Screenshots
* Logs
* Demonstrations

Avoid publishing sensitive information publicly.

---

# Response Process

Maintainers will:

1. Acknowledge receipt of the report.
2. Investigate the issue.
3. Determine severity.
4. Develop a fix if required.
5. Publish remediation guidance.

---

# Responsible Disclosure

Please allow maintainers reasonable time to investigate and address issues before publicly disclosing vulnerabilities.

Responsible disclosure helps protect:

* Contributors
* Learners
* Organizations using repository code
* Community projects built from roadmap examples

---

# Scope

This policy applies to:

* Example code
* Notebooks
* Docker configurations
* CI/CD workflows
* Deployment templates
* Documentation examples
* AI agent implementations
* RAG systems
* Infrastructure manifests

---

# Out of Scope

The following are generally out of scope:

* Theoretical vulnerabilities without reproduction
* Vulnerabilities in third-party services
* Issues already publicly documented
* Educational examples intentionally demonstrating attacks or defenses

---

# Security Best Practices

Contributors should never commit:

* API keys
* Tokens
* Passwords
* Private certificates
* Personal data
* Proprietary datasets

Use:

* Environment variables
* Secret managers
* Configuration templates

instead of hardcoded credentials.

---

# AI-Specific Security Concerns

Contributors should be aware of:

## Prompt Injection

Validate external inputs before passing them to LLMs.

---

## Data Leakage

Avoid exposing confidential or personally identifiable information.

---

## Tool Abuse

Restrict tool permissions where appropriate.

---

## Model Security

Verify model sources and dependencies.

---

## Supply Chain Risks

Review third-party packages before introducing them into the repository.

---

# Acknowledgements

We appreciate responsible disclosure and security-focused contributions that help improve the safety and reliability of this project.

Thank you for helping keep the AI Engineer Roadmap secure.
