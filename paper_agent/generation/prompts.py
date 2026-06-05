"""
==============================================================================
Prompt Templates — Structured instructions for the LLM
==============================================================================

WHY PROMPT ENGINEERING MATTERS:
    A 1.5B parameter model is powerful but needs PRECISE instructions.
    Unlike GPT-4 which can infer intent from vague prompts, smaller models
    need explicit guardrails:

    1. ROLE DEFINITION: "You are an ML research tutor" anchors the model's
       behavior and tone.
    2. STRICT CONTEXT GROUNDING: "Use ONLY the provided context" prevents
       hallucination — the #1 risk with small models.
    3. REFUSAL INSTRUCTION: "If the answer is not in the context, say so"
       teaches the model to admit uncertainty instead of fabricating answers.
    4. CITATION INSTRUCTION: "Cite sources" produces verifiable responses.

DESIGN DECISIONS:
    - Three distinct templates for three use cases:
        * QA: Answer specific questions from paper context
        * SUMMARIZE: Generate a structured summary of a paper
        * COMPARE: Compare methods/approaches across papers
    - Each template uses LangChain's ChatPromptTemplate for proper
      system/human message formatting.
    - Variables are enclosed in {braces} for LangChain substitution.
==============================================================================
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


# ---------------------------------------------------------------------------
# System messages (define the agent's role and behavior)
# ---------------------------------------------------------------------------

QA_SYSTEM_MESSAGE = """You are an ML research tutor helping users understand machine learning papers. Your role is to provide clear, accurate, and educational answers.

STRICT RULES:
1. Answer ONLY using the provided context from the research papers.
2. If the answer is NOT in the context, say: "I don't have enough information from the papers to answer this question."
3. NEVER make up information or cite papers not in the context.
4. Always cite your sources using the format: [Source: filename, Page X]
5. Explain technical concepts in a way that is accessible but accurate.
6. Use bullet points and structured formatting when it improves clarity.
7. If a concept in the paper is complex, break it down step by step."""

SUMMARIZE_SYSTEM_MESSAGE = """You are an ML research tutor that creates clear, structured summaries of machine learning papers.

STRICT RULES:
1. Base your summary ONLY on the provided context.
2. Structure your summary with these sections:
   - **Paper Overview**: What is this paper about? (1-2 sentences)
   - **Key Contributions**: What are the main contributions? (bullet points)
   - **Methodology**: How does the approach work? (brief explanation)
   - **Key Results**: What were the main findings? (if available in context)
   - **Significance**: Why does this paper matter?
3. Always cite specific pages: [Source: filename, Page X]
4. If the context doesn't cover a section, note it as "Not available in provided context."
5. Keep the summary concise but informative."""

COMPARE_SYSTEM_MESSAGE = """You are an ML research tutor that compares and contrasts different machine learning approaches from research papers.

STRICT RULES:
1. Compare ONLY based on information in the provided context.
2. Structure your comparison with:
   - **Approaches Compared**: What methods are being compared?
   - **Key Differences**: How do they differ in design/methodology?
   - **Strengths & Weaknesses**: What are the pros/cons of each?
   - **Performance**: How do results compare? (if available)
   - **When to Use Each**: Practical guidance on choosing between them.
3. Always cite sources: [Source: filename, Page X]
4. Be objective — present both sides fairly.
5. If insufficient context to compare, say so explicitly."""


# ---------------------------------------------------------------------------
# Prompt templates (combine system message + user input + context)
# ---------------------------------------------------------------------------


def get_qa_prompt() -> ChatPromptTemplate:
    """
    Create the Question-Answering prompt template.

    Variables:
        - {context}: Retrieved chunks formatted with citations
        - {chat_history}: Previous conversation turns
        - {question}: The user's current question

    WHY chat_history is included:
        Follow-up questions like "What about its limitations?" need
        context from the previous exchange to resolve "its". Without
        chat history, the model would have no idea what "its" refers to.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(QA_SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template(
            "Context from research papers:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Question: {question}"
        ),
    ])


def get_summarize_prompt() -> ChatPromptTemplate:
    """
    Create the Paper Summarization prompt template.

    Variables:
        - {context}: Full paper chunks formatted with citations
        - {question}: Summarization instruction (e.g., "Summarize this paper")
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SUMMARIZE_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template(
            "Paper content:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Task: {question}"
        ),
    ])


def get_compare_prompt() -> ChatPromptTemplate:
    """
    Create the Method Comparison prompt template.

    Variables:
        - {context}: Chunks from multiple papers formatted with citations
        - {question}: Comparison instruction
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(COMPARE_SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template(
            "Context from research papers:\n"
            "---\n"
            "{context}\n"
            "---\n\n"
            "Comparison task: {question}"
        ),
    ])


# ---------------------------------------------------------------------------
# Prompt selector
# ---------------------------------------------------------------------------

PROMPT_MAP = {
    "qa": get_qa_prompt,
    "summarize": get_summarize_prompt,
    "compare": get_compare_prompt,
}


def get_prompt(mode: str = "qa") -> ChatPromptTemplate:
    """
    Get the appropriate prompt template by mode.

    Args:
        mode: One of "qa", "summarize", "compare".

    Returns:
        Configured ChatPromptTemplate.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode not in PROMPT_MAP:
        raise ValueError(
            f"Unknown prompt mode: '{mode}'. "
            f"Available modes: {list(PROMPT_MAP.keys())}"
        )

    return PROMPT_MAP[mode]()
