from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


SUMMARY_SYSTEM = (
    "You are an expert technical recruiter. Summarize the candidate vs JD. "
    "Return: strengths (bullets), gaps (bullets), skill_match_percent (0-100)"
    "Be concise."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM),
        (
            "human",
            "Job description:\n{jd}\n\nCandidate context (top matches):\n{context}\n\nReturn structured summary.",
        ),
    ]
)


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([f"Source: {d.metadata.get('source','')}\n{d.page_content[:1200]}" for d in docs])


def build_retrieval_qa_chain(retriever, llm):
    context_chain = RunnableLambda(lambda q: retriever.get_relevant_documents(q)) | format_docs
    chain = RunnableParallel({"jd": RunnablePassthrough(), "context": context_chain}) | PROMPT | llm
    return chain


