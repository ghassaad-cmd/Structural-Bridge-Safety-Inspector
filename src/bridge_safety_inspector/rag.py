from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_TOKEN_ALIASES = {
    "inspect": "inspection",
    "inspected": "inspection",
    "inspecting": "inspection",
    "monitor": "monitoring",
    "monitored": "monitoring",
    "documented": "document",
    "reviewed": "review",
}
_STOP_WORDS = {
    "a",
    "about",
    "and",
    "are",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "s",
    "should",
    "tell",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}

NO_INFORMATION_ANSWER = (
    "I could not find relevant information in the available knowledge base. "
    "No conclusion can be made from these sources."
)


def _tokens(text: str) -> set[str]:
    tokens = set()
    for token in _TOKEN_PATTERN.findall(text.lower()):
        if token in _STOP_WORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(_TOKEN_ALIASES.get(token, token))
    return tokens


@dataclass(frozen=True)
class KnowledgeEntry:
    title: str
    content: str
    tags: tuple[str, ...] = ()
    source_name: str = "Unspecified source"
    source_url: str = ""


@dataclass(frozen=True)
class KnowledgeSource:
    title: str
    url: str


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    sources: tuple[KnowledgeSource, ...]


class BridgeKnowledgeBase:
    """Small in-memory retriever and extractive answerer for sourced guidance."""

    def __init__(self, entries: Iterable[KnowledgeEntry] | None = None) -> None:
        self.entries = list(entries or [])

    def add_entry(self, entry: KnowledgeEntry) -> None:
        self.entries.append(entry)

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeEntry]:
        if top_k <= 0:
            return []

        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        scored: list[tuple[int, int, KnowledgeEntry]] = []
        for index, entry in enumerate(self.entries):
            entry_tokens = _tokens(
                f"{entry.title} {' '.join(entry.tags)} {entry.content}"
            )
            score = len(query_tokens & entry_tokens)
            if score >= min(2, len(query_tokens)):
                scored.append((score, index, entry))

        scored.sort(key=lambda result: (-result[0], result[1]))
        return [entry for _, _, entry in scored[:top_k]]

    def answer(self, query: str, top_k: int = 3) -> RAGAnswer:
        """Return a short answer composed only from retrieved entries."""
        results = [
            entry
            for entry in self.search(query, top_k=top_k)
            if entry.source_name != "Unspecified source"
            and entry.source_url.startswith(("https://", "http://"))
        ]
        if not results:
            return RAGAnswer(
                question=query,
                answer=NO_INFORMATION_ANSWER,
                sources=(),
            )

        answer_lines = [
            "The available sources say:",
            *(
                f"- {entry.content} "
                f"[Source: {entry.source_name}]({entry.source_url})"
                for entry in results
            ),
            "",
            "This is general reference information, not certified structural "
            "engineering advice or a bridge-safety determination.",
        ]
        sources = tuple(
            KnowledgeSource(title=entry.source_name, url=entry.source_url)
            for entry in results
        )
        return RAGAnswer(
            question=query,
            answer="\n".join(answer_lines),
            sources=sources,
        )


FHWA_BIRM_URL = "https://www.fhwa.dot.gov/bridge/nbis/pubs/nhi21002.pdf"
FHWA_INSPECTION_QA_URL = "https://www.fhwa.dot.gov/bridge/nbis/nbisframework.cfm"
FHWA_PRESERVATION_GUIDE_URL = (
    "https://www.fhwa.dot.gov/bridge/preservation/guide/guide.pdf"
)

DEFAULT_BRIDGE_KB = BridgeKnowledgeBase(
    entries=[
        KnowledgeEntry(
            title="Concrete cracks: inspection reference",
            content=(
                "FHWA's Bridge Inspector's Reference Manual is a reference for "
                "bridge inspection. Record descriptive observations such as a "
                "visible crack's location, orientation or pattern, and apparent "
                "extent under the applicable agency inspection procedures. "
                "Interpretation requires qualified review; this knowledge base "
                "supplies no crack-width threshold or safety conclusion."
            ),
            tags=(
                "concrete",
                "crack",
                "cracking",
                "characteristics",
                "visual",
                "inspect",
                "inspection",
                "defect",
            ),
            source_name="FHWA Bridge Inspector's Reference Manual",
            source_url=FHWA_BIRM_URL,
        ),
        KnowledgeEntry(
            title="Crack inspection and monitoring records",
            content=(
                "FHWA's recommended bridge inspection QC/QA framework calls for "
                "review and validation of inspection reports and data, procedures "
                "to identify and resolve errors or omissions, and tracking "
                "follow-up actions. Keep observations consistent and refer "
                "changes in cracking to the responsible qualified inspector."
            ),
            tags=(
                "crack",
                "monitoring",
                "track",
                "records",
                "inspection",
                "report",
                "documentation",
                "follow-up",
                "quality",
                "review",
            ),
            source_name="FHWA Recommended Framework for a Bridge Inspection QC/QA Program",
            source_url=FHWA_INSPECTION_QA_URL,
        ),
        KnowledgeEntry(
            title="Bridge inspection procedures and quality",
            content=(
                "FHWA's recommended framework describes documenting an "
                "inspection manual and QC/QA procedures, defining inspection "
                "roles, reviewing field work and reports, and documenting "
                "review results and follow-up actions. Apply the procedures "
                "adopted by the responsible bridge owner."
            ),
            tags=(
                "bridge",
                "inspection",
                "procedure",
                "procedures",
                "manual",
                "quality",
                "control",
                "assurance",
                "review",
                "report",
                "field",
                "team",
            ),
            source_name="FHWA Recommended Framework for a Bridge Inspection QC/QA Program",
            source_url=FHWA_INSPECTION_QA_URL,
        ),
        KnowledgeEntry(
            title="Maintenance and preservation planning",
            content=(
                "FHWA's Bridge Preservation Guide is a public reference for "
                "bridge preservation and maintenance planning. Select any "
                "maintenance or repair approach based on documented inspection "
                "findings and review by the responsible qualified professionals; "
                "this knowledge base does not prescribe a repair design, "
                "materials, or project-specific maintenance record."
            ),
            tags=(
                "maintenance",
                "repair",
                "repairs",
                "preservation",
                "approach",
                "planning",
                "bridge",
                "concrete",
                "rehabilitation",
            ),
            source_name="FHWA Bridge Preservation Guide",
            source_url=FHWA_PRESERVATION_GUIDE_URL,
        ),
    ]
)
