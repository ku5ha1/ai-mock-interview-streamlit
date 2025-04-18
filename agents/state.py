from typing import Any, TypedDict, List, Dict, Optional
from dataclasses import dataclass

class InterviewState(TypedDict):
    topic: str
    questions: List[str]
    answers: List[str]
    transcript: List[Dict[str, str]]
    current_question_index: int
    max_questions: int
    feedback: Optional[str]
    llm: Any  # LLM client
    agent_logs: List[str]  # For tracking agent communications

@dataclass
class AgentResponse:
    content: Any
    log_message: str