from langgraph.graph import MessageGraph
from langchain_core.prompts import ChatPromptTemplate
from .state import InterviewState, AgentResponse

def create_interviewer_agent(llm):
    async def interviewer(state: InterviewState):
        if state["current_question_index"] >= len(state["questions"]):
            return AgentResponse(
                content=None,
                log_message="🟢 Interviewer Agent: No more questions to ask"
            )

        current_question = state["questions"][state["current_question_index"]]

        log_msg = (
            f"🟡 Interviewer Agent:\n"
            f"Current Question: {current_question}\n"
            f"Progress: {state['current_question_index']+1}/{len(state['questions'])}"
        )

        return AgentResponse(
            content=current_question,
            log_message=log_msg
        )

    return interviewer