from langgraph.graph import MessageGraph
from langchain_core.prompts import ChatPromptTemplate
from .state import InterviewState, AgentResponse

def create_feedback_analyst_agent(llm):
    async def feedback_analyst(state: InterviewState):
        transcript_parts = []
        num_interactions = min(len(state['questions']), len(state['answers']))
        for i in range(num_interactions):
            transcript_parts.append(f"Q: {state['questions'][i]}\nA: {state['answers'][i]}")
        transcript = "\n".join(transcript_parts)

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful feedback analyst. Based on the following interview transcript,
            provide constructive feedback to the candidate. Highlight strengths and areas for improvement.

            Transcript:
            {transcript}
            """
        )

        formatted = prompt.format_messages(
            transcript=transcript
        )

        response = await llm.invoke(formatted)

        log_msg = (
            f"🔴 Feedback Analyst Agent:\n"
            f"INPUT: Interview Transcript\n"
            f"OUTPUT: Feedback provided:\n{response}"  # Include the raw response in the log
        )

        return AgentResponse(
            content=response,  # Use the raw response as content
            log_message=log_msg,
        )

    return feedback_analyst