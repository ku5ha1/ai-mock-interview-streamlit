from langgraph.graph import MessageGraph
from langchain_core.prompts import ChatPromptTemplate
from .state import InterviewState, AgentResponse
from dotenv import load_dotenv
import re
load_dotenv()

def create_question_generator_agent(llm):
    async def question_generator(state: InterviewState):
        prompt = ChatPromptTemplate.from_template(
            """You are an expert interviewer for the topic: {topic}.
            Generate {max_questions} interview questions that would thoroughly test a candidate's
            knowledge on this topic.

            Return ONLY a numbered list of questions, with each question on a new line.
            """
        )

        formatted = prompt.format_messages(
            topic=state["topic"],
            max_questions=state["max_questions"]
        )

        response = await llm.invoke(formatted)

        # Split the response (which is now a string) into individual questions
        questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
        state["questions"] = questions  # Store questions in the InterviewState

        log_msg = (
            f"🔵 Question Generator Agent:\n"
            f"INPUT: Topic: {state['topic']}, Max Questions: {state['max_questions']}\n"
            f"OUTPUT: Generated and stored {len(questions)} questions."
        )

        return AgentResponse(
            content=questions[0] if questions else None, # Return the first question
            log_message=log_msg
        )

    return question_generator