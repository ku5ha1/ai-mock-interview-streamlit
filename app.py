import os
from typing import Dict
import uuid
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from agents.question_generator import create_question_generator_agent
from agents.interviewer import create_interviewer_agent
from agents.feedback_analyst import create_feedback_analyst_agent
from agents.state import InterviewState
from model_clients import get_llm_client

load_dotenv()

app = FastAPI()
interview_sessions: Dict[str, Dict] = {}

logging.basicConfig(level=logging.INFO)

@app.post("/interview/start")
async def start_interview(topic: str, model: str, max_questions: int):
    logging.info(f"Received /interview/start with topic: '{topic}', model: '{model}', max_questions: {max_questions}")
    session_id = str(uuid.uuid4())
    # Always use Gemini regardless of provided model
    llm = get_llm_client("gemini")
    question_agent = create_question_generator_agent(llm)

    state = InterviewState(
        topic=topic,
        max_questions=max_questions,
        current_question_index=0,
        answers=[],
        feedback="",
        agent_logs=[]
    )

    first_question_response = await question_agent(state)
    interview_sessions[session_id] = {
        "state": state,
        "agents": {
            "interviewer": create_interviewer_agent(llm),
            "feedback_analyst": create_feedback_analyst_agent(llm)
        }
    }

    response_data = {
        "session_id": session_id,
        "content": first_question_response.content,
        "agent_logs": [first_question_response.log_message],
        "completed": False
    }
    logging.info(f"Returning response from /interview/start: {response_data}")
    return response_data

@app.post("/interview/next")
async def next_interview_step(session_id: str = Query(...), answer: str = Query(...)):
    logging.info(f"Received /interview/next for session: '{session_id}', answer: '{answer}'")
    if session_id not in interview_sessions:
        logging.error(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    session = interview_sessions[session_id]
    state = session["state"]
    interviewer = session["agents"]["interviewer"]
    feedback_analyst = session["agents"]["feedback_analyst"]

    state["answers"].append(answer)
    state["current_question_index"] += 1
    logging.info(f"Updated state: current_question_index={state['current_question_index']}, max_questions={state['max_questions']}")

    try:
        if state["current_question_index"] < state["max_questions"]:
            agent_response = await interviewer(state)
            state["agent_logs"].append(agent_response.log_message)
            content = agent_response.content
            completed = False
            logging.info(f"Interviewer response content: '{content}'")
        elif state["current_question_index"] == state["max_questions"]:
            # Await the feedback analyst call
            agent_response = await feedback_analyst(state)
            state["feedback"] = agent_response.content
            state["agent_logs"].append(agent_response.log_message)
            content = agent_response.content
            completed = True
            logging.info(f"Feedback analyst response content: '{content}'")
        else:
            content = "Interview completed." # Should ideally not reach here
            completed = True

        response_data = {
            "content": content,
            "agent_logs": state["agent_logs"],
            "completed": completed
        }
        logging.info(f"Returning response: {response_data}")
        return response_data
    except Exception as e:
        logging.error(f"Error processing /interview/next: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing next step: {e}")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)