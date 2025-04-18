import streamlit as st
import requests
import uuid
import time

BASE_URL = "http://localhost:8000"

def main():
    st.title("Multi-Agent Interview")
    st.markdown("""
    - 🔵 Question Generator Agent
    - 🟡 Interviewer Agent
    - 🔴 Feedback Analyst Agent
    """)

    if "session_id" not in st.session_state:
        # Start new interview
        with st.form("start_interview"):
            topic = st.text_input("Interview Topic", "Data Science")
            model = st.selectbox("LLM Model", ["gemini", "gemma", "deepseek"])
            max_q = st.slider("Number of Questions", 3, 10, 5)

            if st.form_submit_button("Start Interview"):
                try:
                    response = requests.post(
                        f"{BASE_URL}/interview/start?topic={topic}&model={model}&max_questions={max_q}"
                    )
                    response.raise_for_status()
                    data = response.json()
                    st.session_state.session_id = data.get("session_id")
                    st.session_state.messages = [
                        {"role": "assistant", "content": data.get("content")}
                    ]
                    st.session_state.agent_logs = data.get("agent_logs", [])
                    st.session_state.completed = data.get("completed", False)

                    if not st.session_state.session_id:
                        st.error("Failed to start interview. Session ID not received.")

                except requests.exceptions.RequestException as e:
                    try:
                        error_data = e.response.json()
                        st.error(f"Failed to start interview. Backend error: {error_data}")
                    except:
                        st.error(f"Failed to start interview. An unexpected error occurred: {e}")

    if "session_id" in st.session_state:
        # Display chat
        st.header("Interview Conversation")
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Display agent logs
        with st.expander("Agent Activity Log"):
            for log in st.session_state.agent_logs:
                st.markdown(f"```\n{log}\n```")

        # Continue interview if not completed
        if not st.session_state.completed:
            with st.form("answer_form"):
                answer = st.text_area("Your Answer")
                if st.form_submit_button("Submit"):
                    st.session_state.messages.append({"role": "user", "content": answer})

                    response = requests.post(
                        f"{BASE_URL}/interview/next?session_id={st.session_state.session_id}&answer={answer}"
                    ).json()
                    print(f"DEBUG: Backend Response (next): {response}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.get("content")}
                    )
                    st.session_state.agent_logs.extend(response.get("agent_logs", []))
                    st.session_state.completed = response.get("completed", False)
        else:
            st.subheader("Interview Completed!")
            st.markdown(f"**Feedback:**\n{st.session_state.messages[-1]['content']}")

if __name__ == "__main__":
    main()