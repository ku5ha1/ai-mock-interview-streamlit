import streamlit as st
import requests
import uuid
import time
from streamlit import components

BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(page_title="Interview Assistant", page_icon="🎤", layout="wide")
    st.title("Interview Assistant 🎤")
    st.caption("Gemini-powered multi-agent interview with questions and feedback")

    # Sidebar: session controls
    with st.sidebar:
        st.header("Setup")
        topic = st.text_input("Interview Topic", "Data Science")
        max_q = st.slider("Number of Questions", 3, 10, 5)
        start_btn = st.button("Start Interview", type="primary", use_container_width=True, disabled="session_id" in st.session_state)
        st.markdown("---")
        st.caption("Agents: 🔵 Question Generator · 🟡 Interviewer · 🔴 Feedback Analyst")

    # Initialize session state containers
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = []
    if "completed" not in st.session_state:
        st.session_state.completed = False

    # Start flow (model is implicitly Gemini; backend ignores model param anyway)
    if start_btn and "session_id" not in st.session_state:
        with st.spinner("Starting interview..."):
            try:
                # Pass model=gemini implicitly (backend signature still expects it)
                url = f"{BASE_URL}/interview/start?topic={topic}&model=gemini&max_questions={max_q}"
                resp = requests.post(url)
                resp.raise_for_status()
                data = resp.json()
                st.session_state.session_id = data.get("session_id")
                st.session_state.messages = [{"role": "assistant", "content": data.get("content")}]
                st.session_state.agent_logs = data.get("agent_logs", [])
                st.session_state.completed = data.get("completed", False)
                st.success("Interview started.")
            except requests.exceptions.RequestException as e:
                try:
                    st.error(f"Failed to start: {e.response.json()}")
                except Exception:
                    st.error(f"Failed to start: {e}")

    # Main layout: chat left, logs right
    col_chat, col_logs = st.columns([3, 2])

    with col_chat:
        st.subheader("Conversation")
        chat_box = st.container()
        with chat_box:
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
            # Auto-scroll to latest chat message
            components.v1.html(
                """
                <script>
                setTimeout(() => {
                  const doc = window.parent.document;
                  const items = doc.querySelectorAll('[data-testid="stChatMessage"]');
                  if (items && items.length) {
                    items[items.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
                  }
                }, 50);
                </script>
                """,
                height=0,
            )
        if st.button("Jump to latest", use_container_width=True):
            components.v1.html(
                """
                <script>
                const doc = window.parent.document;
                const items = doc.querySelectorAll('[data-testid="stChatMessage"]');
                if (items && items.length) {
                  items[items.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
                }
                </script>
                """,
                height=0,
            )

        if "session_id" in st.session_state and not st.session_state.completed:
            if user_input := st.chat_input("Type your answer and press Enter"):
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.spinner("Sending answer..."):
                    try:
                        next_url = f"{BASE_URL}/interview/next?session_id={st.session_state.session_id}&answer={user_input}"
                        response = requests.post(next_url)
                        response.raise_for_status()
                        payload = response.json()
                        st.session_state.messages.append({"role": "assistant", "content": payload.get("content")})
                        st.session_state.agent_logs.extend(payload.get("agent_logs", []))
                        st.session_state.completed = payload.get("completed", False)
                    except requests.exceptions.RequestException as e:
                        try:
                            st.error(f"Failed to submit answer: {e.response.json()}")
                        except Exception:
                            st.error(f"Failed to submit answer: {e}")

        if "session_id" in st.session_state and st.session_state.completed:
            st.success("Interview Completed!")
            st.markdown("**Final Feedback**")
            st.write(st.session_state.messages[-1]["content"])

    with col_logs:
        st.subheader("Agent Activity Log")
        if st.session_state.agent_logs:
            for log in reversed(st.session_state.agent_logs[-50:]):
                with st.expander("View log", expanded=False):
                    st.code(log)
        else:
            st.caption("Logs will appear here as agents run.")

if __name__ == "__main__":
    main()