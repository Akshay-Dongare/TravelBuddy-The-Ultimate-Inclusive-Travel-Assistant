# TO RUN: t run .\streamlit_ollama_chatbot.py
import ollama
import streamlit as st

#Hyperparameters

st.session_state["model"] = "TravelBuddy:latest"

st.title("Travel Buddy")

from streamlit_extras.stylable_container import stylable_container

st.title("Audio Recorder")
with stylable_container(
        key="bottom_content",
        css_styles="""
            {
                position: fixed;
                bottom: 120px;
            }
            """,
    ):
        audio = audiorecorder("🎙️ start", "🎙️ stop")
        print('audio: ', audio)
        if len(audio) > 0:
            audio.export("audio.mp3", format="mp3")

st.chat_input("These are words.")

with stylable_container(
        key="text_input1",
        css_styles="""
            {
                position: fixed;
                bottom: 200px;
            }
            """,
    ):
    st.text_input(label = 'text' ,value = "These are words.")

# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# init models
if "model" not in st.session_state:
    st.session_state["model"] = ""

# No need to show model list to user
#models = [model["name"] for model in ollama.list()["models"]]
#st.session_state["model"] = st.selectbox("Choose your model", models)

def model_res_generator():
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("TravelBuddy is ready to answer all your travel queries!"):
    # add latest message to history in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant", "content": message})
