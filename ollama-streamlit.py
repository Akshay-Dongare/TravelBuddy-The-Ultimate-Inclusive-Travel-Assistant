# TO RUN: t run .\streamlit_ollama_chatbot.py
import ollama
import streamlit as st
import speech_recognition as sr
import pyttsx3

#Hyperparameters
st.session_state["model"] = "TravelBuddy:latest"
st.set_page_config(layout='wide')

st.title("Travel Buddy: The Ultimate Inclusive Travel Assistant")
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

# Adding voice functionality
if 'button' not in st.session_state:
    st.session_state.button = False

def voice():
    # Toggle button on click
    st.session_state.button = not st.session_state.button


    # Initialize the recognizer 
    r = sr.Recognizer() 

    # Function to convert text to
    # speech
    def SpeakText(command):
        
        # Initialize the engine
        engine = pyttsx3.init("sapi5", True) #driver for windows and debug mode as args
        engine.say(command) 
        engine.runAndWait()

    # Loop infinitely for user to
    # speak
    
    while(st.session_state.button == True):
        
        # Exception handling to handle
        # exceptions at the runtime
        try:
            
            # use the microphone as source for input.
            with sr.Microphone() as source2:
                
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level 
                r.adjust_for_ambient_noise(source2, duration=0.2)
                
                #listens for the user's input 
                audio2 = r.listen(source2)
                
                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
    
                print("Did you say ",MyText)

                # add latest message to history in format {role, content}
                st.session_state["messages"].append({"role": "user", "content": MyText})

                with st.chat_message("user"):
                    st.markdown(MyText)

                with st.chat_message("assistant"):
                    # Without Threading
                    SpeakText("Please give me a minute to get back to you!")
                    message = st.write_stream(model_res_generator())
                    SpeakText(message)
                        
                    st.session_state["messages"].append({"role": "assistant", "content": message})
                    # Toggle Button
                    st.session_state.button = not st.session_state.button
                
                
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            print("unknown error occurred")

st.button('Mic', on_click=voice)
