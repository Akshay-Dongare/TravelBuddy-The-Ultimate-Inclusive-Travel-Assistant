from flask import Flask, render_template, request
import Utils.config as config
import os
from openai import OpenAI
import speech_recognition as sr
import pyttsx3 

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get(config.OPENAI_API_KEY),
)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/voice',methods=["GET", "POST"])
def voice():
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
    
    while(1):    
        
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
                
                #Q&A using gpt3.5
                completion = client.chat.completions.create(
                    messages=[
                        {"role": "user", 
                        "content":f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with [Invalid Question]. Now, I will answer this question:{MyText} to the best of my knowledge."
                        }
                        ],
                model="gpt-3.5-turbo",
                )
                SpeakText(completion.choices[0].message.content)
                
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            print("unknown error occurred")

def response_by_llm(query):
    #Q&A using gpt3.5
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", 
             "content":f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with [Invalid Question]. Now, I will answer this question:{query} to the best of my knowledge."
             }
             ],
    model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return response_by_llm(input)

if __name__ == '__main__':
    app.run(debug=True)
