from flask import Flask, render_template, request
import openai
import config


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return response_by_llm(input)

def response_by_llm(query):
    #Q&A using gpt3.5
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
    {"role": "user", 
    "content":f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with [Invalid Question]. Now, I will answer this question:{query} to the best of my knowledge."}
    ]
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    app.run(debug=True)
