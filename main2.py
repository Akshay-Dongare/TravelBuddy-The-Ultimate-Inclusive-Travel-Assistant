from flask import Flask, render_template, request
import config
import os
from openai import OpenAI


app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("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"),
)

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
    completion = client.chat.completions.create(
        messages=[
            {"role": "user", 
             "content":f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with [Invalid Question]. Now, I will answer this question:{query} to the best of my knowledge."
             }
             ],
    model="gpt-3.5-turbo",
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    app.run(debug=True)
