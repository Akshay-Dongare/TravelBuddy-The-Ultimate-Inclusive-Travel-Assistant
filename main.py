from flask import Flask, render_template, request
import openai
import config
import pinecone
import pinecone_datasets
from langchain.chat_models import ChatOpenAI
#from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain #RAG With sources


app = Flask(__name__)

openai.api_key = config.OPENAI_API_KEY
#cohere initialization
#co = cohere.Client(config.COHERE_API_KEY)

# initialize connection to pinecone
pinecone.init(
    api_key=config.PINECONE_API_KEY,
    environment=config.PINECONE_ENVIRONMENT 
)

dataset = pinecone_datasets.load_dataset('wikipedia-simple-text-embedding-ada-002-100K')


# completion llm
llm = ChatOpenAI(
    openai_api_key= config.OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

#qa = RetrievalQA.from_chain_type(
#    llm=llm,
#    chain_type="stuff",
#    retriever=vectorstore.as_retriever()
#)

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

qa_with_sources(query)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


if __name__ == '__main__':
    app.run(debug=True)
