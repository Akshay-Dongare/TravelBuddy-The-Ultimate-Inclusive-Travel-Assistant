from flask import Flask, render_template, request
import openai
import config
from pinecone import Pinecone
#import pinecone_datasets
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
#from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain #RAG With sources
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from pinecone import ServerlessSpec, PodSpec
import time
from datasets import load_dataset
from tqdm.auto import tqdm
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain_community.tools.tavily_search import TavilySearchResults


app = Flask(__name__)
openai.api_key = config.OPENAI_API_KEY
'''
openai.api_key = config.OPENAI_API_KEY
#cohere initialization
#co = cohere.Client(config.COHERE_API_KEY)

# initialize connection to pinecone
# initialize connection to pinecone (get API key at app.pinecone.io)
# configure client
pinecone = Pinecone(api_key=config.PINECONE_API_KEY)


#dataset = pinecone_datasets.load_dataset('wikipedia-simple-text-embedding-ada-002-100K') #change to custom travel dataset later

#load Stanford Question-Answering Dataset (SQuAD) dataset
data = load_dataset('squad', split='train')

#pre-process dataset
data = data.to_pandas()
data.drop_duplicates(subset='context', keep='first', inplace=True)

#openAI embedding model
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=config.OPENAI_API_KEY
)


##########################################################
use_serverless = False #Pinecone serverless is paid but pod-based architecture is in free tier

if use_serverless:
    spec = ServerlessSpec(cloud='aws', region='us-west-2')
else:
    # if not using a starter index, you should specify a pod_type too
    spec = PodSpec(environment=config.PINECONE_ENVIRONMENT)

index_name = "langchain-retrieval-agent"
existing_indexes = [
    index_info["name"] for index_info in pinecone.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,  # dimensionality of ada 002
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pinecone.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()
#####################################################################################

#upserting to vectordb in batches
batch_size = 100

texts = []
metadatas = []

for i in tqdm(range(0, 5, batch_size)):#change 5 to len(data)
    # get end of batch
    i_end = min(5, i+batch_size) #change 5 to len(data)
    batch = data.iloc[i:i_end]
    # first get metadata fields for this record
    metadatas = [{
        'title': record['title'],
        'text': record['context']
    } for j, record in batch.iterrows()]
    # get the list of contexts / documents
    documents = batch['context']
    # create document embeddings
    embeds = embed.embed_documents(documents)
    # get IDs
    ids = batch['id']
    # add everything to pinecone
    index.upsert(vectors=zip(ids, embeds, metadatas))

index.describe_index_stats()


######################################################################################
#creating a vectorstore
text_field = "text"  # the metadata field that contains our text

# switch back to normal index for langchain (DONT KNOW IF THIS IS REQD)
#index = pinecone.Index(index_name)

# initialize the vector store object
vectorstore = Pinecone(
    index, embed , text_field
)

# Now we can query the vector store directly using vectorstore.similarity_search:   
#query = "who was Benito Mussolini?"
#vectorstore.similarity_search(
#    query,  # our search query
#    k=3  # return 3 most relevant docs
#)
'''
#But lets use it to perform RAG Query Answering
#Initializing the Conversational Agent
# completion llm

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=config.OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
#qa = RetrievalQA.from_chain_type(
#    llm=llm,
#    chain_type="stuff",
#    retriever=vectorstore.as_retriever()
#)
'''
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Knowledge Base',
        func=qa_with_sources.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
]
'''
import os
os.environ["tavily_api_key"] = "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"

tools = [TavilySearchResults(max_results=1)]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return agent(input)['output']


if __name__ == '__main__':
    app.run(debug=True)
