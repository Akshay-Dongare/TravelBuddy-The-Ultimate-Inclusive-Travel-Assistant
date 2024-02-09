from datasets import load_dataset

data = load_dataset('squad', split='train')

#############################################################################################################

data.drop_duplicates(subset='context', keep='first', inplace=True)

#############################################################################################################

import os
from getpass import getpass
from langchain.embeddings.openai import OpenAIEmbeddings

# get API key from top-right dropdown on OpenAI website
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

#############################################################################################################

from pinecone import Pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")

# configure client
pc = Pinecone(api_key=api_key)

#############################################################################################################

from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws", region="us-west-2"
)

#############################################################################################################

import time

index_name = "langchain-retrieval-agent"
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of ada 002
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

#############################################################################################################

from tqdm.auto import tqdm

batch_size = 100

texts = []
metadatas = []

for i in tqdm(range(0, len(data), batch_size)):
    # get end of batch
    i_end = min(len(data), i+batch_size)
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

#############################################################################################################

print(index.describe_index_stats())

#############################################################################################################
    
from langchain.vectorstores import Pinecone

text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)


query = "when was the college of engineering in the University of Notre Dame established?"

vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

#############################################################################################################

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
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
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

#############################################################################################################

qa.run(query)

#############################################################################################################

from langchain.agents import Tool

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
]

#############################################################################################################

from langchain.agents import initialize_agent

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

#############################################################################################################

agent(query)

#############################################################################################################

agent("what is 2 * 7?")

#############################################################################################################

agent("can you tell me some facts about the University of Notre Dame?") 

#############################################################################################################

agent("can you summarize these facts in two short sentences")

#############################################################################################################

# pc.delete_index(index_name) to save resources

#############################################################################################################