from flask import Flask, request, render_template
import openai
import config
import pinecone
from flaskext.markdown import Markdown
#import cohere #use cohere to create embeddings using a multilingual model


app = Flask(__name__)
Markdown(app)

openai.api_key = config.OPENAI_API_KEY

#cohere initialization
#co = cohere.Client(config.COHERE_API_KEY)

# initialize connection to pinecone
pinecone.init(
    api_key=config.PINECONE_API_KEY,
    environment=config.PINECONE_ENVIRONMENT 
)

#ner
#nlp = spacy.load("en_core_web_sm")


@app.route('/static/<path:filename>')
def serve_static(filename):
  return app.send_static_file(filename)

@app.route('/')
def search_form():
  available_indexes = pinecone.list_indexes()
  return render_template('search_form.html',available_indexes=available_indexes)

@app.route('/search')
def search():
    # Get the search query from the URL query string
    query = request.args.get('query')

    #query synonym expansion using gpt3.5
    #Below is an example of a synonym finder prompt for gpt model
    #f"I want you to act as a synonyms provider. I will tell you a search query, and you will reply to me with a list of synonym alternatives according to my prompt. Provide only 2 synonyms for my search query. You will only reply the words list, and nothing else. Words should exist. Do not write explanations. Strictly, do not write anything else than a comma seperated list of two words. My search query is:{query}?"
    qeury_expansion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "user", 
          "content":f"I want you to act as a synonyms provider. I will tell you a search query, and you will reply to me with a list of synonym alternatives according to my prompt. Provide only 2 synonyms for my search query. You will only reply the words list, and nothing else. Words should exist. Do not write explanations. Strictly, do not write anything else than a comma seperated list of two words. My search query is:{query}?"}
        ]
        )
    #expand search query by concatenating synonym list
    expanded_query = query + " " + qeury_expansion.choices[0].message.content
    
    #embed the expanded search query
    xq = openai.Embedding.create(input=expanded_query, engine="text-embedding-ada-002")['data'][0]['embedding']
    
    #use cohere's multilingual model if you want to search from corpus containing content in different languages
    #before using this for query, make sure that corpus is embedded using multilingual model as well
    
    #xq = co.embed(texts=list(query),model='multilingual-22-12')

    #connect to pinecone index
    index = pinecone.Index(pinecone.list_indexes()[0])

    #retrieve the top three matches
    res = index.query([xq], top_k=3, include_metadata=True)

    '''
    scores = []
    NER_texts = []

    for match in res['matches']:
      doc = nlp(match['metadata']['text'])
      NER_texts.append(displacy.render(doc, style="ent"))
      scores.append(f"{match['score']:.2f}")

    #send result object to HTML to display
    results = zip(scores,NER_texts)
    '''

    #concatenate top 3 search results to provide more context for gpt model to answer the query 
    #note that we do not use the expanded_query for Q&A
    top3_search_results_concatenated = res['matches'][0]['metadata']['text']+res['matches'][1]['metadata']['text']+res['matches'][2]['metadata']['text']

    #Q&A using gpt3.5
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user", 
       "content":f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with [Invalid Question]. Now, I will answer this question:{query} in 10 lines using only this document:{top3_search_results_concatenated}"}
    ]
    )

    gpt_result = completion.choices[0].message.content
    # Render the search results template, passing in the search query and results
    return render_template('search_results.html', query=query, results=results,gpt_result=gpt_result)

@app.route('/show_upload_page')
def show_upload_page():
  return render_template("upload.html")

@app.route('/google_picker')
def show_google_picker():
  return render_template("google_picker.html")


@app.route('/upload_from_drive',methods=['GET','POST'])
def upload_from_drive():
  txtfile = request.form.get("my_file_name") #storing file name
  file_obj = open(txtfile,"r",encoding='utf8')
  file_data = file_obj.read()
  txtparas = file_data.split('\n\n')

  #blankspace cannot be embedded so pre-processing is done
  if ("" in txtparas):
      txtparas.remove("")

  #create embeddings   
  res = openai.Embedding.create(
      input=txtparas, engine="text-embedding-ada-002"
  )
  #use cohere multilingual model to create embeddings if you want support for other languages

  # extract embeddings to a list
  embeds = [record['embedding'] for record in res['data']] #embeds is a list

  index_name = str(txtfile[:-4])
  if index_name not in pinecone.list_indexes():
    if pinecone.list_indexes() != []:
      pinecone.delete_index(pinecone.list_indexes()[0]) #deleting current index because free tier of pinecone allows creation of only one index at a time
    pinecone.create_index(index_name, dimension=len(embeds[0]))

  work_around = None #keep retrying until connection is made
  while work_around is None:
    try:
      # connect to index
      index = pinecone.Index(index_name)

      # upsert to Pinecone
      ids = [str(n) for n in range(1, len(embeds[0])+1)]
      meta = [{'text': para} for para in txtparas]
      to_upsert = zip(ids, embeds, meta)
      index.upsert(vectors=list(to_upsert))

      work_around = 'worked'
    except:
      pass

  return render_template('upserted_in_index.html')


@app.route('/upload',methods=['GET','POST'])
def upload():
  if 'txtfile' not in request.files:
    return 'No file uploaded.'

  txtfile = request.files['txtfile']
  if txtfile.filename == '':
    return render_template('no_file_selected.html')

  if txtfile and txtfile.filename.endswith('.txt'):
    txtfile.save(txtfile.filename)  
    file_obj = open(txtfile.filename,"r",encoding='utf8')
    file_data = file_obj.read()
    txtparas = file_data.split('\n\n')

    #blankspace cannot be embedded so pre-processing is done
    if ("" in txtparas):
        txtparas.remove("")

    #create embeddings   
    res = openai.Embedding.create(
        input=txtparas, engine="text-embedding-ada-002"
    )
    #use cohere multilingual model to create embeddings if you want support for other languages

    # extract embeddings to a list
    embeds = [record['embedding'] for record in res['data']] #embeds is a list

    index_name = str(txtfile.filename[:-4])
    if index_name not in pinecone.list_indexes():
      if pinecone.list_indexes() != []:
        pinecone.delete_index(pinecone.list_indexes()[0]) #deleting current index because free tier of pinecone allows creation of only one index at a time
      pinecone.create_index(index_name, dimension=len(embeds[0]))

    work_around = None #keep retrying until connection is made
    while work_around is None:
      try:
        # connect to index
        index = pinecone.Index(index_name)

        # upsert to Pinecone
        ids = [str(n) for n in range(1, len(embeds[0])+1)]
        meta = [{'text': para} for para in txtparas]
        to_upsert = zip(ids, embeds, meta)
        index.upsert(vectors=list(to_upsert))

        work_around = 'worked'
      except:
        pass

    return render_template('upserted_in_index.html')

  return render_template("invalid_file_type.html")

if __name__ == '__main__':
  app.run(debug=True,port=8080)
