# command to launc this app - flask run -h localhost -p 6000
from flask import Flask, Blueprint, render_template, request, url_for, redirect, jsonify, flash,make_response
import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import os
from sklearn.neighbors import NearestNeighbors
app = Flask(__name__)
PORT = 6000

openai_api_key = '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'

def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[{idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('./models/universal-sentence-encoder_4',)
        self.fitted = False
    
    
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings



def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'


def generate_text(openAI_key,prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_answer(question,openAI_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
              "Make sure the answer is correct and don't output false content. "\
              "If the text does not relate to the query, simply state 'Sorry no relevant answer found'. Ignore outlier ."\
              "Do not use double quotes to quote the answer"\
              "search results which has nothing to do with the question. Only answer what is asked. The "\
              "answer should be short and concise.\n\nQuery: {question}\nAnswer: "
    
    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(openAI_key, prompt,"text-davinci-003")
    return answer


def question_answer(url, file, question,openAI_key):
    if openAI_key.strip()=='':
        return make_response(jsonify({'message':'Please enter you Open AI Key. Get your key here : https://platform.openai.com/account/api-keys'}),400)
    if url.strip() == '' and file == None:
        return make_response(jsonify({'message':'Both URL and PDF is empty. Provide atleast one'}),400)
    if url.strip() != '' and file != None:
        return make_response(jsonify({'message':'Both URL and PDF is provided. Please provide only one (eiter URL or PDF)'}),400)

    # if url.strip() != '':
    #     glob_url = url
    #     download_pdf(glob_url, 'corpus.pdf')
    #     load_recommender('corpus.pdf')

    else:
        file_name = file.name
        load_recommender(file_name)

    if question.strip() == '':
        return make_response(jsonify({'message':'Question field is empty'}),400)
    
    try:
        answer = generate_answer(question,openAI_key)
        print(answer)
        return make_response(jsonify({'message':answer}),200)
    except Exception as e:
        print(str(e))
        return make_response(jsonify({'message':'Some internal connection error occured. Please try this later.'}),500)

recommender = SemanticSearch()

@app.route('/quickpeek-into-pdf', methods=['POST'])
def quickpeek_pdf():
    payload = request.json
    question = payload['question']
    filepath = '../documents/'+payload['category']+'/'+payload['subCategory']+'/'+payload['documentName']
    print(payload,filepath)
    file = open(filepath,'r')
    return question_answer('',file,question or '',openai_api_key)

if __name__ == "__main__":
    app.run(debug=True)