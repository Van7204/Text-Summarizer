from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from flask_cors import CORS

# Download necessary nltk data
nltk.download('punkt')
nltk.download('stopwords')


app = Flask(__name__)

# Load models
abstractive_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
abstractive_tokenizer = AutoTokenizer.from_pretrained("t5-base")
abstractive_summarizer = pipeline(
    "summarization",
    model=abstractive_model,
    tokenizer=abstractive_tokenizer,
    framework="pt"
)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper Functions
def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

def generate_summary_embeddings(sentences, embeddings, num_sentences=3):
    sentence_embeddings = np.array(embeddings)
    similarity_matrix = cosine_similarity(sentence_embeddings)
    sentence_scores = np.sum(similarity_matrix, axis=1)
    ranked_sentences_with_scores = sorted(((score, index) for index, score in enumerate(sentence_scores)), reverse=True)
    summary_sentences = [sentences[index] for _, index in ranked_sentences_with_scores[:num_sentences]]
    summary = ' '.join(summary_sentences)
    return summary

def fun_embeddings(text):
    sentences = preprocess_text(text)
    embeddings = embedding_model.encode(sentences)
    summary = generate_summary_embeddings(sentences, embeddings)
    return summary

# Routes
@app.route('/extractive', methods=['POST'])
def extractive():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    summary = fun_embeddings(text)
    print(summary)
    return jsonify({'summary': summary})

@app.route('/abstractive', methods=['POST'])
def abstractive():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    # Adjusting max_length dynamically based on input text length
    input_length = len(text.split())  # Number of words in input text

    summary = abstractive_summarizer(text, max_length=500, min_length=100, do_sample=False)
    print(summary)
    return jsonify({'summary': summary[0]['summary_text']})

if __name__ == '__main__':
    CORS(app)
    app.run(host='0.0.0.0', port=7000)
