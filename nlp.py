
# # USING TF-IDF Method


# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.probability import FreqDist
# import string
# from sklearn.feature_extraction.text import TfidfVectorizer

# stop_words = set(stopwords.words("english"))

# def preprocess_text(text, stop_words = stop_words):
    
#     sentences = sent_tokenize(text)

   
#     words = word_tokenize(text.lower())

   
#     filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]

#     return sentences, filtered_words

# def compute_word_frequency(filtered_words):
    
#     freq_dist = FreqDist(filtered_words)

#     max_freq = max(freq_dist.values())
#     for word in freq_dist.keys():
#         freq_dist[word] = (freq_dist[word] / max_freq)

#     return freq_dist

# def score_sentences(sentences, freq_dist):
    
#     sentence_scores = {}
#     for sentence in sentences:
#         words = word_tokenize(sentence.lower())
#         score = 0
#         for word in words:
#             if word in freq_dist:
#                 score += freq_dist[word]
#         sentence_scores[sentence] = score

#     return sentence_scores

# def generate_summary(sentence_scores, num_sentences=3):
    
#     sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

#     summary_sentences = [sentence[0] for sentence in sorted_sentences[:num_sentences]]

#     summary = ' '.join(summary_sentences)

#     return summary

# # --- TF-IDF based functions ---
# def score_sentences_tfidf(sentences, text):
#     vectorizer = TfidfVectorizer(stop_words='english')
#     vectorizer.fit([text])  # Fit on the entire text
#     sentence_scores = {}
#     for sentence in sentences:
#         words = word_tokenize(sentence.lower())
#         score = 0
#         for word in words:
#             if word in vectorizer.vocabulary_:
#                 score += vectorizer.idf_[vectorizer.vocabulary_[word]]
#         sentence_scores[sentence] = score
#     return sentence_scores

# def fun_tfidf(text):
#     sentences, filtered_words = preprocess_text(text)
#     sentence_scores = score_sentences_tfidf(sentences, text)
#     summary = generate_summary(sentence_scores)
#     print(f"Summary (TF-IDF):\n{summary}\n")
#     return summary


# USING SENTENCE EMBEDDING

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

stop_words = set(stopwords.words("english"))
model = SentenceTransformer('all-MiniLM-L6-v2') # Pre-trained Sentence-BERT model

def preprocess_text(text, stop_words = stop_words):
    
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
    embeddings = model.encode(sentences)
    summary = generate_summary_embeddings(sentences, embeddings)
    print(f"Summary (Sentence Embeddings):\n{summary}\n")
    return summary


if __name__ == "__main__":
    text = """
    Artificial intelligence is transforming the world. It is being used in healthcare, finance, education, and many other fields.
    Machine learning, a subset of AI, enables systems to learn and improve from experience without being explicitly programmed.
    This technology has the potential to revolutionize how we live and work. However, there are concerns about privacy, job displacement, and ethical issues.
    Researchers are working on making AI more transparent and trustworthy.
    """
    
    fun_embeddings(text)
