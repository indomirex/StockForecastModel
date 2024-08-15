
import numpy as np
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
import torch

class IntentClassifier:
    def __init__(self):
        # Load spaCy model and BERT components
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

        # Load training data from file
        self.training_questions, self.intents = self.load_training_data()
        
        # Create BERT embeddings for training data
        self.train_embeddings = self.get_bert_embeddings(self.training_questions)
        
        # Train a classifier on the BERT embeddings
        self.classifier = LogisticRegression()
        self.classifier.fit(self.train_embeddings, self.intents)

    def load_training_data(self):
        try:
            with open('training_data.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return [], []

    def preprocess_text(self, text):
        # Replace full names with abbreviations if necessary
        return text

    def get_bert_embeddings(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            embeddings.append(cls_embedding)
        return np.array(embeddings)

    def classify_question(self, question):
        # Preprocess the question
        question = self.preprocess_text(question)
        
        # Get BERT embeddings for the question
        inputs = self.tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
        
        # Predict intent using the trained classifier
        predicted_intent = self.classifier.predict([query_embedding])[0]
        
        return predicted_intent, f"The intent of the question is '{predicted_intent}'."

    def update_training_data(self, question, correct_intent):
        self.training_questions.append(question)
        self.intents.append(correct_intent)
        self.train_embeddings = self.get_bert_embeddings(self.training_questions)
        self.classifier.fit(self.train_embeddings, self.intents)
        with open('training_data.pkl', 'wb') as f:
            pickle.dump((self.training_questions, self.intents), f)

    def run(self):
        while True:
            user_question = input("Please enter your question about stock market volatility: ")
            intent, subject_info = self.classify_question(user_question)
            print(f"The intent of the question is: {intent}")
            print(subject_info)

            feedback = input("Is this intent correct? (yes/no): ").strip().lower()
            if feedback == 'no':
                correct_intent = input("Please specify the correct intent (bullish/bearish/volatile): ").strip().lower()
                if correct_intent in ['bullish', 'bearish', 'volatile']:
                    self.update_training_data(user_question, correct_intent)
                    print("Thank you! The training data has been updated.")
                else:
                    print("Invalid intent specified. No changes made.")
            elif feedback == 'exit':
                break
            else:
                print("Invalid response. Please type 'yes', 'no', or 'exit'.")

intent = IntentClassifier()
intent.run()























