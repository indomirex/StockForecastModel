import spacy
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dictionary mapping tickers to stock names
ticker_to_name = {
    'CVNA': 'Carvana',
    'TSM': 'Taiwan Semiconductor Manufacturing Company',
    'PANW': 'Palo Alto Networks',
    'CRM': 'Salesforce',
    'RXRX': 'Recursion Pharmaceuticals',
    'SOUN': 'SoundHound AI',
    'ARM': 'Arm Holdings',
    'GUTS': 'Gut Health Inc.',
    'INTC': 'Intel Corporation',
    'PYPL': 'PayPal Holdings',
    'SMCI': 'Super Micro Computer',
    'XOM': 'Exxon Mobil Corporation',
    'AMAT': 'Applied Materials',
    'AMZN': 'Amazon',
    'MSFT': 'Microsoft',
    'NVDA': 'Nvidia',
    'NFLX': 'Netflix',
    'META': 'Meta Platforms',
    'GOOGL': 'Alphabet Inc. (Google)',
    'AAPL': 'Apple',
    'TSLA': 'Tesla',
    'AVGO': 'Broadcom Inc.',
    'LULU': 'Lululemon Athletica'
}

class SubjectFinder:
    def __init__(self, model_name='en_core_web_sm', dataset_name='stock_volatility_questions.pkl'):
        # Load spaCy model
        self.nlp = spacy.load(model_name)
        self.dataset_name = dataset_name
        self.vectorizer = TfidfVectorizer()
        self.intents = []
        self.training_questions = []
        self.subject_vectorizer = TfidfVectorizer()
        self.subject_X = None
        self.interrogative_pronouns = ['who', 'what', 'where', 'when', 'which']

        # Load training data and prepare datasets
        self.load_training_data()
        self.prepare_datasets()

    def load_training_data(self):
        try:
            with open('training_data.pkl', 'rb') as f:
                self.training_questions, self.intents = pickle.load(f)
        except FileNotFoundError:
            self.training_questions = []
            self.intents = []

    def prepare_datasets(self):
        # Fit the vectorizer on the training questions
        self.X = self.vectorizer.fit_transform(self.training_questions)
        self.subject_X = self.subject_vectorizer.fit_transform(self.training_questions)

    def get_subject_from_ner(self, text):
        doc = self.nlp(text)
        subjects = []
        for ent in doc.ents:
            if ent.label_ in ('ORG', 'PRODUCT', 'NORP'):  # Include other labels if needed
                subjects.append(ent.text)
        return subjects

    def preprocess_text(self, text):
        # Replace tickers with their stock names
        for ticker, name in ticker_to_name.items():
            if ticker in text:
                text = text.replace(ticker, name)
        return text

    def classify_question(self, question):
        # Preprocess the question for company name abbreviations
        question = self.preprocess_text(question)
        
        query_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vec, self.X)
        most_similar_index = np.argmax(similarities)
        
        # Extract the subject using NER
        subjects = self.get_subject_from_ner(question)
        subject_info = "No clear subject found."
        
        # Include abbreviations in subject information
        for subject in subjects:
            subject_info = f"The subject is '{subject}'."
            break
        
        # If no subject found from NER, check for interrogative pronouns
        if not subjects:
            for pronoun in self.interrogative_pronouns:
                if pronoun in question.lower():
                    subject_info = f"The subject is an interrogative pronoun: '{pronoun}'."
                    break

        return subject_info

    def update_training_data(self, question, correct_intent):
        self.training_questions.append(question)
        self.intents.append(correct_intent)
        self.X = self.vectorizer.fit_transform(self.training_questions)
        self.subject_X = self.subject_vectorizer.fit_transform(self.training_questions)
        with open('training_data.pkl', 'wb') as f:
            pickle.dump((self.training_questions, self.intents), f)

    def run(self):
        while True:
            user_question = input("Please enter your question about stock market volatility: ")
            subject_info = self.classify_question(user_question)
            print(subject_info)

            feedback = input("Is this subject identification correct? (yes/no): ").strip().lower()
            if feedback == 'no':
                correct_subject = input("Please specify the correct subject: ").strip()
                self.update_training_data(user_question, correct_subject)
                print("Thank you! The training data has been updated.")
            elif feedback == 'exit':
                break
            else:
                print("Invalid response. Please type 'yes', 'no', or 'exit'.")

if __name__ == "__main__":
    subject_finder = SubjectFinder()
    subject_finder.run()













