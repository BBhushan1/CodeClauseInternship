import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from fuzzywuzzy import process
import re



data = pd.read_csv('IMBD.csv')


data = data.dropna(subset=['movie', 'description', 'genre'])

data['primary_genre'] = data['genre'].apply(lambda x: x.split(',')[0].strip())

data['text'] = data['movie'] + ' ' + data['description']

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    
    tokens = word_tokenize(text.lower())
   
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return ' '.join(tokens)


data['processed_text'] = data['text'].apply(preprocess_text)


tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(data['processed_text'])
y = data['primary_genre']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))


def get_movie_details(plot_summary):
   
    preprocessed_summary = preprocess_text(plot_summary)
   
    X_new = tfidf.transform([preprocessed_summary])
   
    predicted_genre = model.predict(X_new)[0]
    
    return {
        'predicted_genre': predicted_genre
    }

while True:
    movie_input = input("Enter the plot summary of the movie (or type 'exit' to quit): ")
    if movie_input.lower() == 'exit':
        break
    
    movie_details = get_movie_details(movie_input)
    if movie_details:
        print(f"Predicted Genre: {movie_details['predicted_genre']}")
    else:
        print("No genre prediction available.")
