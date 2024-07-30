import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('movie.csv')

data = data.dropna()


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words if w.isalnum()]
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

data['processed_text'] = data['text'].apply(preprocess_text)


sia = SentimentIntensityAnalyzer()
data['sentiment'] = data['processed_text'].apply(lambda x: sia.polarity_scores(x)['compound'])


data['text_length'] = data['processed_text'].apply(len)
features = data[['sentiment', 'text_length']]


scaler = StandardScaler()
features = scaler.fit_transform(features)


kmeans = KMeans(n_clusters=5, random_state=42)
data['cluster'] = kmeans.fit_predict(features)


sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=data['cluster'], palette='viridis')
plt.xlabel('Sentiment Score')
plt.ylabel('Text Length')
plt.title('K-Means Clustering of Movie Reviews')
plt.show()

score = silhouette_score(features, data['cluster'])
print(f'Silhouette Score: {score}')
