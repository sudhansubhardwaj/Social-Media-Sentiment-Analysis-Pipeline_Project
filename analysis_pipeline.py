import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Data Preprocessing
def preprocess_text(text):
    """
    Clean and preprocess text data.
    """
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub(r"http\S+|@\S+|#\S+|[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_text = " ".join([word for word in tokens if word not in stop_words])
    return filtered_text

# Sentiment Analysis
def analyze_sentiment(text):
    """
    Perform sentiment analysis using VADER.
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    # Classify sentiment based on compound score
    if scores["compound"] > 0.05:
        sentiment = "Positive"
    elif scores["compound"] < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, scores

# Visualization
def visualize_sentiments(results):
    """
    Visualize the sentiment distribution.
    """
    sentiment_counts = results["Sentiment"].value_counts()
    sentiment_counts.plot(kind="bar", color=["green", "red", "blue"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()

# Main Pipeline
def sentiment_analysis_pipeline(input_file):
    """
    Sentiment analysis pipeline.
    """
    # Load data
    data = pd.read_csv(input_file)
    print(f"Loaded {len(data)} posts.")
    
    # Preprocess text
    data["Processed Text"] = data["post"].apply(preprocess_text)
    
    # Perform sentiment analysis
    results = data["Processed Text"].apply(analyze_sentiment)
    data["Sentiment"] = [res[0] for res in results]
    data["Scores"] = [res[1] for res in results]
    
    # Display and save results
    print(data[["post", "Sentiment"]])
    visualize_sentiments(data)
    data.to_csv("results_sentiment_analysis.csv", index=False)
    print("Results saved to 'results_sentiment_analysis.csv'.")

# Run the pipeline
if __name__ == "__main__":
    input_file = "data/social_media_posts.csv"
    sentiment_analysis_pipeline(input_file)
