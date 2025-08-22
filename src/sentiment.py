import os  
import requests  
from datetime import datetime, timedelta  
from dotenv import load_dotenv  
import yfinance as yf  
import nltk  
from nltk.sentiment.vader import SentimentIntensityAnalyzer  
from textblob import TextBlob 
from transformers import pipeline  

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# 'vader_lexicon' is the dictionary used by the VADER sentiment analyzer.
nltk.download("vader_lexicon")
# 'punkt' is a pre-trained tokenizer for splitting text into sentences.
nltk.download("punkt")

# We iniatialize a pretrained sentiment analysis pipeline from Hugging Face.
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")



def _get_company_name_from_ticker(ticker):
    """
    Fetches the full company name from a given stock ticker using yfinance.
    
    Working:
            - Uses yfinance to get the company's info.
            - Tries to extract 'longName' or 'shortName' from the info.
    """

    info = yf.Ticker(ticker).info
    
    name = info.get("longName") or info.get("shortName")

    if name and isinstance(name, str):
        return name
    return None



def get_headlines(ticker):
    """
    Retrieves recent news headlines for a given company from the NewsAPI.
    
    Working:
            - Uses NewsAPI to fetch articles related to the company.
            - Filters articles to only include those from the last 30 days.
    """

    # Get the company's full name using the ticker symbol. If not found, use the ticker itself.
    company = _get_company_name_from_ticker(ticker) or ticker
    
    # Set the date range for the news search to ensure we get recent articles.
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=30)

    # The search query combines the company name and ticker symbol.
    q = f'"{company}" OR {ticker}'
    
    # Parameters for the NewsAPI request
    params = {

        "q": q,  # The search query

        "from": start_date.isoformat(),  # Start date of the search

        "to": end_date.isoformat(),  # End date of the search

        "language": "en",  # Filter for English language articles

        "sortBy": "relevancy",  # Sort articles by relevance to the query

        "pageSize": 30,  # Number of results to return

        "apiKey": NEWSAPI_KEY  # Your API key for authentication
    }
    
    # Make the GET request to the NewsAPI endpoint
    resp = requests.get("https://newsapi.org/v2/everything", params=params)

    # Raise an exception if the request returned an error status code
    resp.raise_for_status()
    
    # We parse the JSON response to extract article information
    data = resp.json()

    # Safely get the list of articles from the response data
    articles = data.get("articles", [])

    # Extract just the title from each article
    headlines = [a.get("title") for a in articles if a.get("title")]
    
    return headlines



def analyze_sentiment(headlines):
    """
    Analyzes a list of headlines using three different sentiment analysis methods
    and calculates an average sentiment score.
    
    Args:
        headlines (list): A list of headline strings.
        
    Returns:
        tuple: A tuple containing a list of individual headline scores and the overall average score.
    """
    # Initialize the VADER sentiment intensity analyzer
    sid = SentimentIntensityAnalyzer()
    scores = []

    # Iterate over each headline to analyze its sentiment
    for h in headlines:
        parts = []  # To store scores from the three different models for the current headline

        # 1. VADER Analysis: Scores range from -1 (most negative) to +1 (most positive)
        # The 'compound' score is a single, normalized value.
        v = sid.polarity_scores(h)["compound"]
        parts.append(v)

        # 2. TextBlob Analysis: Polarity is a float within the range [-1.0, 1.0]
        tb = TextBlob(h).sentiment.polarity
        parts.append(tb)

        # 3. Hugging Face Transformers Analysis
        # The model can only process sequences up to 512 tokens long, so we truncate the headline.
        r = classifier(h[:512])[0]
        label = r.get("label", "")
        sc = r.get("score", 0.0)
        # Convert the model's output (label + score) to a single float from -1.0 to 1.0.
        # If the label is POSITIVE, the score is positive. If NEGATIVE, we make it negative.
        if label.upper().startswith("POS"):
            parts.append(sc)
        else:
            parts.append(-sc)

        # If we have scores, calculate their average for this one headline
        if parts:
            combined = sum(parts) / len(parts)
            scores.append(combined)

    # If no scores were generated, return empty list and 0.0
    if not scores:
        return [], 0.0
        
    # Calculate the overall average sentiment across all headlines
    average = sum(scores) / len(scores)
    
    return scores, average



# Testing Script
if __name__ == "__main__":
    
    hs = get_headlines("TGT")
    
    # Print each headline with its number
    for i, h in enumerate(hs, 1):
        print(i, h)
    
    # We analyze the sentiment of the fetched headlines
    scores, avg = analyze_sentiment(hs)
    
    # Log the results
    print("Headlines found:", len(hs))
    print("Average sentiment:", avg)