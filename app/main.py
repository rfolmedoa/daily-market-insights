from fastapi import FastAPI
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon (only runs the first time)
nltk.download("vader_lexicon")

app = FastAPI()

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

@app.get("/")
def root():
    return {"status": "ok", "message": "YFinance API running on Render"}

@app.get("/price/{symbol}")
def get_price(symbol: str):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "symbol": symbol,
        "price": info.get("regularMarketPrice"),
        "currency": info.get("currency"),
        "name": info.get("shortName")
    }

@app.get("/history/{symbol}")
def get_history(symbol: str, period: str = "1mo", interval: str = "1d"):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    return hist.to_dict()

@app.get("/news/{symbol}")
def get_news(symbol: str):
    ticker = yf.Ticker(symbol)
    raw_news = ticker.news

    enriched_news = []
    for article in raw_news:
        title = article.get("title", "")
        summary = article.get("summary", "")

        # Combine title + summary for stronger signal
        text = f"{title}. {summary}"

        # Sentiment analysis
        sentiment = sia.polarity_scores(text)

        enriched_news.append({
            **article,
            "sentiment": sentiment
        })

    return enriched_news
