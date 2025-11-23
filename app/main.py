from fastapi import FastAPI, HTTPException
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon (only first run)
nltk.download("vader_lexicon")

app = FastAPI()

# Initialize VADER sentiment analyzer
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
    return ticker.news


# ------------------------------------------------------------
# NEW ENDPOINT: SENTIMENT ANALYSIS
# ------------------------------------------------------------
@app.get("/sentiment/{symbol}")
def get_sentiment(symbol: str):
    """
    Returns sentiment scores for each news article related to the symbol.
    Also returns an aggregated sentiment score.
    """
    ticker = yf.Ticker(symbol)
    news = ticker.news

    if not news:
        raise HTTPException(status_code=404, detail="No news found for this symbol")

    sentiment_results = []
    compound_scores = []

    for article in news:
        title = article.get("title", "")
        summary = article.get("summary", "")
        text = f"{title}. {summary}"

        scores = sia.polarity_scores(text)
        compound_scores.append(scores["compound"])

        sentiment_results.append({
            "title": title,
            "publisher": article.get("publisher"),
            "link": article.get("link"),
            "summary": summary,
            "sentiment": scores
        })

    # Aggregate sentiment
    avg_compound = round(sum(compound_scores) / len(compound_scores), 4)

    sentiment_label = (
        "positive" if avg_compound > 0.05
        else "negative" if avg_compound < -0.05
        else "neutral"
    )

    return {
        "symbol": symbol.upper(),
        "overall_sentiment": sentiment_label,
        "compound_score": avg_compound,
        "articles": sentiment_results
    }
