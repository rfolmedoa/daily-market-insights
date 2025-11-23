from fastapi import FastAPI, HTTPException
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon (first run only)
nltk.download("vader_lexicon")

app = FastAPI(
    title="YFinance API Backend",
    version="1.1.0",
    description="A simple API wrapper around yfinance with a sentiment analysis endpoint."
)

sia = SentimentIntensityAnalyzer()

@app.get("/")
def root():
    return {"status": "ok", "message": "YFinance API running on Render"}

@app.get("/price/{symbol}")
def get_price(symbol: str):
    ticker = yf.Ticker(symbol)

    # Try fast_info first (no JSON decoding)
    try:
        fi = ticker.fast_info
        return {
            "symbol": symbol,
            "price": fi.get("last_price"),
            "currency": fi.get("currency"),
            "name": ticker.get_info().get("shortName")  # safe fallback
        }
    except Exception:
        pass

    # Fallback to get_info() safely
    try:
        info = ticker.get_info()
        return {
            "symbol": symbol,
            "price": info.get("regularMarketPrice"),
            "currency": info.get("currency"),
            "name": info.get("shortName")
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "error": "Could not fetch price data",
            "details": str(e)
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

@app.get("/sentiment/{symbol}")
def get_sentiment(symbol: str):
    ticker = yf.Ticker(symbol)
    news = ticker.news

    if not news:
        raise HTTPException(status_code=404, detail="No news found for this symbol")

    results = []
    scores = []

    for article in news:
        title = article.get("title", "")
        summary = article.get("summary", "")
        text = f"{title}. {summary}"

        sentiment = sia.polarity_scores(text)
        scores.append(sentiment["compound"])

        results.append({
            "title": title,
            "publisher": article.get("publisher"),
            "link": article.get("link"),
            "summary": summary,
            "sentiment": sentiment
        })

    avg = round(sum(scores) / len(scores), 4)
    label = (
        "positive" if avg > 0.05
        else "negative" if avg < -0.05
        else "neutral"
    )

    return {
        "symbol": symbol.upper(),
        "overall_sentiment": label,
        "compound_score": avg,
        "articles": results
    }
