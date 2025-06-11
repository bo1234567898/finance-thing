import gradio as gr
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

import requests
from bs4 import BeautifulSoup

def scrape_headlines(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    # Target news section specifically
    news_section = soup.find_all("li", class_="js-stream-content")
    for item in news_section:
        title_tag = item.find("h3")
        if title_tag and title_tag.text:
            headlines.append(title_tag.text.strip())
        if len(headlines) >= 5:
            break

    if not headlines:
        return ["No relevant headlines found."]
    return headlines

import yfinance as yf

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    current_price = info.get("currentPrice", "N/A")
    previous_close = info.get("previousClose", "N/A")
    market_cap = info.get("marketCap", "N/A")
    summary = info.get("longBusinessSummary", "No summary available.")

    return f""":
- Current Price: ${current_price}
- Previous Close: ${previous_close}
- Market Cap: {market_cap}
- Description: {summary[:300]}..."""  # Shorten if too long

def get_historical_data(ticker, period="1mo", interval="1d"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    if hist.empty:
        return "No historical data found."

    summary = hist.tail(5).to_string()
    return summary

def get_financial_ratios(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    ratios = {
        "Trailing P/E": info.get("trailingPE", "N/A"),
        "Forward P/E": info.get("forwardPE", "N/A"),
        "PEG Ratio": info.get("pegRatio", "N/A"),
        "Profit Margin": info.get("profitMargins", "N/A"),
        "Return on Equity": info.get("returnOnEquity", "N/A"),
        "Debt/Equity": info.get("debtToEquity", "N/A"),
        "EPS": info.get("trailingEps", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A"),
    }

    return "\n".join([f"{k}: {v}" for k, v in ratios.items()])

import pandas_ta as ta

def get_rsi(ticker, period="14d", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            return "No historical data available for RSI."
        rsi = ta.rsi(data["Close"], length=14)
        latest_rsi = rsi.dropna().iloc[-1]
        return f"RSI (14-day): {latest_rsi:.2f}"
    except Exception as e:
        return f"Error retrieving RSI: {e}"

from fredapi import Fred

fred = Fred(api_key="56ea706209d874a634057cba9c49241b")

def get_inflation_fred():
    api_key = "56ea706209d874a634057cba9c49241b"
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={api_key}&file_type=json"
    response = requests.get(url)
    data = response.json()
    latest = data["observations"][-1]
    return f"US Inflation (CPI): {latest['value']}% as of {latest['date']}"

model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def summarize_with_pegasus(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT sentiment model
sentiment_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def analyze_sentiment_from_summary(summary):
    inputs = sentiment_tokenizer(summary, return_tensors="pt", truncation=True)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    result = {label: float(prob) for label, prob in zip(labels, probs[0])}

    sentiment = max(result, key=result.get)
    confidence = result[sentiment]

    # Add simple advice
    if sentiment == "positive":
        advice = "News looks promising. Consider a bullish outlook."
    elif sentiment == "neutral":
        advice = "No strong signals from the news. Hold or monitor further."
    else:
        advice = "News is negative. Exercise caution or review your position."

    explanation = f"Sentiment: **{sentiment}** ({confidence:.2%} confidence)\n📌 Advice: {advice}"
    return explanation

def summarize_and_analyze(text):
    summary = summarize_with_pegasus(text)
    gpt_analysis = analyze_sentiment_from_summary(summary)
    return f"📄 Summary:\n{summary}\n\n💬 GPT Analysis:\n{gpt_analysis}"

import ollama

def analyze_with_mistral(summary, stock_info, historical_data, financial_ratios, rsi, macro_data):
    prompt = f"""
You are a financial analyst.

📄 Company Summary:
{summary}

💹 Stock Info:
{stock_info}

📊 Recent Historical Stock Prices:
{historical_data}

📈 Financial Ratios:
{financial_ratios}

📈 RSI (Relative Strength Index):
{rsi}

🌐 Macroeconomic Context:
{macro_data}

Give a well-rounded, professional analysis. Include trend, momentum, valuation, macro impact, and any actionable advice.
"""
    output = ollama.chat(model="mistral", messages=[
        {"role": "user", "content": prompt}
    ])
    return output['message']['content']

def full_pipeline(ticker):
    headlines = scrape_headlines(ticker)
    joined_headlines = "\n".join(headlines)  # ✅ Safely join when needed
    summary = summarize_with_pegasus(joined_headlines)
    sentiment = analyze_sentiment_from_summary(summary)
    stock_info = get_stock_data(ticker)
    historical = get_historical_data(ticker)
    financials = get_financial_ratios(ticker)
    rsi = get_rsi(ticker)
    macro_data = get_inflation_fred()

    mistral_opinion = analyze_with_mistral(summary, stock_info, historical, financials, rsi, macro_data)
    result = f"""🔎 **Scraped Headlines:**  
{joined_headlines}

📄 **Summary:**  
{summary}

💬 **Sentiment Analysis:**  
{sentiment}

💹 **Stock Info:**  
{stock_info}

📊 **Financial Ratios:**  
{financials}

📈 **RSI (Relative Strength Index):**
{rsi}

🌐 **Macroeconomic Indicators:**  
{macro_data}

🧠 **Mistral Insights:**  
{mistral_opinion}
"""
    return result

app = gr.Interface(
    fn=full_pipeline,
    inputs=gr.Textbox(label="Enter stock ticker (e.g., AAPL)"),
    outputs="text",
    title="📊 Auto Financial News Analyzer",
    description="Enter a stock ticker. The app scrapes Yahoo Finance, summarizes the news, and analyzes tone/strategy with MISTRAL."
)

app.launch()