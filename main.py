import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from yahooquery import search
from urllib.parse import urljoin
import spacy
from datetime import datetime
from langchain_community.llms import Ollama
import streamlit as st
import subprocess
import importlib.util
import sys

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_company_name(text):
    doc = nlp(text)
    # Extract first ORGANIZATION entity found
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return ent.text
    # If no ORG entity found, fallback to None or some default
    return None
def get_ticker_from_company_name(text):
    results = search(text)
    quotes = results.get("quotes", [])
    if quotes:
        return quotes[0]["symbol"]  # Take the top result
    return None

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_sec_filing_text(ticker):
    base_url = f"https://www.sec.gov/cgi-bin/browse-edgar"
    headers = {"User-Agent": "Mozilla/5.0"}

    # Step 1: Get the filings page
    params = {
        "action": "getcompany",
        "CIK": ticker,
        "type": "10-",
        "owner": "exclude",
        "count": "10"
    }

    try:
        res = requests.get(base_url, headers=headers, params=params)
        soup = BeautifulSoup(res.text, "html.parser")

        # Step 2: Find the first filing row with a "Documents" link
        doc_link_tag = soup.find("a", string="Documents")
        if not doc_link_tag:
            return "No filings found."

        filing_page_url = urljoin("https://www.sec.gov", doc_link_tag["href"])
        filing_page = requests.get(filing_page_url, headers=headers)
        filing_soup = BeautifulSoup(filing_page.text, "html.parser")

        # Step 3: Get the primary document (typically .htm or .txt)
        table = filing_soup.find("table", class_="tableFile", summary="Document Format Files")
        if not table:
            return "No document table found in filing."

        first_row = table.find_all("tr")[1]  # Skip header row
        cols = first_row.find_all("td")
        if len(cols) < 3:
            return "No valid document row found."

        doc_url = urljoin("https://www.sec.gov", cols[2].a["href"])
        doc_text = requests.get(doc_url, headers=headers).text
        return doc_text[:5000]  # Truncate for performance
    except Exception as e:
        return f"Error during SEC scraping: {str(e)}"

def gather_and_retrieve(text, k=5):
    ticker = get_ticker_from_company_name(text)
    if not ticker:
        return ["No ticker found for query. Please be more specific."]

    chunks = []

    # --- API Agent: Latest stock data including today's intraday ---
    stock = yf.Ticker(ticker)
    try:
        # Get last 7 days with daily resolution (including today if market open)
        hist = stock.history(period="7d", interval="1d").to_string()

        # Optionally, get intraday (last trading day, 1m interval) - smaller chunk
        intraday = stock.history(period="1d", interval="1m").tail(10).to_string()

        info = stock.info
        chunks.append(f"[{ticker}] Recent daily stock prices (last 7 days including today):\n" + hist)
        chunks.append(f"[{ticker}] Intraday (last 10 minutes) stock prices for today:\n" + intraday)
        chunks.append(f"[{ticker}] Company summary:\n" + info.get("longBusinessSummary", "N/A"))
    except Exception as e:
        chunks.append(f"[{ticker}] Failed to get Yahoo Finance data: {str(e)}")

    # --- Scraping Agent ---
    filing_text = get_sec_filing_text(ticker)
    chunks.append(f"[{ticker}] SEC filing snippet:\n" + filing_text)

    # --- (Optional) Add other data sources if needed ---

    # --- Retriever Agent ---
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    query_vec = model.encode([text])
    _, indices = index.search(np.array(query_vec), k)
    formatted_chunks = []
    for i in indices[0]:
        chunk = chunks[i]
        # Clean and shorten each chunk to make it LLM-friendly
        clean_chunk = chunk.strip().replace("\n", " ").replace("  ", " ")
        # Limit very long sections (e.g., SEC text or long price tables)
        if len(clean_chunk) > 800:
            clean_chunk = clean_chunk[:800] + "..."
        formatted_chunks.append(f"{i+1}. {clean_chunk}")

    return formatted_chunks

def gather_market_insights(text):
    text_lower = text.lower()
    insights = []

    # --- Broad indices ---
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC"
    }

    for name, symbol in indices.items():
        try:
            data = yf.Ticker(symbol).history(period="1d", interval="5m")
            latest = data.iloc[-1]
            open_price = data.iloc[0]["Open"]
            change = latest["Close"] - open_price
            percent = (change / open_price) * 100
            insights.append(f"{name}: {latest['Close']:.2f} ({percent:+.2f}%)")
        except:
            continue

    # --- Sector/theme ETF mappings ---
    etf_map = {
        "tech": "XLK",
        "energy": "XLE",
        "finance": "XLF",
        "health": "XLV",
        "utilities": "XLU",
        "industrials": "XLI",
        "consumer": "XLY",
        "materials": "XLB",
        "semiconductors": "SMH",
        "asia": "AIA",
        "emerging": "EEM",
        "china": "FXI",
        "india": "INDA"
    }

    matched = False
    for keyword, etf in etf_map.items():
        if keyword in text_lower:
            matched = True
            try:
                hist = yf.Ticker(etf).history(period="1d", interval="5m")
                latest = hist.iloc[-1]
                open_price = hist.iloc[0]["Open"]
                change = latest["Close"] - open_price
                percent = (change / open_price) * 100
                insights.append(f"{keyword.title()} theme ({etf}): {latest['Close']:.2f} ({percent:+.2f}%)")
            except Exception as e:
                insights.append(f"Failed to load {keyword} ETF ({etf}): {e}")

    # --- If nothing matched, add hint ---
    if not matched:
        insights.append("No specific sector/theme detected — showing broad market indices only.")

    # --- Optional: volatility indicator (risk sentiment) ---
    if "risk" in text_lower or "volatility" in text_lower:
        try:
            vix = yf.Ticker("^VIX").history(period="1d", interval="5m")
            latest = vix.iloc[-1]["Close"]
            insights.append(f"VIX (volatility index): {latest:.2f}")
        except:
            insights.append("Failed to retrieve VIX (volatility index).")

    return insights[:8]

llm = Ollama(model="phi3")
def synthesize_insights(query: str, insights: list[str]) -> str:
    system_prompt = (
        "You are a financial assistant. Given a user query and a list of market insights, "
        "answer only what's asked using simple, clear language. Don't give just stats. Answer in a few sentences. Do not include irrelevant data."
    )
    full_prompt = system_prompt + "\n\nUser query: " + query + "\n\nInsights:\n" + "\n".join(insights) + "\n\nAnswer:"
    return llm(full_prompt)


query = "Show me Amazon's latest earnings"
company = extract_company_name(query)

# if company:
#     result = gather_and_retrieve(company)
#     summary = synthesize_insights(query, result)
# else:
#
#     insights = gather_market_insights(query)
#
#     summary = synthesize_insights(query, insights)
#
# print(summary)

st.set_page_config(page_title="Financial Query Assistant", page_icon="💹")

st.title("💹 Financial Query Assistant")
st.write("Enter your query about companies, stocks, or market insights:")

# Input box for user query
user_query = st.text_input("Enter your question or company name:", "")

if user_query:
    company = extract_company_name(user_query)

    if company:
        result = gather_and_retrieve(company)
        summary = synthesize_insights(user_query, result)
    else:

        insights = gather_market_insights(query)

        summary = synthesize_insights(user_query, insights)
    with st.spinner("Fetching data and generating answer..."):
        # # Get relevant data chunks from your gather_and_retrieve function
        relevant_chunks = gather_and_retrieve(user_query)
        #
        # # Get market insights based on query keywords
        market_insights = gather_market_insights(user_query)

        # Synthesize final answer combining query and gathered insights
        #answer = synthesize_insights(user_query, market_insights + relevant_chunks)
        answer=summary
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Relevant Data Snippets")
    for chunk in relevant_chunks:
        st.markdown(f"- {chunk}")

    st.subheader("Market Insights")
    for insight in market_insights:
        st.markdown(f"- {insight}")
else:
    st.info("Please enter a query to get started.")
