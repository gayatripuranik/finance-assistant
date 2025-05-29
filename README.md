# Financial Assistant

A lightweight financial assistant that takes a natural language query and returns relevant market data, SEC filings, stock performance, and synthesized insights using language models.

## Overview

This Streamlit-based web application allows users to input a financial query (e.g., "What’s going on with Tesla today?") and get back a concise answer generated from structured and unstructured data sources.

## Features

- Named entity recognition to extract company names from user queries
- Ticker symbol retrieval via YahooQuery
- Stock data (daily and intraday) from Yahoo Finance using `yfinance`
- SEC 10-K/10-Q filing scraping from the official SEC website
- Sector/theme market sentiment using ETFs
- Vector similarity search using `faiss` and `SentenceTransformer`
- Final response generation using `Ollama` and LLM (`phi3`)

## Tools and Libraries

- `streamlit` – Web interface
- `yfinance` – Stock and ETF price data
- `yahooquery` – Company search and ticker mapping
- `requests` + `beautifulsoup4` – SEC filings scraping
- `spacy` – Named entity recognition
- `faiss-cpu` – Vector similarity search
- `sentence-transformers` – Text embeddings
- `ollama` – Lightweight local language model backend (phi3)
- `python-dotenv` (optional) – Environment variable management

