# WhatsApp Sentiment Analyzer & Chat Analytics Dashboard

An ML-powered conversational intelligence platform that combines NLP-based sentiment classification with comprehensive WhatsApp chat analytics.

## Overview
This project progresses through two stages of model development — classical ML baselines followed by a fine-tuned transformer — and pairs the sentiment engine with a full behavioral analytics dashboard for WhatsApp conversations.

## Features
- **Sentiment Analysis** — 3-class classification (Positive / Neutral / Negative) using SVM, Logistic Regression, and fine-tuned DistilBERT
- **Batch Processing** — CSV upload or multiline paste with downloadable annotated output
- **WhatsApp Parser** — supports 8 timestamp format variants across iOS, Android, and regional locales
- **Chat Analytics** — 20+ behavioral metrics: response times, double-text frequency, emoji usage, peak activity hours, conversation streaks
- **Privacy-First** — fully local execution, no data ever leaves your machine

## Models
| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 80.53% | 0.77 |
| SVM (LinearSVC) | 80.74% | 0.78 |
| DistilBERT (fine-tuned) | 85.45% | 0.84 |

## Tech Stack
Python, Streamlit, Scikit-learn, HuggingFace Transformers, Pandas, Plotly
