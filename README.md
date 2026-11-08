# FraudLens

**Explainable real-time fraud detection: risk scoring, SHAP explanations, a live dashboard and plain-English alert summaries.**

[Live demo](YOUR_DEMO_LINK) · [Demo video](YOUR_VIDEO_LINK)

![Demo](docs/demo.gif)

> Replace the links above and add `docs/demo.gif` (a 20 to 30 second screen recording of alerts appearing and an explanation being shown).

---

## Overview

Fraud is rare (under 1% of transactions in this dataset), costly, and has to be explained to analysts and regulators. FraudLens is an end-to-end system that:

1. Scores each transaction for fraud risk with a gradient-boosted model.
2. Explains **why** it was flagged using SHAP.
3. Serves predictions through a REST API.
4. Shows a live (simulated) transaction stream and alerts on a dashboard.
5. Turns the SHAP output into a short plain-English note using an LLM, with a deterministic fallback if the LLM is unavailable.

## Features

- Imbalance-aware training (class weights and SMOTE applied inside CV folds only, so there is no leakage)
- Metrics that fit the problem: precision, recall, PR-AUC (not accuracy)
- Cost-based decision threshold instead of the default 0.5
- Global and per-transaction SHAP explanations
- FastAPI scoring service with input validation
- Streamlit dashboard with a simulated live stream and an adjustable threshold
- LLM-written alert summaries that use only the structured facts provided
- Dockerized, tested, with CI on every push

## Architecture

```
Simulated transaction stream
          |
          v
   FastAPI  /score  --->  Model (XGBoost) + SHAP
          |                      |
          v                      v
   Streamlit dashboard  <---  LLM explainer (with template fallback)
```

## Tech stack

| Area | Tools |
|---|---|
| Modeling | Python, pandas, scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit, Plotly |
| LLM layer | [Gemini / Groq / Ollama: edit to match what you used] |
| DevOps | Docker, pytest, GitHub Actions |

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1): a synthetic mobile-money transaction dataset. It is synthetic, so results here show the method works but do not predict performance on real bank data. Download it and place the CSV in `data/` (the folder is gitignored).

## Results

> Fill this table with numbers you actually measured on a held-out test set. Do not leave placeholders in the published version.

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | [ ] | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] | [ ] |
| XGBoost (tuned) | [ ] | [ ] | [ ] | [ ] |

**Threshold selection:** with a missed fraud assumed to cost [N]x a false alarm, the cost-optimal threshold was [T], which [reduced total simulated cost by Z%] compared with 0.5.

## Key design decisions

- **Why PR-AUC and recall:** with so few fraud cases, accuracy is misleading. A model that flags nothing is about 99% "accurate".
- **Why resampling happens inside CV folds:** resampling before splitting leaks information and inflates results.
- **Why a cost-based threshold:** a missed fraud and a false alarm have very different costs, so 0.5 is rarely the right cutoff.
- **Why the LLM only explains:** the model makes the decision. The LLM just restates the SHAP facts in plain language, and the app works without it.
- **Why PaySim over the anonymized credit card dataset:** readable features make explanations meaningful.

## Getting started

### Prerequisites

- Python 3.10+
- (Optional) Docker
- (Optional) an LLM API key for the explanation layer

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/fraudlens.git
cd fraudlens

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train the model (expects data/ to contain the PaySim CSV)
python -m src.train

# Start the API
uvicorn api.main:app --reload

# In a second terminal, start the dashboard
streamlit run app/dashboard.py
```

### Run with Docker

```bash
docker compose up --build
```

### Environment variables

Create a `.env` file (never commit it):

```
LLM_API_KEY=your_key_here
```

Without a key, alert explanations fall back to a template-based summary.

## API example

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
        "type": "TRANSFER",
        "amount": 181000.0,
        "oldbalanceOrg": 181000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
      }'
```

Example response:

```json
{
  "risk_score": 0.97,
  "is_flagged": true,
  "top_reasons": [
    {"feature": "newbalanceOrig", "value": 0.0, "impact": 1.8},
    {"feature": "amount", "value": 181000.0, "impact": 1.2},
    {"feature": "type", "value": "TRANSFER", "impact": 0.9}
  ]
}
```

> Adjust the field names and the response to match your actual API.

## Project structure

```
fraudlens/
├── api/main.py            # FastAPI app
├── app/dashboard.py       # Streamlit dashboard
├── src/
│   ├── features.py        # preprocessing
│   ├── train.py           # training and evaluation
│   ├── explain.py         # SHAP helpers
│   └── llm_explainer.py   # LLM summaries with fallback
├── notebooks/             # EDA and experiments
├── tests/
├── models/                # saved model artifacts
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Testing

```bash
pytest
```

Tests cover feature processing, the API endpoints and the threshold logic. CI runs them on every push.

## Limitations

- Trained on **synthetic** data, so real-world performance would differ.
- The transaction stream is simulated by replaying held-out data.
- Fraud patterns drift over time, and this project does not retrain automatically.
- LLM summaries depend on an external API and can be unavailable, which is why the template fallback exists.

## Future work

- Drift monitoring and scheduled retraining
- Evaluation on additional datasets
- Feature stores and real streaming (Kafka) instead of replay
- Analyst feedback loop to improve the model

## Author

**YOUR NAME** · [LinkedIn](YOUR_LINKEDIN) · [GitHub](YOUR_GITHUB)
MIT. See [LICENSE](LICENSE).
