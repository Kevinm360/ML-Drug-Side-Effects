# Side Effects Signal Detection Dashboard

![Dashboard Preview](screenshot.png)

## TL;DR

This project uses FDA adverse event reports to look for unusual drug side-effect patterns and test whether machine learning can help identify reports that are more likely to be classified as serious.

In plain English: it looks for weird reporting spikes, unusually common drug-reaction combinations, and patterns that separate serious reports from non-serious ones.

I used two classification models:

- **Logistic Regression** as the simpler, easier-to-explain baseline
- **XGBoost** to pick up more complicated patterns between report features

The dashboard compares both models using metrics like precision, recall, F1, ROC-AUC, and PR-AUC. It also lets you change the classification threshold and immediately see how that changes false positives and missed serious cases.

One thing I paid close attention to was **data leakage**. FAERS includes fields like hospitalization, death, and other outcome indicators that would basically hand the model the answer. Those are excluded from training because that would be cheating, and unfortunately the model does not get to peek at the answer key.

I also tested a reduced version of the model with several reporter and reporting-system fields removed. The goal was to see whether the model was learning useful report patterns or just getting suspiciously good at recognizing how reports were submitted.

## What the Dashboard Covers

The app includes:

- Adverse event trends over time
- Common reported reactions
- PRR and ROR safety signals
- Reporting spikes and burst detection
- Demographic filtering
- Drug comparisons
- Serious outcome classification
- Model performance charts
- Feature importance
- Adjustable classification thresholds

## Data

The project uses the FDA Adverse Event Reporting System, or **FAERS**, through openFDA.

FAERS is useful for finding reporting patterns, but it has limits. Reports can be incomplete, duplicated, biased, or missing important information.

Because of that, this project does **not** claim that a drug caused a side effect or that a model can predict someone’s personal medical risk.

It is a signal-detection and analysis tool, not Dr. House.

## Tech Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Streamlit
- Plotly
- Altair
- openFDA / FAERS

## Running the Project

```bash
git clone https://github.com/Kevinm360/ML-Drug-Side-Effects.git
cd ML-Drug-Side-Effects

python -m venv .venv
pip install -r requirements.txt

streamlit run app/streamlit_app.py
