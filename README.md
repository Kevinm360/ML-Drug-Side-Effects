# Side Effects Signal Detection Dashboard

![Dashboard Preview](screenshot.png)

A Streamlit dashboard for exploring FDA adverse event reports and spotting unusual drug-side effect patterns.

The project started as a signal detection tool and later grew into something a little more ambitious: it now also compares machine learning models that try to identify whether a submitted FAERS report is likely to be classified as serious.

Basically, it went from “show me weird reporting patterns” to “okay, what else can we learn from these reports?”

---

## Demo

![Dashboard Demo](demo.gif)

---

## What It Does

The dashboard lets users explore FDA Adverse Event Reporting System (FAERS) data by drug and look at:

- Commonly reported side effects
- Changes in reporting volume over time
- Sudden spikes in specific reactions
- Differences across age and sex groups
- Drug-reaction signals using PRR and ROR
- Serious vs. non-serious report classification

The goal is to surface patterns worth investigating, not to prove that a drug caused an event.

---

## Serious Outcome Classification

The dashboard includes a classification section that predicts whether a FAERS report is labeled as **serious**.

I compare two models:

**Logistic Regression**  
Used as a simple, interpretable baseline.

**XGBoost**  
Used to capture more complex relationships between report characteristics.

The models use information such as:

- Patient demographics
- Number of drugs listed
- Number of reported reactions
- Route of administration
- Drug role information
- Reporter and report characteristics

Fields that directly reveal the outcome, such as hospitalization or death indicators, are excluded so the model cannot simply cheat its way to a good score.

---

## Model Evaluation

The models are tested on held-out reports and compared using:

- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- Confusion matrices
- ROC and precision-recall curves

The dashboard also includes an adjustable classification threshold so users can see how changing the cutoff affects false positives and missed serious reports.

Because one score is never enough. Unfortunately.

---

## Checking for Reporting Bias

FAERS contains a lot of information about how a report was submitted, not just what happened medically.

That means a model could accidentally learn reporting habits instead of meaningful case characteristics.

To check this, the dashboard also trains reduced versions of the models with several reporting-related variables removed and compares performance using the same train/test split.

This helps show how much of the model's performance may depend on reporting-system metadata.

---

## Important Limitation

FAERS is a spontaneous reporting database, so the data includes missing values, duplicate reports, under-reporting, and reporting bias.

The models do **not** predict a patient's personal medical risk and do not establish that a drug caused a specific reaction.

They classify patterns found in submitted FAERS reports.

In other words: useful for analysis, not a replacement for your doctor.

---

## Tech Stack

- **Python**
- **Streamlit**
- **pandas / NumPy**
- **scikit-learn**
- **XGBoost**
- **Plotly / Altair**
- **openFDA / FAERS**
- **PRR / ROR signal detection**

---

## Quickstart

```bash
git clone https://github.com/Kevinm360/ML-Drug-Side-Effects.git
cd ML-Drug-Side-Effects

python -m venv .venv
