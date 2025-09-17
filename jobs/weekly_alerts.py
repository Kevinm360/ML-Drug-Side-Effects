# jobs/weekly_alerts.py
from __future__ import annotations
import os, json, sys
import pandas as pd
from datetime import datetime
from app.alerts import compute_weekly_alerts, send_slack, send_email

# pick how you choose drugs:
MONITORED_DRUGS = os.getenv("MONITORED_DRUGS","IBUPROFEN,ACETAMINOPHEN,METFORMIN").split(",")

CONFIG = {
  "prr_min": float(os.getenv("PRR_MIN","2.0")),
  "chi2_min": float(os.getenv("CHI2_MIN","4.0")),
  "min_reports": int(os.getenv("MIN_REPORTS","5")),
  "z_min": float(os.getenv("Z_MIN","2.0")),     # set empty to disable burst z
  "top_n_per_drug": int(os.getenv("TOP_N_PER_DRUG","15")),
  "serious_only": os.getenv("SERIOUS_ONLY","false").lower() == "true",
  "search_extra": os.getenv("SEARCH_EXTRA",""),  # e.g. ' AND patient.patientsex:1'
}

def main():
    start = os.getenv("ALERT_START")  # if None, library picks last 7 days
    end   = os.getenv("ALERT_END")
    df = compute_weekly_alerts(MONITORED_DRUGS, start=start, end=end, config=CONFIG)
    # persist a breadcrumb
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    out_dir = os.getenv("ALERTS_OUTDIR","alerts")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"alerts_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_json(orient="records"))

    # Slack
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if webhook:
        send_slack(df, webhook_url=webhook, start=df.attrs.get("start", start) or "", end=df.attrs.get("end", end) or "")

    # Email
    if os.getenv("EMAIL_HOST"):
        send_email(
            df,
            smtp_host=os.getenv("EMAIL_HOST"),
            smtp_port=int(os.getenv("EMAIL_PORT","465")),
            smtp_user=os.getenv("EMAIL_USER"),
            smtp_pass=os.getenv("EMAIL_PASS"),
            to_addr=os.getenv("EMAIL_TO"),
            start=start or "", end=end or ""
        )

    print(f"Alerts computed for {len(df)} rows. Saved to {path}")

if __name__ == "__main__":
    main()
