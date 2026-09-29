"""Leakage-controlled report-level FAERS seriousness classification."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "serious"
DATE_COLUMN = "receivedate"

NUMERIC_FEATURES = [
    "patient_age_years",
    "patient_weight_kg",
    "drug_count",
    "reaction_count",
    "primary_suspect_drug_count",
    "concomitant_drug_count",
    "interacting_drug_count",
]

CATEGORICAL_FEATURES = [
    "patient_sex",
    "reporter_qualification",
    "reporter_country",
    "primary_source_country",
    "occurrence_country",
    "report_type",
    "subject_drug_role",
    "subject_drug_route",
]

PREDICTOR_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ADMINISTRATIVE_PROXY_FEATURES = [
    "primary_suspect_drug_count",
    "concomitant_drug_count",
    "interacting_drug_count",
    "reporter_qualification",
    "reporter_country",
    "primary_source_country",
    "occurrence_country",
    "report_type",
    "subject_drug_role",
]

REDUCED_PREDICTOR_COLUMNS = [
    feature for feature in PREDICTOR_COLUMNS
    if feature not in ADMINISTRATIVE_PROXY_FEATURES
]
REDUCED_NUMERIC_FEATURES = [
    feature for feature in NUMERIC_FEATURES
    if feature in REDUCED_PREDICTOR_COLUMNS
]
REDUCED_CATEGORICAL_FEATURES = [
    feature for feature in CATEGORICAL_FEATURES
    if feature in REDUCED_PREDICTOR_COLUMNS
]

FEATURE_LABELS = {
    "patient_age_years": "Patient age (years)",
    "patient_weight_kg": "Patient weight (kg)",
    "drug_count": "Number of drugs listed",
    "reaction_count": "Number of reactions listed",
    "primary_suspect_drug_count": "Primary-suspect drug count",
    "concomitant_drug_count": "Concomitant drug count",
    "interacting_drug_count": "Interacting drug count",
    "patient_sex": "Patient sex",
    "reporter_qualification": "Reporter qualification",
    "reporter_country": "Reporter country",
    "primary_source_country": "Primary-source country",
    "occurrence_country": "Occurrence country",
    "report_type": "FAERS report type",
    "subject_drug_role": "Subject drug role",
    "subject_drug_route": "Subject drug route",
}

FEATURE_DESCRIPTIONS = {
    "patient_age_years": "Patient age normalized to years",
    "patient_weight_kg": "Reported patient weight in kilograms",
    "drug_count": "Number of drugs listed on the report",
    "reaction_count": "Number of reactions listed on the report (not reaction terms/outcomes)",
    "primary_suspect_drug_count": "Count of drugs marked primary suspect",
    "concomitant_drug_count": "Count of drugs marked concomitant",
    "interacting_drug_count": "Count of drugs marked interacting",
    "patient_sex": "Coded patient sex",
    "reporter_qualification": "Reporter qualification code",
    "reporter_country": "Reporter country",
    "primary_source_country": "Primary-source country",
    "occurrence_country": "Country where the event occurred",
    "report_type": "FAERS report type",
    "subject_drug_role": "Queried drug's suspect/concomitant/interacting role",
    "subject_drug_route": "Queried drug's administration-route code",
}

FEATURE_PROXY_AUDIT = {
    "patient_age_years": (
        "Low", "Patient characteristic; missingness and population mix can still reflect reporting patterns."
    ),
    "patient_weight_kg": (
        "Moderate", "Patient characteristic, but whether weight is recorded can proxy report completeness."
    ),
    "drug_count": (
        "Moderate", "Report-composition count that can reflect case complexity or documentation depth."
    ),
    "reaction_count": (
        "Moderate", "Report-composition count that can reflect event complexity or documentation depth."
    ),
    "primary_suspect_drug_count": (
        "High", "Derived from reporter/case-processor drug-role coding, which may vary with case handling."
    ),
    "concomitant_drug_count": (
        "High", "Derived from administrative drug-role coding rather than a direct patient characteristic."
    ),
    "interacting_drug_count": (
        "High", "Derived from administrative drug-role coding and vulnerable to reporting practice differences."
    ),
    "patient_sex": (
        "Low", "Patient characteristic; population and missingness differences remain possible."
    ),
    "reporter_qualification": (
        "High", "Reporter type can influence documentation, follow-up, and serious-case submission patterns."
    ),
    "reporter_country": (
        "High", "Can encode national reporting rules, submission channels, and surveillance practices."
    ),
    "primary_source_country": (
        "High", "Can encode source-system and jurisdiction-specific reporting practices."
    ),
    "occurrence_country": (
        "High", "Geography can proxy regulatory, healthcare, and reporting-system differences."
    ),
    "report_type": (
        "High", "Administrative report classification that can reflect the submission pathway."
    ),
    "subject_drug_role": (
        "High", "Reporter/case-processor assessment that may reflect downstream case interpretation."
    ),
    "subject_drug_route": (
        "Moderate", "Treatment-context characteristic, but route coding and missingness can reflect setting."
    ),
}

# These are intentionally not candidates for the preprocessor. Some are direct
# outcome definitions and others are close post-outcome proxies that would make
# a seemingly strong model scientifically misleading.
LEAKAGE_EXCLUSIONS = [
    "serious (target only)",
    "seriousnessdeath",
    "seriousnesshospitalization",
    "seriousnesslifethreatening",
    "seriousnessdisabling",
    "seriousnesscongenitalanomali",
    "seriousnessother",
    "patient.death and death-detail fields",
    "patient.reaction.reactionoutcome",
    "fulfillexpeditecriteria",
    "derived severity_score and all severity subtype counts",
]

OTHER_EXCLUSIONS = [
    "safetyreportid (identifier; retained only for deduplication)",
    "receivedate (split metadata only; not a predictor)",
    "raw drug names, indications, and reaction terms (high-cardinality text)",
    "company/report duplicate identifiers",
]

MIN_ROWS = 200
MIN_CLASS_COUNT = 20
RANDOM_STATE = 42


def feature_audit() -> pd.DataFrame:
    """Human-readable predictor audit, including the reduced-set decision."""
    rows = []
    for name in PREDICTOR_COLUMNS:
        proxy_risk, proxy_rationale = FEATURE_PROXY_AUDIT[name]
        rows.append({
            "predictor": name,
            "display_name": FEATURE_LABELS[name],
            "type": "numeric" if name in NUMERIC_FEATURES else "categorical",
            "proxy_risk": proxy_risk,
            "reduced_model": "Removed" if name in ADMINISTRATIVE_PROXY_FEATURES else "Retained",
            "audit_rationale": proxy_rationale,
        })
    return pd.DataFrame(rows)


def validate_training_data(frame: pd.DataFrame) -> None:
    missing = [c for c in [TARGET_COLUMN, DATE_COLUMN, *PREDICTOR_COLUMNS] if c not in frame]
    if missing:
        raise ValueError("Report-level data is missing required columns: " + ", ".join(missing))
    if len(frame) < MIN_ROWS:
        raise ValueError(
            f"At least {MIN_ROWS} reports are required; only {len(frame)} were returned. "
            "Increase the date range or report limit."
        )
    counts = frame[TARGET_COLUMN].value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("Both serious and non-serious reports are required for classification.")
    if int(counts.min()) < MIN_CLASS_COUNT:
        raise ValueError(
            f"Each class needs at least {MIN_CLASS_COUNT} reports; the minority class has "
            f"{int(counts.min())}. Increase the date range or report limit."
        )


def _has_both_classes(values: pd.Series) -> bool:
    return values.nunique(dropna=True) == 2 and int(values.value_counts().min()) >= 5


def _split_indices(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, str]:
    """Prefer an honest older-to-newer evaluation, with a documented fallback."""
    dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    valid_date_share = float(dates.notna().mean())

    if valid_date_share >= 0.95 and dates.nunique() >= 5:
        dated = frame.assign(_split_date=dates).sort_values("_split_date")
        unique_dates = np.array(sorted(dated["_split_date"].dropna().unique()))
        for fraction in (0.80, 0.75, 0.70, 0.85):
            cut_position = max(0, min(len(unique_dates) - 2, int(len(unique_dates) * fraction) - 1))
            cutoff = pd.Timestamp(unique_dates[cut_position])
            train_idx = dated.index[dated["_split_date"] <= cutoff].to_numpy()
            test_idx = dated.index[dated["_split_date"] > cutoff].to_numpy()
            if (
                len(train_idx) >= 100
                and len(test_idx) >= 40
                and _has_both_classes(frame.loc[train_idx, TARGET_COLUMN])
                and _has_both_classes(frame.loc[test_idx, TARGET_COLUMN])
            ):
                train_dates = dates.loc[train_idx]
                test_dates = dates.loc[test_idx]
                detail = (
                    "Chronological holdout: older reports were used for training and newer "
                    f"reports for testing. Train {train_dates.min().date()} to "
                    f"{train_dates.max().date()} ({len(train_idx):,} reports); test "
                    f"{test_dates.min().date()} to {test_dates.max().date()} "
                    f"({len(test_idx):,} reports). The cutoff date was {cutoff.date()}."
                )
                return train_idx, test_idx, detail

    indices = frame.index.to_numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=frame[TARGET_COLUMN],
    )
    reason = (
        "a chronological cutoff could not preserve adequate observations and both classes "
        "in each partition"
        if valid_date_share >= 0.95
        else f"only {valid_date_share:.1%} of reports had a usable receivedate"
    )
    detail = (
        "Stratified 80/20 holdout with random_state=42 was used because " + reason + 
        f". Train: {len(train_idx):,} reports; test: {len(test_idx):,} reports."
    )
    return np.asarray(train_idx), np.asarray(test_idx), detail


def _preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, categorical_features),
    ])


def _display_feature_name(name: str) -> str:
    raw_name = name.replace("numeric__", "").replace("categorical__", "")
    if raw_name.startswith("missingindicator_"):
        base_name = raw_name.removeprefix("missingindicator_")
        return f"{FEATURE_LABELS.get(base_name, base_name)} — missing"
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    for feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        prefix = f"{feature}_"
        if raw_name.startswith(prefix):
            value = raw_name[len(prefix):]
            if feature == "patient_sex":
                value = {"0": "Unknown", "1": "Male", "2": "Female"}.get(value, value)
            return f"{FEATURE_LABELS[feature]} — {value}"
    return raw_name.replace("_", " ").strip().capitalize()


def threshold_metrics(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute threshold-sensitive metrics from held-out labels and probabilities."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The classification threshold must be between 0 and 1.")

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if labels.shape != scores.shape:
        raise ValueError("Held-out labels and probabilities must have the same shape.")

    predictions = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]),
    }


def _evaluate(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
    at_default_threshold = threshold_metrics(y_test, probabilities, threshold=0.5)
    probability_distribution = pd.DataFrame({
        "actual_class": np.where(
            np.asarray(y_test, dtype=int) == 1,
            "Serious",
            "Non-serious",
        ),
        "predicted_probability": probabilities,
    })
    return {
        "name": name,
        "precision": at_default_threshold["precision"],
        "recall": at_default_threshold["recall"],
        "f1": at_default_threshold["f1"],
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "confusion_matrix": at_default_threshold["confusion_matrix"],
        "roc_curve": pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr}),
        "pr_curve": pd.DataFrame({"recall": recall_curve, "precision": precision_curve}),
        "held_out_labels": np.asarray(y_test, dtype=int),
        "held_out_probabilities": probabilities,
        "probability_distribution": probability_distribution,
    }


def _prepare_features(
    frame: pd.DataFrame,
    predictors: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """Select only the declared allowlist and normalize input dtypes."""
    features = frame[predictors].copy()
    for column in numeric_features:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in categorical_features:
        features[column] = features[column].map(
            lambda value: str(value) if pd.notna(value) else np.nan
        )
    return features


def _model_pipelines(
    numeric_features: List[str],
    categorical_features: List[str],
    y_train: pd.Series,
    xgb_classifier: Any,
) -> Dict[str, Pipeline]:
    class_counts = y_train.value_counts()
    scale_pos_weight = float(class_counts.get(0, 1) / max(1, class_counts.get(1, 1)))
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", _preprocessor(numeric_features, categorical_features)),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocess", _preprocessor(numeric_features, categorical_features)),
            ("classifier", xgb_classifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=2,
                tree_method="hist",
            )),
        ]),
    }


def _interpretability_frames(models: Dict[str, Pipeline]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logistic = models["Logistic Regression"]
    logistic_names = logistic.named_steps["preprocess"].get_feature_names_out()
    logistic_coefficients = logistic.named_steps["classifier"].coef_[0]
    coefficient_frame = pd.DataFrame({
        "feature": [_display_feature_name(name) for name in logistic_names],
        "coefficient": logistic_coefficients,
    })
    negative = coefficient_frame.nsmallest(10, "coefficient")
    positive = coefficient_frame.nlargest(10, "coefficient")
    coefficient_frame = pd.concat([negative, positive]).drop_duplicates("feature")
    coefficient_frame["association"] = np.where(
        coefficient_frame["coefficient"] >= 0,
        "Higher model score",
        "Lower model score",
    )
    coefficient_frame["coefficient_label"] = coefficient_frame["coefficient"].map(
        lambda value: f"{value:+.2f}"
    )

    xgboost = models["XGBoost"]
    xgb_names = xgboost.named_steps["preprocess"].get_feature_names_out()
    raw_importance = np.asarray(
        xgboost.named_steps["classifier"].feature_importances_, dtype=float
    )
    importance_total = float(raw_importance.sum())
    if importance_total > 0:
        importance_share = raw_importance / importance_total
    else:
        importance_share = raw_importance
    importance_frame = pd.DataFrame({
        "feature": [_display_feature_name(name) for name in xgb_names],
        "importance": raw_importance,
        "importance_share": importance_share,
    }).nlargest(15, "importance")
    importance_frame["importance_label"] = importance_frame["importance_share"].map(
        lambda value: f"{value:.1%}"
    )
    return (
        coefficient_frame.sort_values("coefficient"),
        importance_frame.sort_values("importance_share", ascending=False),
    )


def train_seriousness_models(frame: pd.DataFrame) -> Dict[str, Any]:
    """Train paired full/reduced models on one shared train/test split."""
    validate_training_data(frame)
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for Serious Outcome Classification. "
            "Install the updated requirements.txt."
        ) from exc

    y = frame[TARGET_COLUMN].astype(int).copy()
    train_idx, test_idx, split_description = _split_indices(frame)
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    feature_specs = {
        "Full": (PREDICTOR_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        "Reduced": (
            REDUCED_PREDICTOR_COLUMNS,
            REDUCED_NUMERIC_FEATURES,
            REDUCED_CATEGORICAL_FEATURES,
        ),
    }
    feature_set_results: Dict[str, Dict[str, Any]] = {}
    comparison_rows = []

    for feature_set, (predictors, numeric_features, categorical_features) in feature_specs.items():
        # Each variant receives identical row indices. Its separate pipeline is fit only
        # on the shared training rows, including all imputing, scaling, and encoding.
        features = _prepare_features(
            frame, predictors, numeric_features, categorical_features
        )
        X_train, X_test = features.loc[train_idx], features.loc[test_idx]
        models = _model_pipelines(
            numeric_features, categorical_features, y_train, XGBClassifier
        )
        evaluations = {}
        for model_name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            evaluations[model_name] = _evaluate(model_name, pipeline, X_test, y_test)
            evaluation = evaluations[model_name]
            comparison_rows.append({
                "feature_set": feature_set,
                "model": model_name,
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "f1": evaluation["f1"],
                "roc_auc": evaluation["roc_auc"],
                "pr_auc_average_precision": evaluation["average_precision"],
            })

        coefficients, importance = _interpretability_frames(models)
        feature_set_results[feature_set] = {
            "predictor_columns": list(predictors),
            "models": models,
            "evaluations": evaluations,
            "logistic_coefficients": coefficients,
            "xgboost_importance": importance,
        }

    comparison = pd.DataFrame(comparison_rows)
    delta_rows = []
    metric_columns = [
        "precision", "recall", "f1", "roc_auc", "pr_auc_average_precision"
    ]
    for model_name in comparison["model"].unique():
        model_rows = comparison[comparison["model"] == model_name].set_index("feature_set")
        delta_row = {"model": model_name}
        for metric in metric_columns:
            delta_row[f"{metric}_change"] = (
                model_rows.loc["Reduced", metric] - model_rows.loc["Full", metric]
            )
        delta_rows.append(delta_row)
    ablation_deltas = pd.DataFrame(delta_rows)

    full_result = feature_set_results["Full"]
    return {
        "feature_sets": feature_set_results,
        # Backward-compatible aliases keep existing threshold/curve rendering stable.
        "models": full_result["models"],
        "evaluations": full_result["evaluations"],
        "comparison": comparison,
        "ablation_deltas": ablation_deltas,
        "logistic_coefficients": full_result["logistic_coefficients"],
        "xgboost_importance": full_result["xgboost_importance"],
        "split_description": split_description,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "class_distribution": frame[TARGET_COLUMN].value_counts().sort_index(),
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "reduced_predictor_columns": list(REDUCED_PREDICTOR_COLUMNS),
        "removed_proxy_features": list(ADMINISTRATIVE_PROXY_FEATURES),
    }

