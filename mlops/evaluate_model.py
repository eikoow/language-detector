"""Download 100 FLEURS rows and compare two bounded model improvements."""

from __future__ import annotations

import json
import re
import unicodedata
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from training_data import EXAMPLES


DATASET = "google/fleurs"
DATASET_LICENSE = "CC BY 4.0"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
LANGUAGE_CONFIGS = {
    "ko": "ko_kr",
    "en": "en_us",
    "my": "my_mm",
    "ja": "ja_jp",
    "zh": "zh_cn",
    "fr": "fr_fr",
    "es": "es_419",
    "de": "de_de",
}
COUNTS = {code: 12 if index < 4 else 13 for index, code in enumerate(LANGUAGE_CONFIGS)}


def prepare_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def fetch_rows(code: str, config: str, length: int) -> list[dict[str, str]]:
    query = urlencode({"dataset": DATASET, "config": config, "split": "train", "offset": 0, "length": length})
    request = Request(f"{ROWS_URL}?{query}", headers={"User-Agent": "language-detector-mlops/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return [{"text": row["row"]["sentence"], "label": code} for row in payload["rows"]]


def make_model(ngram_range: tuple[int, int], alpha: float, normalize: bool) -> Pipeline:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, sublinear_tf=True)
    model = Pipeline([("tfidf", vectorizer), ("classifier", MultinomialNB(alpha=alpha))])
    return model


def fit_predict(model: Pipeline, train: list[dict[str, str]], rows: list[dict[str, str]], normalize: bool) -> list[str]:
    text = lambda item: prepare_text(item["text"]) if normalize else item["text"]
    model.fit([text(item) for item in train], [item["label"] for item in train])
    return model.predict([text(item) for item in rows])


def main() -> None:
    rows = [item for code, config in LANGUAGE_CONFIGS.items() for item in fetch_rows(code, config, COUNTS[code])]
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 downloaded rows, got {len(rows)}")

    # The fixed quotas make the 30/70 split reproducible and keep every
    # language represented in both partitions. Evaluation rows never enter
    # tuning or model fitting.
    tuning_quota = {"ko": 4, "en": 4, "my": 4, "ja": 4, "zh": 4, "fr": 4, "es": 3, "de": 3}
    tuning = []
    evaluation = []
    for code in LANGUAGE_CONFIGS:
        language_rows = [item for item in rows if item["label"] == code]
        tuning.extend(language_rows[:tuning_quota[code]])
        evaluation.extend(language_rows[tuning_quota[code]:])
    if len(tuning) != 30 or len(evaluation) != 70:
        raise RuntimeError(f"Expected a 30/70 split, got {len(tuning)}/{len(evaluation)}")
    baseline_train = [{"text": text, "label": code} for code, samples in EXAMPLES.items() for text in samples]
    baseline = make_model((2, 5), 0.1, False)
    baseline_predictions = fit_predict(baseline, baseline_train, evaluation, False)
    baseline_accuracy = accuracy_score([item["label"] for item in evaluation], baseline_predictions)

    attempts = [
        ("attempt-1 normalize + tuning rows", (2, 5), 0.1, True),
        ("attempt-2 short-text ngrams", (1, 5), 0.1, True),
    ]
    results = []
    for name, ngram_range, alpha, normalize in attempts:
        model = make_model(ngram_range, alpha, normalize)
        train = baseline_train + tuning
        predictions = fit_predict(model, train, evaluation, normalize)
        results.append((name, accuracy_score([item["label"] for item in evaluation], predictions)))

    best_name, best_accuracy = max(results, key=lambda result: result[1])
    print(f"Dataset: {DATASET} ({DATASET_LICENSE}; {DATASET_LICENSE_URL})")
    print("Downloaded: 100 rows; tuning: 30; final evaluation: 70")
    print("| Model | Accuracy on same 70-row evaluation set |")
    print("|---|---:|")
    print(f"| baseline v1 | {baseline_accuracy:.3f} |")
    for name, accuracy in results:
        print(f"| {name} | {accuracy:.3f} |")
    print(f"Selected: {best_name}; final model version: v2")


if __name__ == "__main__":
    main()
