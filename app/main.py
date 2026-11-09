from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from training_data import EXAMPLES as TRAINING_EXAMPLES


LANGUAGES = {
    "ko": "Korean",
    "en": "English",
    "my": "Burmese (Myanmar)",
    "ja": "Japanese",
    "zh": "Chinese",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
}

EXAMPLES = {
    "ko": ["안녕하세요 오늘 날씨가 정말 좋습니다.", "저는 한국어를 공부하고 있습니다.", "맛있는 저녁을 먹고 공원에 갔어요."],
    "en": ["Hello, the weather is beautiful today.", "I am learning about language detection.", "Please enjoy your evening and have a good day."],
    "my": ["မင်္ဂလာပါ၊ ဒီနေ့ ရာသီဥတု အရမ်းကောင်းပါတယ်။", "ကျွန်တော် မြန်မာစာကို လေ့လာနေပါတယ်။", "ညစာစားပြီး ပန်းခြံကို သွားခဲ့တယ်။"],
    "ja": ["こんにちは、今日はとても良い天気です。", "私は日本語を勉強しています。", "おいしい夕食を食べて公園へ行きました。"],
    "zh": ["你好，今天天气真的很好。", "我正在学习语言检测。", "吃过晚饭以后我们去了公园。"],
    "fr": ["Bonjour, le temps est très agréable aujourd'hui.", "J'apprends la détection automatique des langues.", "Nous avons mangé un bon dîner au restaurant."],
    "es": ["Hola, el tiempo es muy agradable hoy.", "Estoy aprendiendo sobre la detección de idiomas.", "Después de cenar fuimos al parque."],
    "de": ["Hallo, das Wetter ist heute wirklich schön.", "Ich lerne etwas über Spracherkennung.", "Nach dem Abendessen gingen wir in den Park."],
}

BASE_DIR = Path(__file__).resolve().parent
model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)),
    ("classifier", MultinomialNB(alpha=0.1)),
])
training_texts = [text for samples in TRAINING_EXAMPLES.values() for text in samples]
training_labels = [language for language, samples in TRAINING_EXAMPLES.items() for _ in samples]
model.fit(training_texts, training_labels)

app = FastAPI(title="Language Detector", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "algorithm": "Character n-gram TF-IDF + Multinomial Naive Bayes",
        "training_examples": len(training_texts),
        "languages": list(LANGUAGES.values()),
        "prediction_notice": "Results are predictions from a small startup-trained model.",
    }


@app.post("/detect")
def detect(request: DetectRequest) -> dict[str, Any]:
    text = request.text.strip()
    if not text:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Please enter some text to analyze.")
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:3]
    top_language, top_probability = ranked[0]
    return {
        "language": LANGUAGES[top_language],
        "language_code": top_language,
        "confidence": round(float(top_probability), 4),
        "top_languages": [
            {"language": LANGUAGES[code], "language_code": code, "probability": round(float(prob), 4)}
            for code, prob in ranked
        ],
        "is_prediction": True,
        "notice": "This result is a prediction and may be incorrect, especially for short text.",
    }
