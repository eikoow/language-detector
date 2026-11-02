# Local Language Detector

## Run from PowerShell

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The model trains at server startup using the included examples: character n-gram TF-IDF features and Multinomial Naive Bayes. Results are predictions, not guaranteed translations or identifications.
