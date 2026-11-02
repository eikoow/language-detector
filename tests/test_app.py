from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_model_info():
    data = client.get('/model-info').json()
    assert data['training_examples'] == 24
    assert len(data['languages']) == 8


def test_detect_returns_prediction_and_top_three():
    response = client.post('/detect', json={'text': 'Hello, how are you today?'})
    assert response.status_code == 200
    data = response.json()
    assert data['language'] == 'English'
    assert data['is_prediction'] is True
    assert len(data['top_languages']) == 3


def test_detect_rejects_blank_text():
    response = client.post('/detect', json={'text': '   '})
    assert response.status_code == 422
