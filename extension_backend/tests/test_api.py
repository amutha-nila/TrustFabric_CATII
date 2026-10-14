from fastapi.testclient import TestClient
from app.main import app, extract_claims
client=TestClient(app)

def test_health():
    r=client.get('/health'); assert r.status_code==200 and r.json()['status']=='ok'

def test_claim_extraction():
    claims=extract_claims('The Earth is a planet in the Solar System. Hello there.')
    assert len(claims)==1 and 'Earth' in claims[0]

def test_verify_rejects_empty_text():
    r=client.post('/api/v1/verify',json={'text':''})
    assert r.status_code==422

def test_unknown_privacy_profile():
    r=client.get('/api/v1/privacy/platform?host=example.invalid')
    assert r.status_code==404

def test_supported_chat_origin_cors_preflight():
    r=client.options('/api/v1/verify',headers={
        'Origin':'https://chatgpt.com',
        'Access-Control-Request-Method':'POST',
        'Access-Control-Request-Headers':'content-type',
    })
    assert r.status_code==200
    assert r.headers['access-control-allow-origin']=='https://chatgpt.com'
