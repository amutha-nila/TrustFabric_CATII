from __future__ import annotations
import os, re, secrets, logging, tempfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
log = logging.getLogger('trustfabric')
MAX_TEXT_CHARS = int(os.getenv('TF_MAX_TEXT_CHARS', '20000'))
MAX_UPLOAD_BYTES = int(os.getenv('TF_MAX_UPLOAD_BYTES', str(25 * 1024 * 1024)))
API_KEY = os.getenv('TF_API_KEY', '').strip()
MEDIA_MODEL_ID = os.getenv('TF_MEDIA_MODEL_ID', 'umm-maybe/AI-image-detector').strip()
ALLOW_REMOTE_MEDIA = os.getenv('TF_ALLOW_MEDIA_UPLOAD', 'true').lower() == 'true'

app = FastAPI(title='TrustFabric API', version='1.0.0', description='Evidence verification, media risk analysis and documented privacy reports.')
# Local development origins only. For hosted deployment, set TF_CORS_ORIGINS explicitly.
origins = [x.strip() for x in os.getenv('TF_CORS_ORIGINS', 'http://127.0.0.1:8765,http://localhost:8765,https://chatgpt.com,https://chat.openai.com,https://gemini.google.com,chrome-extension://*').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=[x for x in origins if '*' not in x], allow_origin_regex=r'^chrome-extension://[a-p]{32}$', allow_credentials=False, allow_methods=['GET','POST','OPTIONS'], allow_headers=['Content-Type','X-TrustFabric-Key'])

class VerifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
class Evidence(BaseModel):
    title: str
    url: str
    snippet: str = ''
class ClaimResult(BaseModel):
    claim: str
    status: Literal['supported','contradicted','insufficient_evidence','error']
    explanation: str
    evidence: list[Evidence] = []
class VerifyResponse(BaseModel):
    overall_status: str
    claims: list[ClaimResult]
    message: str

@app.middleware('http')
async def limit_and_security(request: Request, call_next):
    if request.url.path.startswith('/api/') and request.method == 'POST':
        length = request.headers.get('content-length')
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 1_000_000:
            return __import__('fastapi').responses.JSONResponse({'detail':'Request body too large'}, status_code=413)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response

def require_key(provided: str | None):
    if API_KEY and not secrets.compare_digest(API_KEY, provided or ''):
        raise HTTPException(status_code=401, detail='Invalid or missing TrustFabric API key')

@app.get('/health')
def health():
    return {'status':'ok','service':'trustfabric-api','version':'1.0.0','media_model_configured':bool(MEDIA_MODEL_ID)}

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
_CLAIM_CUES = re.compile(r'\b(is|are|was|were|has|have|had|causes|caused|will|won|founded|invented|discovered|located|contains|increased|decreased|according to|percent|million|billion|always|never)\b', re.I)
def extract_claims(text: str) -> list[str]:
    parts = [re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', x).strip() for x in _SENTENCE_SPLIT.split(text.strip())]
    parts = [p for p in parts if len(p.split()) >= 5 and _CLAIM_CUES.search(p)]
    # Limit cost and avoid sending huge text to retrieval/model layers.
    return list(dict.fromkeys(parts))[:8]

async def search_evidence(query: str) -> list[Evidence]:
    """Use DuckDuckGo HTML results as a best-effort discovery layer; never treat search rank as truth."""
    try:
        import httpx
        from bs4 import BeautifulSoup
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, headers={'User-Agent':'TrustFabricResearch/1.0 (evidence retrieval)'}) as client:
            r = await client.post('https://html.duckduckgo.com/html/', data={'q': query[:350]})
            r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        out=[]
        for item in soup.select('.result')[:5]:
            a=item.select_one('.result__a'); snippet=item.select_one('.result__snippet')
            if not a or not a.get('href'): continue
            url=a['href']
            if not url.startswith('http'): continue
            out.append(Evidence(title=a.get_text(' ', strip=True)[:250], url=url, snippet=snippet.get_text(' ',strip=True)[:600] if snippet else ''))
        return out
    except Exception as exc:
        log.warning('Evidence retrieval unavailable: %s', type(exc).__name__)
        return []

_nli = None
def get_nli():
    global _nli
    if _nli is not None: return _nli
    model_id = os.getenv('TF_NLI_MODEL_ID', 'cross-encoder/nli-deberta-v3-small')
    try:
        from transformers import pipeline
        _nli = pipeline('text-classification', model=model_id, tokenizer=model_id, truncation=True, device=-1)
        return _nli
    except Exception as exc:
        log.warning('NLI model unavailable: %s', type(exc).__name__)
        _nli = False
        return None

def nli_label(premise: str, hypothesis: str) -> tuple[str,float] | None:
    clf=get_nli()
    if not clf: return None
    try:
        result=clf({'text': premise[:1200], 'text_pair': hypothesis[:500]})[0]
        label=result['label'].lower(); score=float(result['score'])
        if 'entail' in label: return 'supported',score
        if 'contradict' in label: return 'contradicted',score
        return 'insufficient_evidence',score
    except Exception as exc:
        log.warning('NLI inference failed: %s', type(exc).__name__); return None

@app.post('/api/v1/verify', response_model=VerifyResponse)
async def verify(req: VerifyRequest, x_trustfabric_key: str | None = Header(default=None)):
    require_key(x_trustfabric_key)
    claims=extract_claims(req.text)
    if not claims:
        return VerifyResponse(overall_status='insufficient_evidence', claims=[], message='No clear factual claims were extracted. This is not a judgment that the answer is true or false.')
    results=[]
    for claim in claims:
        evidence=await search_evidence(claim)
        comparisons=[]
        for ev in evidence[:3]:
            if ev.snippet:
                nli=nli_label(ev.snippet, claim)
                if nli: comparisons.append((nli[0],nli[1],ev))
        status='insufficient_evidence'; explanation='Relevant search results were not available or did not provide enough evidence to classify this claim.'
        if comparisons:
            best=max(comparisons,key=lambda x:x[1])
            if best[1] >= 0.72:
                status=best[0]; explanation=f"A local NLI model judged one search snippet as {status.replace('_',' ')} (score {best[1]:.2f}). This score is not a probability of truth; inspect the source." 
            else: explanation='Available evidence was ambiguous; manual source review is recommended.'
        elif evidence:
            explanation='Search results were found, but the entailment model was unavailable or snippets were insufficient. No truth claim is made.'
        results.append(ClaimResult(claim=claim,status=status,explanation=explanation,evidence=evidence))
    statuses={r.status for r in results}
    overall='contradicted' if 'contradicted' in statuses else ('supported' if statuses=={'supported'} else 'insufficient_evidence')
    return VerifyResponse(overall_status=overall,claims=results,message='Automated evidence triage only. Search results can be incomplete, stale, or misleading; this is not a guarantee of truth.')

_media_classifier=None
_media_load_error=None
def get_media_classifier():
    global _media_classifier,_media_load_error
    if _media_classifier is not None: return _media_classifier
    if _media_load_error: return None
    try:
        from transformers import pipeline
        _media_classifier=pipeline('image-classification', model=MEDIA_MODEL_ID, device=-1)
        return _media_classifier
    except Exception as exc:
        _media_load_error=type(exc).__name__; log.exception('Unable to load media model')
        return None

def classify_image(image: Image.Image):
    clf=get_media_classifier()
    if not clf: raise HTTPException(status_code=503, detail=f'Media model could not be loaded ({_media_load_error or "model unavailable"}). Check internet/model configuration and retry.')
    preds=clf(image.convert('RGB'))
    # Model label vocabularies vary; map only labels that clearly signal AI/synthetic.
    normalized=[(str(p.get('label','')).lower(),float(p.get('score',0))) for p in preds]
    ai=max([s for l,s in normalized if any(k in l for k in ('artificial','generated','synthetic','fake','ai-generated','diffusion'))] or [0.0])
    real=max([s for l,s in normalized if any(k in l for k in ('real','human','natural','authentic'))] or [0.0])
    if ai > real and ai >= 0.60: label='likely_ai_generated'; score=ai
    elif real > ai and real >= 0.60: label='model_predicts_real'; score=real
    else: label='inconclusive'; score=max(ai,real)
    return label,score,normalized

@app.post('/api/v1/media/analyze')
async def analyze_media(file: UploadFile = File(...), x_trustfabric_key: str | None = Header(default=None)):
    require_key(x_trustfabric_key)
    if not ALLOW_REMOTE_MEDIA: raise HTTPException(status_code=403,detail='Media upload is disabled by server configuration')
    content_type=(file.content_type or '').lower()
    if not (content_type.startswith('image/') or content_type in {'video/mp4','video/webm','video/quicktime'}):
        raise HTTPException(status_code=415,detail='Only supported image and video files are accepted')
    raw=await file.read(MAX_UPLOAD_BYTES+1)
    if len(raw)>MAX_UPLOAD_BYTES: raise HTTPException(status_code=413,detail='File exceeds configured upload limit')
    if not raw: raise HTTPException(status_code=400,detail='Empty file')
    is_video=content_type.startswith('video/')
    frames=[]
    try:
        if not is_video:
            try: frames=[Image.open(__import__('io').BytesIO(raw)).convert('RGB')]
            except (UnidentifiedImageError, OSError): raise HTTPException(status_code=415,detail='File content is not a readable image')
        else:
            # Validate and sample at most 8 frames; temporary file is removed in all cases.
            suffix={ 'video/mp4':'.mp4','video/webm':'.webm','video/quicktime':'.mov'}.get(content_type,'.video')
            with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as tmp:
                tmp.write(raw); path=tmp.name
            cap=None
            try:
                cap=cv2.VideoCapture(path)
                if not cap.isOpened(): raise HTTPException(status_code=415,detail='Could not decode video. Supported: MP4, WebM, MOV when OpenCV codec support is available.')
                count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if count < 1: raise HTTPException(status_code=415,detail='Video contains no readable frames')
                indexes=sorted(set(int(i) for i in np.linspace(0,max(0,count-1),min(8,count))))
                for idx in indexes:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,frame=cap.read()
                    if ok: frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            finally:
                if cap: cap.release()
                try: os.unlink(path)
                except OSError: pass
    finally:
        await file.close()
    if not frames: raise HTTPException(status_code=415,detail='No frames could be analysed')
    preds=[]
    for frame in frames:
        label,score,rawpred=classify_image(frame)
        preds.append((label,score,rawpred))
    ai_votes=sum(1 for l,_,_ in preds if l=='likely_ai_generated')
    real_votes=sum(1 for l,_,_ in preds if l=='model_predicts_real')
    if ai_votes > real_votes: final='likely_ai_generated'; score=float(np.mean([s for l,s,_ in preds if l=='likely_ai_generated']))
    elif real_votes > ai_votes: final='model_predicts_real'; score=float(np.mean([s for l,s,_ in preds if l=='model_predicts_real']))
    else: final='inconclusive'; score=float(np.mean([s for _,s,_ in preds]))
    media_type='video' if is_video else 'image'
    return {'media_type':media_type,'label':final,'score':round(score,4),'frames_analyzed':len(frames) if is_video else None,'summary':f'The configured image classifier returned {final.replace("_"," ")} for this {media_type}.','limitations':'This is a model estimate, not forensic proof. Video analysis samples frames and does not establish temporal consistency, provenance, or authenticity. Results may fail on new generators, edited media, screenshots, or compressed files. The uploaded file was processed in memory and temporary video files are removed after analysis; do not use sensitive media unless you accept sending it to this backend.','model':MEDIA_MODEL_ID}

POLICIES={
 'chatgpt.com': {'name':'ChatGPT','url':'https://openai.com/policies/row-privacy-policy/','controls':'https://help.openai.com/en/articles/7730893-data-controls-faq','summary':'Review OpenAI’s current privacy policy and ChatGPT Data Controls for your account and region.'},
 'chat.openai.com': {'name':'ChatGPT','url':'https://openai.com/policies/row-privacy-policy/','controls':'https://help.openai.com/en/articles/7730893-data-controls-faq','summary':'Review OpenAI’s current privacy policy and ChatGPT Data Controls for your account and region.'},
 'gemini.google.com': {'name':'Google Gemini','url':'https://support.google.com/gemini/answer/13594961','controls':'https://myaccount.google.com/data-and-privacy','summary':'Review Google’s current Gemini Apps Privacy Hub and activity controls; handling may depend on settings and account type.'}
}
@app.get('/api/v1/privacy/platform')
def privacy_platform(host: str, x_trustfabric_key: str | None = Header(default=None)):
    require_key(x_trustfabric_key)
    host=host.lower().split(':')[0]
    if host not in POLICIES: raise HTTPException(status_code=404,detail='No maintained policy profile for this website. TrustFabric does not infer a platform policy without a known source.')
    p=POLICIES[host]
    return {'platform':p['name'],'summary':p['summary'],'risk_level':'info','findings':[
        {'level':'review','topic':'Data use and retention','detail':'Read the current policy for how prompts, uploads, and generated content are handled, including retention periods and possible human review. This extension does not independently inspect provider systems.','source':p['url']},
        {'level':'review','topic':'Model improvement and training','detail':'Check the controls and policy that apply to your account, product, region, and plan. Do not assume settings are identical across consumer apps and APIs.','source':p['controls']},
        {'level':'review','topic':'Deletion and activity controls','detail':'Review available activity history, deletion, and privacy controls before submitting sensitive material.','source':p['controls']}
    ],'disclaimer':'Policy summary is a navigation aid, not legal advice or proof of misuse.'}
