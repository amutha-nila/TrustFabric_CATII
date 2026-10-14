from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from joblib import load
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("TRUSTFABRIC_MODEL_PATH", ROOT / "outputs/models/trustfabric_model.joblib"))
app = FastAPI(
    title="TrustFabric ML API",
    version="1.0.0",
    description="Research prototype for three-class claim-credibility triage and evidence-based verification.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chatgpt.com",
        "https://chat.openai.com",
        "https://gemini.google.com",
    ],
    allow_origin_regex=r"^chrome-extension://[a-p]{32}$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-TrustFabric-Key"],
)
logger = logging.getLogger(__name__)
model = None
_nli_model = None
_nli_error = None
_SEARCH_CACHE: OrderedDict[str, tuple[float, list[EvidenceItem], str | None]] = OrderedDict()
_SEARCH_CACHE_LOCK = threading.RLock()
_SEARCH_CACHE_TTL_SECONDS = 600
_SEARCH_CACHE_MAX_ENTRIES = 128
try:
    _SEARCH_RESULTS_LIMIT = max(1, min(8, int(os.getenv("TRUSTFABRIC_EVIDENCE_RESULTS", "5"))))
except ValueError:
    logger.warning("Invalid TRUSTFABRIC_EVIDENCE_RESULTS; using the default of 5.")
    _SEARCH_RESULTS_LIMIT = 5
_PRIMARY_SOURCE_DOMAINS = (
    "nasa.gov", "nih.gov", "cdc.gov", "noaa.gov", "nps.gov",
    "nobelprize.org", "who.int", "un.org", "europa.eu",
)
_RELIABLE_SOURCE_DOMAINS = (
    "apnews.com", "bbc.com", "bbc.co.uk", "britannica.com", "nature.com",
    "reuters.com", "science.org", "scientificamerican.com", "plos.org",
    "nejm.org", "cell.com", "nationalgeographic.com", "thehindu.com",
    "timesofindia.indiatimes.com",
)
_NLI_SCORE_THRESHOLD = 0.80
_nli_lock = threading.Lock()
_nli_inference_lock = threading.Lock()


class PredictRequest(BaseModel):
    text: str = Field(min_length=5, max_length=10000, description="A short factual statement to assess")


class EvidenceItem(BaseModel):
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None


class ClaimResult(BaseModel):
    claim: str
    claim_type: Literal["factual", "subjective"] = "factual"
    status: Literal[
        "supported",
        "contradicted",
        "mixed_or_conflicting_evidence",
        "insufficient_evidence",
    ]
    explanation: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    evidence_status: Literal["available", "no_results", "unavailable", "irrelevant", "inconclusive", "not_needed"]


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000, description="AI answer or claim text to verify")

    @field_validator("text")
    @classmethod
    def strip_and_reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must contain at least one non-whitespace character.")
        return value


class VerifyResponse(BaseModel):
    overall_status: Literal[
        "supported",
        "contradicted",
        "mixed_or_conflicting_evidence",
        "insufficient_evidence",
    ]
    claims: list[ClaimResult]
    message: str
    evidence_retrieval_status: Literal["available", "no_results", "unavailable", "partial", "not_needed"]


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_path": str(MODEL_PATH)}


@app.get("/api/v1/health")
def health_v1():
    return health()


@app.get("/api/metrics")
def metrics():
    metrics_path = ROOT / "outputs/metrics/benchmark.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Training metrics have not been generated yet. Run: python -m trustfabric.train")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((ROOT / "app" / "index.html").read_text(encoding="utf-8"))


@app.post("/api/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained yet. Run: python -m trustfabric.train")
    probs = model.predict_proba([req.text])[0]
    classes = list(model.classes_)
    idx = int(probs.argmax())
    return {
        "prediction": str(classes[idx]),
        "confidence_score": round(float(probs[idx]), 4),
        "class_probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, probs)},
        "interpretation": "A dataset-learned language pattern, not a fact check. Consult independent evidence; the model may be wrong or biased.",
    }


def _leading_subject(statement: str) -> str | None:
    match = re.match(
        r"^\s*(.+?)\s+(?:is|are|was|were|has|have|had|does|do|did|can|could|will|would|"
        r"won|received|became|remains|located|founded|born|died|contains|includes|occurred|"
        r"happened|measures|weighs|costs|reached|caused|created|produced|launched|opened|"
        r"closed|served|works|wrote|built|led|runs|speaks|operates|affects|orbits|visited|"
        r"lived|stood|lasted|grew|released|boils|announced|approved|signed|passed|elected|"
        r"appointed|invented|discovered)\b",
        statement,
        re.I,
    )
    return match.group(1).strip(" ,.;:") if match else None


def extract_claims(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    raw = re.sub(r"```[\s\S]*?```", " ", raw)
    raw = re.sub(r"`([^`]+)`", r"\1", raw)
    chunks = re.split(r"\n+|(?<=[.!?])\s+", raw)
    claims = []
    prior_subject = None
    for chunk in chunks:
        statement = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", chunk).strip()
        if not statement or _is_nonfactual_answer_text(statement):
            continue
        if prior_subject and re.match(r"^(?:he|she|they|it|this|that|these|those)\b", statement, re.I):
            statement = re.sub(
                r"^(?:he|she|they|it|this|that|these|those)\b",
                prior_subject,
                statement,
                count=1,
                flags=re.I,
            )
        clauses = re.split(
            r"(?:[,;]\s*\b(?:and|but)\b\s*|[,;]\s*|\s+\b(?:and|but)\b\s*)"
            r"(?=(?:he|she|they|it|this|that|these|those)\b|"
            r"[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,4}\s+"
            r"(?:is|are|was|were|has|have|had|won|received|became|remains|located|founded|"
            r"born|died|contains|includes|occurred|happened|measures|weighs|costs|reached|"
            r"caused|created|produced|launched|opened|closed|served|works|wrote|built|led|"
            r"runs|speaks|operates|affects|orbits|visited|lived|stood|lasted|grew|released|"
            r"boils|announced|approved|signed|passed|elected|appointed|invented|discovered)\b)",
            statement,
            flags=re.I,
        )
        subject = _leading_subject(clauses[0]) if len(clauses) > 1 else None
        if clauses and not _is_nonfactual_answer_text(clauses[0]):
            claims.append(clauses[0].strip())
            prior_subject = _leading_subject(clauses[0]) or prior_subject
        for clause in clauses[1:]:
            clause = clause.strip()
            if subject and re.match(r"^(?:he|she|they|it|this|that)\b", clause, re.I):
                clause = re.sub(
                    r"^(?:he|she|they|it|this|that)\b",
                    subject,
                    clause,
                    count=1,
                    flags=re.I,
                )
            if not _is_nonfactual_answer_text(clause):
                claims.append(clause)
    dedup = []
    seen = set()
    for claim in claims:
        if (
            claim.endswith("?")
            or len(claim.split()) < 4
            or not re.search(r"[A-Za-z]", claim)
            or not re.search(
                r"\b(?:is|are|was|were|has|have|had|does|do|did|can|could|will|would|"
                r"won|received|became|remains|located|founded|born|died|contains|includes|"
                r"occurred|happened|measures|weighs|costs|reached|caused|created|produced|"
                r"launched|opened|closed|served|works|wrote|built|led|runs|speaks|operates|"
                r"affects|orbits|reached|visited|lived|stood|lasted|grew|released|boils|"
                r"announced|approved|signed|passed|elected|appointed|invented|discovered)\b|"
                r"\b\d{4}\b",
                claim,
                re.I,
            )
            or classify_claim(claim) == "subjective"
        ):
            continue
        key = re.sub(r"\s+", " ", claim).lower()
        if key not in seen:
            dedup.append(claim)
            seen.add(key)
    return dedup[:8]


_NONFACTUAL_ANSWER_PATTERNS = (
    r"^(?:hello|hi|hey|good morning|good afternoon|good evening)\b",
    r"^(?:how can i help|what would you like|is there anything else)\b",
    r"^(?:i hope this helps|hope this helps|let me know|feel free to ask)\b",
    r"^(?:please note|in summary|to summarize|here(?:'s| is) (?:the answer|a summary))\b",
    r"^(?:you should|you can|try to|follow these|first,|next,|finally,)\b",
    r"^(?:i think|i believe|in my opinion|personally)\b",
    r"^(?:as an ai|i cannot|i can(?:not)? provide|i don't have access)\b",
    r"^(?:debug(?:ging)?|copilot|trustfabric|console|stack trace|error:|warning:)\b",
)


def _is_nonfactual_answer_text(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or cleaned.endswith("?"):
        return True
    if any(re.search(pattern, cleaned, re.I) for pattern in _NONFACTUAL_ANSWER_PATTERNS):
        return True
    if cleaned.startswith(("{", "}", "$ ", ">>>", "Traceback")):
        return True
    return False


_SUBJECTIVE_CLAIM_PATTERN = re.compile(
    r"\b(?:in my opinion|i think|i believe|is|are|was|were|seems|looks)\s+"
    r"(?:an?\s+)?(?:great|good|bad|wonderful|terrible|excellent|awful|beautiful|"
    r"amazing|inspiring|boring|kind|brilliant|lovely|horrible)\b"
    r"|\b(?:best|worst|greatest|most wonderful|most beautiful)\b",
    re.I,
)


def classify_claim(claim: str) -> Literal["factual", "subjective"]:
    return "subjective" if _SUBJECTIVE_CLAIM_PATTERN.search(claim) else "factual"


_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "in",
    "into", "is", "it", "its", "of", "on", "or", "the", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "with",
}


def build_search_queries(claim: str) -> list[str]:
    cleaned = re.sub(r"^\s*(?:claim|statement|according to)\s*:\s*", "", claim, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n.!?")
    if not cleaned:
        return []
    keywords = [
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", cleaned)
        if token.casefold() not in _QUERY_STOP_WORDS
    ]
    focused = " ".join(keywords[:20])
    queries = [cleaned[:350]]
    if focused and focused.casefold() != cleaned.casefold():
        queries.append(focused[:350])
    return list(dict.fromkeys(queries))


def _unwrap_search_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if (host == "duckduckgo.com" or host.endswith(".duckduckgo.com")) and parsed.path == "/l/":
        return parse_qs(parsed.query).get("uddg", [""])[0]
    if url.startswith("//"):
        return "https:" + url
    return url


def _parse_search_results(html: str) -> list[EvidenceItem]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[EvidenceItem] = []
    for item in soup.select(".result"):
        anchor = item.select_one(".result__a")
        if not anchor or not anchor.get("href"):
            continue
        url = _unwrap_search_url(anchor["href"])
        parsed_url = urlparse(url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or parsed_url.username
            or parsed_url.password
        ):
            continue
        snippet = item.select_one(".result__snippet")
        date_node = item.select_one("time[datetime], .result__timestamp")
        published_at = None
        if date_node:
            published_at = date_node.get("datetime") or date_node.get_text(" ", strip=True) or None
        results.append(
            EvidenceItem(
                title=anchor.get_text(" ", strip=True)[:250] or "Source",
                url=url,
                snippet=snippet.get_text(" ", strip=True)[:600] if snippet else "",
                published_at=published_at,
            )
        )
        if len(results) >= _SEARCH_RESULTS_LIMIT:
            break
    return results


def _search_page_error(html: str) -> str | None:
    if not html.strip():
        return "The evidence search provider returned an empty response."
    if re.search(r"\b(?:captcha|unusual traffic|robot check|bot detection)\b", html, re.I):
        return "The evidence search provider returned a bot-check page instead of results."
    soup = BeautifulSoup(html, "html.parser")
    if soup.select_one(".no-results, #no-results") or re.search(r"\bno results found\b", soup.get_text(" ", strip=True), re.I):
        return None
    if not soup.select_one(".result"):
        return "The evidence search provider returned an unrecognized results page."
    return None


async def search_evidence(query: str) -> tuple[list[EvidenceItem], str | None]:
    queries = build_search_queries(query)
    if not queries:
        return [], None
    cache_key = re.sub(r"\s+", " ", "\n".join(queries)).casefold()
    now = time.monotonic()
    with _SEARCH_CACHE_LOCK:
        cached = _SEARCH_CACHE.get(cache_key)
        if cached:
            expires_at, cached_items, cached_error = cached
            if expires_at > now:
                _SEARCH_CACHE.move_to_end(cache_key)
                return [item.model_copy() for item in cached_items], cached_error
            del _SEARCH_CACHE[cache_key]

    successful_responses = []
    errors: list[str] = []
    try:
        timeout = httpx.Timeout(5.0, connect=2.0)
        limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            follow_redirects=True,
            headers={"User-Agent": "TrustFabricResearch/1.0 (evidence retrieval)"},
        ) as client:
            async def fetch(search_query: str) -> None:
                try:
                    response = await client.get(
                        "https://html.duckduckgo.com/html/",
                        params={"q": search_query[:350]},
                    )
                    response.raise_for_status()
                    successful_responses.append(response.text)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429:
                        errors.append("The evidence search provider rate-limited this request (HTTP 429).")
                    else:
                        errors.append(f"The evidence search provider returned HTTP {exc.response.status_code}.")
                except httpx.TimeoutException:
                    errors.append("The evidence search provider did not respond before the timeout.")
                except httpx.RequestError:
                    errors.append("The evidence search provider could not be reached.")

            await asyncio.gather(*(fetch(search_query) for search_query in queries))
    except httpx.TimeoutException:
        errors.append("The evidence search provider did not respond before the timeout.")
    except httpx.RequestError:
        errors.append("The evidence search provider could not be reached.")

    by_url: dict[str, EvidenceItem] = {}
    for html in successful_responses:
        page_error = _search_page_error(html)
        if page_error:
            errors.append(page_error)
            continue
        for item in _parse_search_results(html):
            by_url.setdefault(item.url, item)
    items = _prioritize_search_results(list(by_url.values()))
    provider_error = "; ".join(dict.fromkeys(errors)) if errors else None
    result = items, provider_error

    cache_seconds = _SEARCH_CACHE_TTL_SECONDS if successful_responses and not errors else 20
    with _SEARCH_CACHE_LOCK:
        _SEARCH_CACHE[cache_key] = (time.monotonic() + cache_seconds, result[0], result[1])
        _SEARCH_CACHE.move_to_end(cache_key)
        while len(_SEARCH_CACHE) > _SEARCH_CACHE_MAX_ENTRIES:
            _SEARCH_CACHE.popitem(last=False)
    return [item.model_copy() for item in result[0]], result[1]


def _source_quality(item: EvidenceItem) -> tuple[float, bool]:
    host = (urlparse(item.url).hostname or "").lower().rstrip(".")
    if (
        host.endswith((".gov", ".gov.uk", ".gov.au", ".gc.ca", ".int"))
        or host == "europa.eu"
        or any(host == domain or host.endswith("." + domain) for domain in _PRIMARY_SOURCE_DOMAINS)
    ):
        return 0.95, True
    if any(host == domain or host.endswith("." + domain) for domain in _RELIABLE_SOURCE_DOMAINS):
        return 0.88, False
    if host.endswith(".edu"):
        return 0.82, False
    return 0.55, False


def _prioritize_search_results(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return sorted(
        items,
        key=lambda item: (
            _source_quality(item)[0],
            bool(item.snippet),
            len(item.snippet),
        ),
        reverse=True,
    )[:_SEARCH_RESULTS_LIMIT]


def _nli_label_name(label: str) -> str | None:
    normalized = re.sub(r"[^a-z]", "", label.casefold())
    if "contradict" in normalized:
        return "contradicted"
    if "entail" in normalized or "support" in normalized:
        return "supported"
    if "neutral" in normalized:
        return "insufficient_evidence"
    return None


def _get_nli_model():
    global _nli_error, _nli_model
    if _nli_model is not None:
        return _nli_model if _nli_model is not False else None
    with _nli_lock:
        if _nli_model is not None:
            return _nli_model if _nli_model is not False else None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        except ImportError:
            _nli_error = (
                "The NLI comparison dependency is not installed. Install "
                "requirements-nli.txt to enable general evidence comparison."
            )
            logger.info("%s", _nli_error)
            _nli_model = False
            return None
        try:
            model_id = os.getenv(
                "TRUSTFABRIC_NLI_MODEL",
                "cross-encoder/nli-deberta-v3-small",
            )
            local_only = os.getenv("TRUSTFABRIC_NLI_LOCAL_ONLY", "false").lower() == "true"
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
            model_config = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                local_files_only=local_only,
            )
            _nli_model = pipeline(
                "text-classification",
                model=model_config,
                tokenizer=tokenizer,
                truncation=True,
                device=-1,
            )
            _nli_error = None
            logger.info("Loaded NLI comparison model %s.", model_id)
            return _nli_model
        except Exception as exc:
            _nli_error = (
                "The NLI model could not be loaded. Check the model identifier, "
                "network access, and available disk/memory."
            )
            logger.warning("%s (%s)", _nli_error, type(exc).__name__)
            _nli_model = False
            return None


def nli_label(premise: str, hypothesis: str) -> tuple[str, float] | None:
    clf = _get_nli_model()
    if not clf:
        return None
    try:
        with _nli_inference_lock:
            predictions = clf(
                {"text": premise[:1200], "text_pair": hypothesis[:500]},
                top_k=None,
            )
        if predictions and isinstance(predictions[0], list):
            predictions = predictions[0]
        labels = {
            status: float(item["score"])
            for item in predictions
            if (status := _nli_label_name(str(item.get("label", ""))))
        }
        if not labels:
            logger.warning("NLI model returned unrecognized labels.")
            return None
        best_label, best_score = max(labels.items(), key=lambda item: item[1])
        if best_label == "insufficient_evidence":
            return best_label, best_score
        neutral_score = labels.get("insufficient_evidence", 0.0)
        if best_score < _NLI_SCORE_THRESHOLD or best_score - neutral_score < 0.15:
            return "insufficient_evidence", max(best_score, neutral_score)
        return best_label, best_score
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning("Optional NLI inference failed (%s).", type(exc).__name__)
        return None


def compare_evidence(
    claim: str,
    items: list[EvidenceItem],
) -> tuple[str, str, list[EvidenceItem]]:
    comparisons: list[tuple[str, float, EvidenceItem, float, bool]] = []
    for item in items:
        if not item.snippet:
            continue
        result = nli_label(item.snippet, claim)
        if result is None:
            continue
        status, score = result
        quality, primary = _source_quality(item)
        if status in {"supported", "contradicted"} and score >= _NLI_SCORE_THRESHOLD:
            comparisons.append((status, score, item, quality, primary))

    credible = [row for row in comparisons if row[3] >= 0.80]
    used_sources = [row[2] for row in credible]
    supported = [row for row in credible if row[0] == "supported"]
    contradicted = [row for row in credible if row[0] == "contradicted"]
    if supported and contradicted:
        conflict_sources = []
        seen_urls = set()
        for item in used_sources:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                conflict_sources.append(item)
        return (
            "mixed_or_conflicting_evidence",
            "Credible source comparisons conflict; the evidence does not resolve the claim.",
            conflict_sources,
        )

    decisive = supported or contradicted
    independent_domains = {
        (urlparse(row[2].url).hostname or "").lower() for row in decisive
    }
    has_primary_support = any(row[4] for row in decisive)
    if decisive and (len(independent_domains) >= 2 or has_primary_support):
        status = decisive[0][0]
        direction = "supports" if status == "supported" else "directly contradicts"
        basis = "an authoritative primary source" if has_primary_support and len(independent_domains) == 1 else "independent credible sources"
        return (
            status,
            f"Natural-language inference found evidence that {direction} the claim in {basis}.",
            [row[2] for row in decisive],
        )
    if comparisons:
        return (
            "insufficient_evidence",
            "Relevant passages were found, but they do not meet the source-quality and corroboration threshold for a verdict.",
            [row[2] for row in comparisons],
        )
    return (
        "insufficient_evidence",
        "Search results were available, but no passage clearly supported or contradicted the claim.",
        [],
    )


async def verify_text(text: str):
    claims = extract_claims(text)
    if not claims:
        return VerifyResponse(
            overall_status="insufficient_evidence",
            claims=[],
            message="No clear factual claims were extracted from the supplied text. This is not a judgment that the answer is true or false.",
            evidence_retrieval_status="not_needed",
        )

    factual_claims = [claim for claim in claims if classify_claim(claim) == "factual"]
    semaphore = asyncio.Semaphore(4)

    async def retrieve(claim: str) -> tuple[list[EvidenceItem], str | None]:
        async with semaphore:
            return await search_evidence(claim)

    retrieved = await asyncio.gather(*(retrieve(claim) for claim in factual_claims))
    retrieved_by_claim = dict(zip(factual_claims, retrieved))
    results = []
    statuses = []
    retrieval_states = []
    provider_errors = []
    for claim in claims:
        claim_type = classify_claim(claim)
        if claim_type == "subjective":
            results.append(
                ClaimResult(
                    claim=claim,
                    claim_type=claim_type,
                    status="insufficient_evidence",
                    explanation="This is a subjective opinion rather than an objectively verifiable factual claim.",
                    evidence=[],
                    evidence_status="not_needed",
                )
            )
            continue
        search_results, provider_error = retrieved_by_claim[claim]
        if provider_error and not search_results:
            evidence_status = "unavailable"
            retrieval_states.append("unavailable")
            provider_errors.append(provider_error)
            explanation = provider_error
            evidence = []
            status = "insufficient_evidence"
        elif not search_results:
            evidence_status = "no_results"
            retrieval_states.append("no_results")
            explanation = "The search provider responded successfully but returned no results for this claim."
            evidence = []
            status = "insufficient_evidence"
        else:
            evidence_status = "available"
            retrieval_states.append("partial" if provider_error else "available")
            if provider_error:
                provider_errors.append(provider_error)
            status, explanation, evidence = await asyncio.to_thread(
                compare_evidence,
                claim,
                search_results[:_SEARCH_RESULTS_LIMIT],
            )
            if _nli_error:
                evidence_status = "available"
                explanation = (
                    f"{_nli_error} Search results are shown as candidates only; "
                    "they have not been compared with the claim."
                )
                evidence = search_results[:_SEARCH_RESULTS_LIMIT]
            elif not evidence and status == "insufficient_evidence":
                evidence_status = "irrelevant"
                if explanation.startswith("Search results were available"):
                    explanation = "Search completed, but its results did not contain a passage relevant enough to compare with this claim."
                else:
                    evidence_status = "inconclusive"
            if provider_error:
                explanation = f"Some search queries failed: {provider_error} {explanation}"

        results.append(
            ClaimResult(
                claim=claim,
                claim_type=claim_type,
                status=status,
                explanation=explanation,
                evidence=evidence,
                evidence_status=evidence_status,
            )
        )
        statuses.append(status)

    if statuses and all(status == "supported" for status in statuses):
        overall = "supported"
    elif statuses and all(status == "contradicted" for status in statuses):
        overall = "contradicted"
    elif "mixed_or_conflicting_evidence" in statuses or (
        len(set(statuses)) > 1 and any(status in {"supported", "contradicted"} for status in statuses)
    ):
        overall = "mixed_or_conflicting_evidence"
    else:
        overall = "insufficient_evidence"

    if not retrieval_states:
        retrieval_status = "not_needed"
    elif all(state == "unavailable" for state in retrieval_states):
        retrieval_status = "unavailable"
    elif all(state == "no_results" for state in retrieval_states):
        retrieval_status = "no_results"
    elif all(state == "available" for state in retrieval_states):
        retrieval_status = "available"
    else:
        retrieval_status = "partial"
    message = (
        "Evidence triage uses retrieved search excerpts and an NLI model; search rankings and model outputs can be incomplete or wrong. "
        "A verdict is not a guarantee of truth."
    )
    if provider_errors:
        message = "Evidence search was not fully available. " + " ".join(dict.fromkeys(provider_errors))

    return VerifyResponse(
        overall_status=overall,
        claims=results,
        message=message,
        evidence_retrieval_status=retrieval_status,
    )


@app.post("/api/v1/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    return await verify_text(req.text)


@app.get("/api/v1/metrics")
def metrics_v1():
    return metrics()
