import asyncio
import os

import numpy as np
import pandas as pd
import pytest
from trustfabric.data import load_csv
from trustfabric.models import build_baseline
from trustfabric.train import split_dataset, score_model
from app import api
from app.api import app
from fastapi.testclient import TestClient

def test_label_mapping_and_cleaning(tmp_path):
    p=tmp_path/"claims.csv"
    pd.DataFrame({"statement":[f"Statement number {i} is a factual claim." for i in range(36)],"label":["true","false","half-true"]*12}).to_csv(p,index=False)
    df=load_csv(p)
    assert set(df.label)=={"true","false","mixed"}
    assert len(df)==36

def test_loader_rejects_wrong_columns(tmp_path):
    p=tmp_path/"bad.csv"; pd.DataFrame({"x":[1],"y":[2]}).to_csv(p,index=False)
    with pytest.raises(ValueError): load_csv(p)

def test_loader_cleans_text_deduplicates_and_preserves_source_splits(tmp_path):
    rows=[]
    for split in ("train", "validation", "test"):
        for index in range(12):
            label=("true","false","half-true")[index%3]
            rows.append({"statement":f"  Claim {split} example {index}   \u212b  ", "label":label, "split":split})
    rows.append({"statement":"Claim train example 0 \u00c5", "label":"true", "split":"valid"})
    p=tmp_path/"split_claims.csv"
    pd.DataFrame(rows).to_csv(p,index=False)

    df=load_csv(p)

    assert len(df)==36
    assert set(df.split)=={"train","valid","test"}
    assert "Claim train example 0 Å" in set(df.text)
    assert df.attrs["duplicate_text_rows"]==1

def test_custom_dataset_uses_stratified_train_validation_test_partitions():
    df=pd.DataFrame({
        "text":[f"Claim {i} is documented." for i in range(90)],
        "label":["false","mixed","true"]*30,
    })

    partitions, method=split_dataset(df)

    assert {name:len(part) for name,part in partitions.items()}=={"train":63,"valid":9,"test":18}
    assert "70/10/20" in method
    assert set(partitions["train"].text).isdisjoint(partitions["valid"].text)
    assert set(partitions["train"].text).isdisjoint(partitions["test"].text)
    assert set(partitions["valid"].text).isdisjoint(partitions["test"].text)

def test_baseline_pipeline_can_fit():
    X=["the statement is true", "the statement is false"]*12
    y=["true","false"]*12
    m=build_baseline(); m.fit(X,y)
    assert m.predict(["the statement is true"])[0] in {"true","false"}

def test_health_endpoint():
    client=TestClient(app)
    assert client.get('/health').status_code==200

def test_prediction_validates_short_input():
    client=TestClient(app)
    assert client.post('/api/predict',json={"text":"x"}).status_code==422

def test_prediction_returns_dataset_classifier_output_separately(monkeypatch):
    class FixedModel:
        classes_=np.array(["false","mixed","true"])

        def predict_proba(self, texts):
            return np.tile([0.1,0.2,0.7], (len(texts),1))

    monkeypatch.setattr(api,"model",FixedModel())
    response=TestClient(app).post(
        "/api/predict",
        json={"text":"A sufficiently long sample statement."},
    )

    assert response.status_code==200
    payload=response.json()
    assert payload["prediction"]=="true"
    assert "not a fact check" in payload["interpretation"]

def test_score_model_includes_multiclass_auc_and_per_class_metrics():
    texts=["fraud claim false", "mixed claim maybe", "verified claim true"]*4
    labels=["false","mixed","true"]*4
    model=build_baseline().fit(texts,labels)

    metrics, predictions, probabilities, matrix=score_model(
        "baseline", "test", model, pd.Series(texts), pd.Series(labels)
    )

    assert metrics["roc_auc_ovr_macro"]>=0
    assert metrics["pr_auc_macro"]>=0
    assert set(metrics["per_class"])=={"false","mixed","true"}
    assert probabilities.shape==(len(texts),3)
    assert len(predictions)==len(texts)
    assert np.asarray(matrix).shape==(3,3)

def test_verify_accepts_text_payload_and_returns_sources(monkeypatch):
    evidence = [api.EvidenceItem(
        title="NASA Artemis",
        url="https://www.nasa.gov/mission/artemis-ii/",
        snippet="NASA's Artemis II mission.",
    )]

    async def search(_query):
        return evidence, None

    monkeypatch.setattr(api, "search_evidence", search)
    monkeypatch.setattr(api, "nli_label", lambda _premise, _hypothesis: ("supported", 0.95))
    response = TestClient(app).post(
        "/api/v1/verify",
        json={"text": "NASA's Artemis II mission is planned for a crewed lunar flyby."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["claims"][0]["status"] == "supported"
    assert data["claims"][0]["evidence_status"] == "available"
    assert data["claims"][0]["evidence"][0]["url"] == evidence[0].url
    assert data["evidence_retrieval_status"] == "available"

def test_verify_rejects_missing_text():
    response = TestClient(app).post("/api/v1/verify", json={})
    assert response.status_code == 422
    assert "Field required" in response.text

@pytest.mark.parametrize("text", ["", "   "])
def test_verify_rejects_empty_or_whitespace_text(text):
    response = TestClient(app).post("/api/v1/verify", json={"text": text})
    assert response.status_code == 422
    assert "text" in response.text

def test_verify_reports_malformed_json():
    response = TestClient(app).post(
        "/api/v1/verify",
        content='{"text": ',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "json_invalid" in response.text

def test_verify_reports_provider_rate_limit(monkeypatch):
    async def search(_query):
        return [], "The evidence search provider rate-limited this request (HTTP 429)."

    monkeypatch.setattr(api, "search_evidence", search)
    response = TestClient(app).post(
        "/api/v1/verify",
        json={"text": "NASA's Artemis II mission is planned for a crewed lunar flyby."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["evidence_retrieval_status"] == "unavailable"
    assert data["claims"][0]["status"] == "insufficient_evidence"
    assert "rate-limited" in data["message"]


@pytest.mark.parametrize(
    ("claim", "snippet", "url", "relation", "status"),
    [
        (
            "The capital of Japan is Tokyo.",
            "Tokyo is the capital of Japan.",
            "https://www.cia.gov/the-world-factbook/countries/japan/",
            "supported",
            "supported",
        ),
        (
            "Apollo 11 landed on the Moon in 1970.",
            "Apollo 11 landed on the Moon in 1969.",
            "https://www.nasa.gov/mission/apollo-11/",
            "contradicted",
            "contradicted",
        ),
        (
            "Pure water boils at 100 degrees Celsius at sea level.",
            "Pure water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "https://www.nist.gov/",
            "supported",
            "supported",
        ),
        (
            "Marie Curie received the 1903 Nobel Prize in Chemistry.",
            "Marie Curie received the 1903 Nobel Prize in Physics.",
            "https://www.nobelprize.org/prizes/physics/1903/summary/",
            "contradicted",
            "contradicted",
        ),
        (
            "The Great Wall of China is visible from the Moon without aid.",
            "The Great Wall is not visible to the unaided eye from the Moon.",
            "https://www.nasa.gov/",
            "contradicted",
            "contradicted",
        ),
    ],
)
def test_general_nli_verdicts_return_only_sources_used(
    monkeypatch,
    claim,
    snippet,
    url,
    relation,
    status,
):
    evidence = [api.EvidenceItem(title="Reference result", url=url, snippet=snippet)]

    async def search(_query):
        return evidence, None

    monkeypatch.setattr(api, "search_evidence", search)
    monkeypatch.setattr(api, "nli_label", lambda premise, _hypothesis: (relation, 0.96) if premise == snippet else None)
    monkeypatch.setattr(api, "_nli_error", None)
    response = TestClient(app).post("/api/v1/verify", json={"text": claim})

    assert response.status_code == 200
    result = response.json()["claims"][0]
    assert result["status"] == status
    assert [item["url"] for item in result["evidence"]] == [url]
    assert result["explanation"]


def test_verify_keeps_fictional_claim_insufficient(monkeypatch):
    async def search(_query):
        return [
            api.EvidenceItem(
                title="Lunar library archive",
                url="https://www.britannica.com/topic/lunar-library",
                snippet="A lunar library preserves historical materials and records.",
            )
        ], None

    monkeypatch.setattr(api, "search_evidence", search)
    monkeypatch.setattr(api, "nli_label", lambda _premise, _hypothesis: None)
    monkeypatch.setattr(api, "_nli_error", None)
    response = TestClient(app).post(
        "/api/v1/verify",
        json={"text": "Zorblax's lunar library contains 73,000 books."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["claims"][0]["status"] == "insufficient_evidence"
    assert data["claims"][0]["evidence_status"] == "irrelevant"
    assert data["claims"][0]["evidence"] == []
    assert "did not contain a passage relevant enough" in data["claims"][0]["explanation"]


def test_verify_reports_conflicting_evidence_as_mixed(monkeypatch):
    evidence = [
        api.EvidenceItem(
            title="Eiffel Tower | Britannica",
            url="https://www.nasa.gov/mission/example/",
            snippet="Historical records say the Apollo mission landed in 1969.",
        ),
        api.EvidenceItem(
            title="Apollo mission timeline",
            url="https://www.britannica.com/science/Apollo-program",
            snippet="The Apollo mission landed in 1970.",
        ),
    ]

    async def search(_query):
        return evidence, None

    monkeypatch.setattr(api, "search_evidence", search)
    monkeypatch.setattr(
        api,
        "nli_label",
        lambda premise, _hypothesis: (
            ("supported", 0.96)
            if "1969" in premise
            else ("contradicted", 0.95)
        ),
    )
    monkeypatch.setattr(api, "_nli_error", None)
    response = TestClient(app).post(
        "/api/v1/verify",
        json={"text": "The Apollo mission landed in 1969."},
    )

    assert response.status_code == 200
    claim = response.json()["claims"][0]
    assert claim["status"] == "mixed_or_conflicting_evidence"
    assert len(claim["evidence"]) == 2
    assert "conflict" in claim["explanation"]


def test_successful_empty_search_is_distinct_from_provider_failure(monkeypatch):
    async def search(_query):
        return [], None

    monkeypatch.setattr(api, "search_evidence", search)
    response = TestClient(app).post(
        "/api/v1/verify",
        json={"text": "An undiscovered planet orbits the star Zorblax."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["claims"][0]["status"] == "insufficient_evidence"
    assert data["claims"][0]["evidence_status"] == "no_results"
    assert data["evidence_retrieval_status"] == "no_results"
    assert "responded successfully" in data["claims"][0]["explanation"]


def test_search_provider_timeout_is_reported_separately(monkeypatch):
    class TimedOutClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            raise api.httpx.ReadTimeout("slow")

    api._SEARCH_CACHE.clear()
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **_kwargs: TimedOutClient())
    results, error = asyncio.run(api.search_evidence("A factual claim with enough words"))

    assert results == []
    assert "before the timeout" in error


def test_nli_unavailable_does_not_convert_search_hits_to_verdict(monkeypatch):
    async def search(_query):
        return [
            api.EvidenceItem(
                title="Apollo 11 mission | NASA",
                url="https://www.nasa.gov/mission/apollo-11/",
                snippet="Apollo 11 was the first mission to land humans on the Moon.",
            )
        ], None

    monkeypatch.setattr(api, "search_evidence", search)
    monkeypatch.setattr(api, "nli_label", lambda _premise, _hypothesis: None)
    monkeypatch.setattr(api, "_nli_error", "NLI dependencies unavailable.")
    response = TestClient(app).post(
        "/api/v1/verify",
        json={"text": "Apollo 11 was the first mission to land humans on the Moon."},
    )

    assert response.status_code == 200
    claim = response.json()["claims"][0]
    assert claim["status"] == "insufficient_evidence"
    assert claim["evidence"][0]["url"] == "https://www.nasa.gov/mission/apollo-11/"
    assert "NLI dependencies unavailable" in claim["explanation"]


def test_search_query_generation_preserves_entities_dates_and_negation():
    queries = api.build_search_queries(
        "Claim: Marie Curie did not win the 1903 Nobel Prize in Chemistry."
    )

    assert len(queries) == 2
    assert "Marie Curie" in queries[0]
    assert "not" in queries[1]
    assert "1903" in queries[1]
    assert "Chemistry" in queries[1]


def test_compound_claim_splits_opinion_from_fact_and_resolves_pronoun():
    text = "ratan tata is a great personlality,he is still alive"

    claims = api.extract_claims(text)

    assert claims == ["ratan tata is still alive"]
    assert [api.classify_claim(claim) for claim in claims] == ["factual"]
    assert api.build_search_queries(claims[0])[0] == "ratan tata is still alive"


def test_subjective_compound_clause_does_not_suppress_factual_verification(monkeypatch):
    searched = []

    async def search(query):
        searched.append(query)
        return [
            api.EvidenceItem(
                title="Ratan Tata obituary",
                url="https://www.reuters.com/world/india/",
                snippet="Ratan Tata died on 9 October 2024.",
            ),
            api.EvidenceItem(
                title="Ratan Tata obituary",
                url="https://www.bbc.com/news/world-asia-india-",
                snippet="Ratan Tata died in October 2024.",
            ),
        ], None

    monkeypatch.setattr(api, "search_evidence", search)
    monkeypatch.setattr(api, "nli_label", lambda _premise, _hypothesis: ("contradicted", 0.98))
    monkeypatch.setattr(api, "_nli_error", None)
    response = TestClient(app).post(
        "/api/v1/verify",
        json={"text": "ratan tata is a great personlality,he is still alive"},
    )

    assert response.status_code == 200
    data = response.json()
    assert searched == ["ratan tata is still alive"]
    assert len(data["claims"]) == 1
    assert data["claims"][0]["claim_type"] == "factual"
    assert data["claims"][0]["status"] == "contradicted"
    assert data["claims"][0]["evidence"][0]["url"] == "https://www.reuters.com/world/india/"


def test_extracts_long_answer_facts_and_skips_filler_opinions_and_code():
    answer = (
        "Hello! The Eiffel Tower is located in Paris, France. "
        "It is 330 meters tall. I think it is beautiful. "
        "France is in Europe, and Germany is in Europe. "
        "What do you think? Let me know if you need anything else.\n"
        "```python\nprint('debug output')\n```"
    )

    assert api.extract_claims(answer) == [
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower is 330 meters tall.",
        "France is in Europe",
        "Germany is in Europe.",
    ]


def test_splits_independent_facts_and_reports_mixed_overall_status(monkeypatch):
    searched = []

    async def search(query):
        searched.append(query)
        if "Moon" in query and "natural satellite" in query:
            return [api.EvidenceItem(
                title="NASA Moon facts",
                url="https://www.nasa.gov/moon/",
                snippet="The Moon is Earth's natural satellite.",
            )], None
        return [], None

    monkeypatch.setattr(api, "search_evidence", search)
    monkeypatch.setattr(api, "nli_label", lambda *_args: ("supported", 0.96))
    monkeypatch.setattr(api, "_nli_error", None)
    answer = (
        "The Moon is a natural satellite of Earth. "
        "The Moon is made of cheese. I think it is beautiful."
    )
    response = TestClient(app).post("/api/v1/verify", json={"text": answer})

    assert response.status_code == 200
    data = response.json()
    assert len(searched) == 2
    assert len(data["claims"]) == 2
    assert [item["status"] for item in data["claims"]] == [
        "supported",
        "insufficient_evidence",
    ]
    assert data["overall_status"] == "mixed_or_conflicting_evidence"


def test_time_sensitive_claim_is_searched_and_keeps_its_own_verdict(monkeypatch):
    evidence = [api.EvidenceItem(
        title="Official current record",
        url="https://www.gov.uk/example-current-record",
        snippet="The record was updated in 2025.",
    )]
    searched = []

    async def search(query):
        searched.append(query)
        return evidence, None

    monkeypatch.setattr(api, "search_evidence", search)
    monkeypatch.setattr(api, "nli_label", lambda _premise, _hypothesis: ("supported", 0.95))
    monkeypatch.setattr(api, "_nli_error", None)
    response = TestClient(app).post(
        "/api/v1/verify",
        json={"text": "The current official record was updated in 2025."},
    )

    assert response.status_code == 200
    assert searched == ["The current official record was updated in 2025."]
    assert response.json()["claims"][0]["status"] == "supported"


def test_search_results_prioritize_reliable_sources_before_result_limit():
    weak = [
        api.EvidenceItem(title=f"Low quality {n}", url=f"https://example{n}.com/", snippet="A matching passage.")
        for n in range(api._SEARCH_RESULTS_LIMIT)
    ]
    primary = api.EvidenceItem(
        title="Official source",
        url="https://www.gov.uk/example",
        snippet="Official evidence for the factual claim.",
    )

    prioritized = api._prioritize_search_results(weak + [primary])

    assert len(prioritized) == api._SEARCH_RESULTS_LIMIT
    assert prioritized[0].url == primary.url


def test_search_provider_connection_failure_is_reported_and_not_cached_as_empty(monkeypatch):
    class FailedClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            raise api.httpx.ConnectError("offline")

    api._SEARCH_CACHE.clear()
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **_kwargs: FailedClient())

    results, error = asyncio.run(api.search_evidence("A factual claim with enough words"))

    assert results == []
    assert "could not be reached" in error


def test_search_result_parser_preserves_source_metadata_and_unwraps_provider_url():
    html = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nasa.gov%2Fmission%2Fapollo-11%2F">Apollo 11 | NASA</a>
      <div class="result__snippet">Apollo 11 launched in July 1969.</div>
      <time datetime="2025-07-16">Jul 16, 2025</time>
    </div>
    """

    results = api._parse_search_results(html)

    assert len(results) == 1
    assert results[0].url == "https://www.nasa.gov/mission/apollo-11/"
    assert results[0].title == "Apollo 11 | NASA"
    assert results[0].snippet == "Apollo 11 launched in July 1969."
    assert results[0].published_at == "2025-07-16"


@pytest.mark.parametrize(
    ("predictions", "expected"),
    [
        (
            [
                {"label": "contradiction", "score": 0.02},
                {"label": "entailment", "score": 0.95},
                {"label": "neutral", "score": 0.03},
            ],
            ("supported", 0.95),
        ),
        (
            [
                {"label": "contradiction", "score": 0.92},
                {"label": "entailment", "score": 0.03},
                {"label": "neutral", "score": 0.05},
            ],
            ("contradicted", 0.92),
        ),
        (
            [
                {"label": "contradiction", "score": 0.07},
                {"label": "entailment", "score": 0.08},
                {"label": "neutral", "score": 0.85},
            ],
            ("insufficient_evidence", 0.85),
        ),
    ],
)
def test_nli_three_way_label_mapping(monkeypatch, predictions, expected):
    class FakePipeline:
        def __call__(self, *_args, **_kwargs):
            return predictions

    monkeypatch.setattr(api, "_get_nli_model", lambda: FakePipeline())

    assert api.nli_label("Evidence passage.", "Factual claim.") == expected


def test_live_search_provider_integration():
    if os.getenv("TRUSTFABRIC_RUN_LIVE_SEARCH") != "1":
        pytest.skip("Set TRUSTFABRIC_RUN_LIVE_SEARCH=1 to enable the network integration test.")

    results, error = asyncio.run(api.search_evidence("NASA Apollo 11 mission"))

    assert error is None
    assert results
    assert all(item.url.startswith(("https://", "http://")) for item in results)
    assert any("Apollo" in item.title or "Apollo" in item.snippet for item in results)


def test_search_evidence_caches_repeated_queries(monkeypatch):
    class FakeResponse:
        text = (
            '<div class="result"><a class="result__a" href="https://www.nasa.gov/artemis-ii/">'
            'NASA Artemis II</a><div class="result__snippet">NASA Artemis II mission.</div></div>'
        )

        def raise_for_status(self):
            return None

    class FakeClient:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            self.calls += 1
            return FakeResponse()

    fake_client = FakeClient()
    api._SEARCH_CACHE.clear()
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda **_kwargs: fake_client)
    monkeypatch.setattr(api, "nli_label", lambda _premise, _hypothesis: None)
    test_client = TestClient(app)
    payload = {"text": "NASA's Artemis II mission is scheduled for launch."}

    assert test_client.post("/api/v1/verify", json=payload).status_code == 200
    assert test_client.post("/api/v1/verify", json=payload).status_code == 200
    assert fake_client.calls == len(api.build_search_queries(payload["text"]))


@pytest.mark.parametrize("origin", ["chrome-extension://" + ("a" * 32), "https://chatgpt.com"])
def test_extension_cors_preflight_is_allowed(origin):
    response = TestClient(app).options(
        "/api/v1/verify",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin

def test_home_page_serves_separate_assessment_panels():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "Dataset model assessment" in response.text
    assert "Evidence-based triage" in response.text
    assert "noopener noreferrer" in response.text
