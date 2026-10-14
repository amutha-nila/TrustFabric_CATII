from __future__ import annotations
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_baseline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("features", TfidfVectorizer(max_features=30000, ngram_range=(1, 1), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state))
    ])

def build_optimized(random_state: int = 42) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=40000, min_df=2, sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=20000, min_df=2, sublinear_tf=True))
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))
    ])

def parameter_grid() -> dict:
    return {
        "features__word__ngram_range": [(1,1), (1,2)],
        "classifier__C": [0.25, 1.0, 4.0],
    }
