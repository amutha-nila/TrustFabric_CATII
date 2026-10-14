# IEEE Report Generation

The final report is generated from actual experiment outputs by:

```powershell
python scripts/build_report.py
```

The resulting `TrustFabric_CATII_Report.pdf` is an IEEE-style, two-column paper and is limited to ten pages. The report builder stops rather than generating a placeholder report if the tuned experiment or required plots are missing.

The generated report contains:

1. Abstract and index terms.
2. Introduction and related published work.
3. LIAR dataset source, research-only terms, label mapping, cleaning, split sizes, and limitations.
4. Leakage-conscious preprocessing, TF-IDF representation, baseline, tuned model, and GridSearchCV protocol.
5. Reproducibility details, actual validation/test metrics, per-class results, and same-test-set comparison.
6. Genuine figures, error-analysis references, application/deployment architecture, limitations, conclusion, and IEEE references.

Regenerate the report after every training run. Do not copy smoke-run metrics into the final submission or represent results from another split as directly comparable published results.
