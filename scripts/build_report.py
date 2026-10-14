from __future__ import annotations

import json
import re
from datetime import date
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "outputs/metrics/benchmark.json"
REPORT_PATH = ROOT / "report/TrustFabric_CATII_Report.pdf"
FIGURE_DIR = ROOT / "outputs/figures"

if not METRICS_PATH.exists():
    raise SystemExit("Missing benchmark results. Run `python -m trustfabric.train` first.")
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
if metrics.get("best_cv_f1_macro") is None:
    raise SystemExit("The benchmark has no tuned GridSearchCV result; rerun training without --skip-grid-search.")

test_results = [row for row in metrics["results"] if row["split"] == "test"]
validation_results = [row for row in metrics["results"] if row["split"] == "validation"]
if len(test_results) != 2 or len(validation_results) != 2:
    raise SystemExit("Expected baseline and improved-model results for both validation and test partitions.")

baseline = next(row for row in test_results if "baseline" in row["model"].lower())
improved = next(row for row in test_results if "tuned" in row["model"].lower())
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="PaperTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#173f73"),
        spaceAfter=5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="Affiliation",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        alignment=TA_CENTER,
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="AbstractTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="AbstractText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        alignment=TA_JUSTIFY,
        spaceAfter=3 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="PaperHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        spaceBefore=2.5 * mm,
        spaceAfter=1.1 * mm,
        keepWithNext=True,
        textColor=colors.HexColor("#173f73"),
    )
)
styles.add(
    ParagraphStyle(
        name="PaperBody",
        parent=styles["BodyText"],
        fontSize=7.6,
        leading=9.5,
        alignment=TA_JUSTIFY,
        spaceAfter=1.4 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallTable",
        parent=styles["BodyText"],
        fontSize=5.5,
        leading=6.6,
    )
)
styles.add(
    ParagraphStyle(
        name="FigureCaption",
        parent=styles["BodyText"],
        fontSize=6.4,
        leading=7.4,
        alignment=TA_CENTER,
        spaceAfter=2 * mm,
    )
)

page_width, page_height = A4
margin = 14 * mm
gutter = 6 * mm
column_width = (page_width - 2 * margin - gutter) / 2


def draw_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#53657a"))
    canvas.drawString(margin, 8 * mm, "TrustFabric · 19MAM56 Advanced Machine Learning Lab")
    canvas.drawRightString(page_width - margin, 8 * mm, f"{doc.page}")
    canvas.restoreState()


cover_frame = Frame(
    margin,
    margin + 2 * mm,
    page_width - 2 * margin,
    page_height - 2 * margin - 3 * mm,
    leftPadding=0,
    rightPadding=0,
    topPadding=0,
    bottomPadding=0,
    id="cover",
)
column_frames = [
    Frame(
        margin,
        margin + 2 * mm,
        column_width,
        page_height - 2 * margin - 3 * mm,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="left",
    ),
    Frame(
        margin + column_width + gutter,
        margin + 2 * mm,
        column_width,
        page_height - 2 * margin - 3 * mm,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="right",
    ),
]
doc = BaseDocTemplate(
    str(REPORT_PATH),
    pagesize=A4,
    title="TrustFabric: ML-Based Claim Credibility Triage",
    author="19MAM56 Advanced Machine Learning Lab",
    leftMargin=margin,
    rightMargin=margin,
    topMargin=margin,
    bottomMargin=margin,
)
doc.addPageTemplates(
    [
        PageTemplate(id="Cover", frames=[cover_frame], onPage=draw_page),
        PageTemplate(id="Columns", frames=column_frames, onPage=draw_page),
    ]
)


def p(text: str, style: str = "PaperBody") -> Paragraph:
    return Paragraph(text, styles[style])


def metric_text(record: dict) -> str:
    return (
        f"Accuracy {record['accuracy']:.3f}; macro precision "
        f"{record['precision_macro']:.3f}; macro recall {record['recall_macro']:.3f}; "
        f"macro-F1 {record['f1_macro']:.3f}; macro one-vs-rest ROC-AUC "
        f"{record['roc_auc_ovr_macro']:.3f}; macro PR-AUC "
        f"{record['pr_auc_macro']:.3f}; log loss {record['log_loss']:.3f}."
    )


story = [
    Spacer(1, 15 * mm),
    p("TrustFabric: ML-Based Claim Credibility Triage", "PaperTitle"),
    p("19MAM56 Advanced Machine Learning Lab · CAT II Project Work", "Affiliation"),
    p("III M.Sc. AI &amp; ML, V Semester · Coimbatore Institute of Technology", "Affiliation"),
    p(f"Experiment report · {date.today().isoformat()}", "Affiliation"),
    Spacer(1, 7 * mm),
    p("Abstract", "AbstractTitle"),
    p(
        "This study builds a reproducible three-class claim-credibility classifier using "
        "the LIAR benchmark. The six original labels are aggregated into false, mixed, "
        "and true. A word-unigram TF-IDF/logistic-regression baseline is compared with "
        "a tuned word-and-character TF-IDF logistic-regression model. Official LIAR "
        "train, validation, and test partitions are preserved; a training-only "
        "three-fold GridSearchCV selects hyperparameters using macro-F1. The pipeline "
        "records class-wise and aggregate metrics, ROC/precision-recall curves, "
        "confusion matrices, coefficient inspection, and test-set errors. Its held-out "
        f"results are {baseline['f1_macro']:.3f} baseline macro-F1 and "
        f"{improved['f1_macro']:.3f} improved-model macro-F1. Predictions reflect "
        "dataset labels and language patterns, not proof of truth. A separate FastAPI "
        "evidence-verification workflow and Docker packaging support local inference.",
        "AbstractText",
    ),
    p(
        "<b>Index Terms—</b> claim credibility, LIAR, text classification, TF-IDF, "
        "logistic regression, hyperparameter optimization, evidence-aware verification.",
        "AbstractText",
    ),
    NextPageTemplate("Columns"),
    PageBreak(),
    p("I. INTRODUCTION", "PaperHeading"),
    p(
        "Short factual statements are difficult to assess automatically because their "
        "meaning depends on context, source quality, time, and human annotation. This "
        "project treats classification as a review-prioritization task rather than an "
        "automated truth oracle. Its objectives are to document a real benchmark, "
        "prevent train/test leakage, compare a reproducible baseline with a tuned "
        "pipeline, analyse errors, and expose model output through a validated API."
    ),
    p("II. LITERATURE REVIEW", "PaperHeading"),
    p(
        "Wang introduced LIAR as a benchmark of short political statements with six "
        "fact-checking labels [1]. The original paper's task differs from this "
        "three-class aggregation, so published scores are context rather than a "
        "direct comparator. Pérez-Rosas et al. studied automatic fake-news detection "
        "on a different dataset and task [3]; their reported results are likewise not "
        "numerically comparable here. TF-IDF and linear classifiers provide a "
        "transparent, reproducible text-classification baseline, implemented with "
        "scikit-learn [2]."
    ),
    p("III. DATASET AND PREPROCESSING", "PaperHeading"),
    p(
        "The source is the author-hosted LIAR archive [1]. Its README permits research "
        "use, states that original sources retain copyright, and provides no general "
        "redistribution license. The archive's 12,791 records comprise 10,240 train, "
        "1,284 validation, and 1,267 test statements before cleaning. This experiment "
        f"retains {metrics['dataset_rows']:,} unique usable statements; the official "
        "partitions remain separate. Blank statements, missing/unknown labels, and "
        "duplicate statement text are removed. Text is Unicode NFKC-normalized, "
        "control characters are stripped, and whitespace is collapsed. Six labels "
        "map to false (pants-fire/false), mixed (barely-true/half-true), and true "
        "(mostly-true/true)."
    ),
    p(
        "<b>Data limitations:</b> LIAR contains short US political claims, reflects "
        "human judgments, and may be affected by source, speaker, and time biases. "
        "Three-class aggregation discards distinctions between original labels. "
        "Results do not establish current factual accuracy or generalization to "
        "other domains."
    ),
    p("IV. METHODS AND EXPERIMENTAL DESIGN", "PaperHeading"),
    p(
        "The baseline uses word unigram TF-IDF (maximum 30,000 features) and "
        "class-weighted logistic regression. The improved estimator combines word "
        "n-gram TF-IDF (1–2 grams, up to 40,000 features) and character-boundary "
        "n-gram TF-IDF (3–5 grams, up to 20,000 features) before logistic regression. "
        "Both models use the same training, validation, and test partitions. The "
        "six-candidate grid searches word n-gram range and regularization strength "
        "C ∈ {0.25, 1, 4}; stratified three-fold CV is performed on training data only."
    ),
    p(
        "Validation results are measured on the original validation partition before "
        "refitting. After selection, baseline and improved estimators are refit on "
        "train plus validation and evaluated once on the held-out official test set. "
        "The random seed is 42. Vectorization resides within each scikit-learn "
        "pipeline, so each CV fold learns its vocabulary only from that fold's "
        "training data. The logged metrics are accuracy, macro precision/recall/F1, "
        "per-class precision/recall/F1/support, one-vs-rest macro ROC-AUC, macro "
        "average precision, and log loss. Because TF-IDF is sparse and already "
        "high-dimensional, dense PCA is not applied; capped word/character "
        "vocabularies and class-specific linear coefficients provide the "
        "representation analysis without discarding sparse lexical signals."
    ),
    p("V. IMPLEMENTATION", "PaperHeading"),
    p(
        "The modular code separates data ingestion, model construction, training, "
        "visualization, and inference. `trustfabric.train` writes both joblib model "
        "artifacts, JSON benchmark and confusion matrices, CSV comparison and "
        "hyperparameter history, classification reports, and paired test "
        "predictions. Seven plots are generated from the actual LIAR distribution "
        "and held-out experiment: class counts, optimized and baseline confusion "
        "matrices, one-vs-rest ROC, precision–recall, model comparison, and "
        "coefficient-based feature inspection. The FastAPI model endpoint presents "
        "dataset-model output separately from evidence verification."
    ),
    p("VI. RESULTS AND COMPARATIVE ANALYSIS", "PaperHeading"),
]

table_data = [[
    p("<b>Model</b>", "SmallTable"),
    p("<b>Acc.</b>", "SmallTable"),
    p("<b>P</b>", "SmallTable"),
    p("<b>R</b>", "SmallTable"),
    p("<b>F1</b>", "SmallTable"),
    p("<b>ROC</b>", "SmallTable"),
    p("<b>PR</b>", "SmallTable"),
]]
for label, row in (("Baseline", baseline), ("Word+char tuned", improved)):
    table_data.append(
        [
            p(label, "SmallTable"),
            *[
                p(f"{row[key]:.3f}", "SmallTable")
                for key in (
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                    "roc_auc_ovr_macro",
                    "pr_auc_macro",
                )
            ],
        ]
    )
comparison_table = Table(
    table_data,
    colWidths=[27 * mm, 9 * mm, 9 * mm, 9 * mm, 9 * mm, 10 * mm, 9 * mm],
    repeatRows=1,
)
comparison_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173f73")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ]
    )
)
story.extend(
    [
        comparison_table,
        Spacer(1, 1.5 * mm),
        p(
            f"Test-set baseline: {metric_text(baseline)} "
            f"Improved model: {metric_text(improved)} "
            f"Best training-only CV macro-F1 was {metrics['best_cv_f1_macro']:.3f}; "
            f"selected parameters were {escape(json.dumps(metrics['best_params'], sort_keys=True))}. "
            f"Test support was {metrics['split_rows']['test']:,} statements. "
            "The tuned model's gains are modest and are specific to this split and "
            "label mapping; no claim of state-of-the-art performance is made."
        ),
        p(
            "The most frequent error types and per-class results are saved in "
            "`outputs/metrics/error_analysis.csv` and "
            "`outputs/metrics/classification_report.json`. The comparison is "
            "reproduced under one shared test set; published experiments use "
            "incompatible data/label protocols and are not scored side-by-side."
        ),
        p("VII. DEPLOYMENT AND LIMITATIONS", "PaperHeading"),
        p(
            "The FastAPI app serves a browser interface, `/api/predict`, "
            "`/api/v1/verify`, and health/metrics endpoints. The classifier endpoint "
            "returns a dataset-label prediction with an explicit non-fact-check "
            "warning. The separate evidence workflow searches for sources and "
            "compares passages with an NLI model; search snippets and NLI outputs are "
            "not verified ground truth. The Docker image pins direct dependencies, "
            "runs as a non-root user, and uses a model-aware health check. A public "
            "AWS endpoint is intentionally not asserted unless deployed and tested."
        ),
        p(
            "Limitations include subjective labels, political-domain bias, label "
            "aggregation, limited context, dataset age, possible distribution shift, "
            "and uncalibrated probabilities. The model should support human review, "
            "not make high-stakes decisions. Evidence retrieval depends on external "
            "search availability and can return incomplete, stale, or low-quality "
            "sources."
        ),
        p("VIII. CONCLUSION AND FUTURE WORK", "PaperHeading"),
        p(
            "TrustFabric demonstrates a reproducible LIAR-based comparison with "
            "leakage-conscious preprocessing, a tuned TF-IDF ensemble of word and "
            "character representations, held-out analysis, a separate evidence "
            "verification workflow, and containerized inference. Future work should "
            "evaluate calibration, temporal and cross-domain robustness, additional "
            "annotators, transformer models under identical partitions, and "
            "human-in-the-loop decision support."
        ),
        p("REFERENCES", "PaperHeading"),
        p(
            "[1] W. Y. Wang, “Liar, Liar Pants on Fire: A New Benchmark Dataset for "
            "Fake News Detection,” in <i>Proc. 55th Annual Meeting of the Association "
            "for Computational Linguistics (Volume 2: Short Papers)</i>, 2017, "
            "pp. 422–426, doi: 10.18653/v1/P17-2067."
        ),
        p(
            "[2] F. Pedregosa <i>et al.</i>, “Scikit-learn: Machine Learning in "
            "Python,” <i>Journal of Machine Learning Research</i>, vol. 12, no. 85, "
            "pp. 2825–2830, 2011."
        ),
        p(
            "[3] V. Pérez-Rosas, B. Kleinberg, A. Lefevre, and R. Mihalcea, "
            "“Automatic Detection of Fake News,” in <i>Proc. 27th International "
            "Conference on Computational Linguistics</i>, 2018, pp. 3391–3401."
        ),
    ]
)

figure_specs = [
    ("01_label_distribution.png", "Fig. 1. Class counts after the three-way label mapping."),
    ("02_confusion_matrix.png", "Fig. 2. Improved model confusion matrix on the held-out test set."),
    ("02b_baseline_confusion_matrix.png", "Fig. 3. Baseline confusion matrix on the same test set."),
    ("03_roc_curves.png", "Fig. 4. One-vs-rest ROC curves for the improved model."),
    ("04_precision_recall.png", "Fig. 5. One-vs-rest precision–recall curves."),
    ("05_model_comparison.png", "Fig. 6. Baseline/improved comparison on the shared test set."),
    ("06_feature_coefficients.png", "Fig. 7. Largest positive class coefficients from the linear model."),
]
for filename, caption in figure_specs:
    path = FIGURE_DIR / filename
    if not path.exists():
        raise SystemExit(f"Missing experiment visualization: {path}")
    story.extend(
        [
            Spacer(1, 1 * mm),
            Image(str(path), width=column_width - 3 * mm, height=43 * mm, kind="proportional"),
            p(caption, "FigureCaption"),
        ]
    )

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
doc.build(story)
page_count = len(re.findall(rb"/Type\s*/Page\b", REPORT_PATH.read_bytes()))
if not 1 <= page_count <= 10:
    raise SystemExit(f"Report page limit violated: generated {page_count} pages (maximum 10).")
print(f"Wrote {REPORT_PATH} ({page_count} pages)")
