from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize

def make_visualizations(
    df,
    y_test,
    y_pred,
    y_score,
    labels,
    comparison,
    feature_names=None,
    coefficients=None,
    out_dir="outputs/figures",
    baseline_predictions=None,
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths=[]
    # 1. label distribution
    fig, ax = plt.subplots(figsize=(7,4)); df.label.value_counts().reindex(["false","mixed","true"]).plot(kind="bar", ax=ax)
    ax.set_title("LIAR label distribution after 3-class mapping"); ax.set_xlabel("Class"); ax.set_ylabel("Number of statements"); fig.tight_layout()
    p=out/"01_label_distribution.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(p)
    # 2. confusion matrix
    fig, ax=plt.subplots(figsize=(6,5)); ConfusionMatrixDisplay.from_predictions(y_test,y_pred,labels=labels,display_labels=labels,cmap="Blues",ax=ax,colorbar=False)
    ax.set_title("Held-out test confusion matrix"); fig.tight_layout(); p=out/"02_confusion_matrix.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(p)
    if baseline_predictions is not None:
        fig, ax=plt.subplots(figsize=(6,5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            baseline_predictions,
            labels=labels,
            display_labels=labels,
            cmap="Oranges",
            ax=ax,
            colorbar=False,
        )
        ax.set_title("Baseline held-out test confusion matrix")
        fig.tight_layout()
        p=out/"02b_baseline_confusion_matrix.png"
        fig.savefig(p,dpi=160)
        plt.close(fig)
        paths.append(p)
    # 3. ROC, one-vs-rest where score values are available
    fig, ax=plt.subplots(figsize=(6,5))
    y_bin=label_binarize(y_test,classes=labels)
    for i,lab in enumerate(labels):
        RocCurveDisplay.from_predictions(y_bin[:,i],y_score[:,i],name=lab,ax=ax)
    ax.plot([0,1],[0,1],"--",label="chance"); ax.set_title("One-vs-rest ROC curves"); fig.tight_layout()
    p=out/"03_roc_curves.png"; fig.savefig(p,dpi=160); paths.append(p)
    plt.close(fig)
    # 4. precision-recall
    fig, ax=plt.subplots(figsize=(6,5))
    for i,lab in enumerate(labels):
        PrecisionRecallDisplay.from_predictions(y_bin[:,i],y_score[:,i],name=lab,ax=ax)
    ax.set_title("One-vs-rest precision-recall curves"); fig.tight_layout()
    p=out/"04_precision_recall.png"; fig.savefig(p,dpi=160); paths.append(p)
    plt.close(fig)
    # 5. model comparison
    comp=pd.DataFrame(comparison)
    if "split" in comp:
        comp=comp[comp["split"]=="test"]
    if not comp.empty:
        metrics=[m for m in ["accuracy","precision_macro","recall_macro","f1_macro","roc_auc_ovr_macro","pr_auc_macro"] if m in comp.columns]
        fig,ax=plt.subplots(figsize=(10,5)); comp.set_index("model")[metrics].plot(kind="bar",ax=ax)
        ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_title("Held-out performance comparison")
        ax.tick_params(axis="x",labelrotation=12)
        ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),fontsize=8)
        fig.subplots_adjust(bottom=0.28,right=0.73,top=0.9)
        p=out/"05_model_comparison.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(p)
    # 6. coefficient-based explainability (linear model)
    if feature_names is not None and coefficients is not None:
        rows=[]
        for i, lab in enumerate(labels):
            coef=np.asarray(coefficients[i]).ravel(); inds=np.argsort(np.abs(coef))[-15:]
            rows.extend([(lab,str(feature_names[j]),float(coef[j])) for j in inds])
        fi=pd.DataFrame(rows,columns=["class","feature","coefficient"])
        fi.to_csv(out/"feature_coefficients.csv",index=False)
        fig,ax=plt.subplots(figsize=(9,6)); sub=fi.sort_values("coefficient").tail(25)
        ax.barh(sub.feature,sub.coefficient); ax.set_title("Largest positive linear-model feature coefficients"); ax.set_xlabel("Coefficient"); fig.tight_layout()
        p=out/"06_feature_coefficients.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(p)
    return [str(p) for p in paths]
