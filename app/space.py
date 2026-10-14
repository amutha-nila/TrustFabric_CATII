"""Hugging Face Spaces Gradio UI. Set TRUSTFABRIC_MODEL_PATH if model artifact is elsewhere."""
from pathlib import Path
import os
import gradio as gr
from joblib import load
ROOT=Path(__file__).resolve().parent
if not (ROOT/"outputs").exists(): ROOT=ROOT.parent
MODEL_PATH=Path(os.getenv("TRUSTFABRIC_MODEL_PATH",ROOT/"outputs/models/trustfabric_model.joblib"))
model=load(MODEL_PATH) if MODEL_PATH.exists() else None

def assess(text):
    if not text or len(text.strip())<5: return {"error":"Enter a statement of at least 5 characters."}
    if model is None: return {"error":"No trained model artifact found. Train locally, then include the artifact in Space storage or configure TRUSTFABRIC_MODEL_PATH."}
    p=model.predict_proba([text])[0]; classes=list(model.classes_); idx=int(p.argmax())
    return {"prediction":str(classes[idx]),"confidence_score":round(float(p[idx]),4),"class_probabilities":{str(c):round(float(v),4) for c,v in zip(classes,p)},"caution":"Not proof of truth; independently verify with reliable evidence."}

with gr.Blocks(title="TrustFabric") as demo:
    gr.Markdown("# TrustFabric — Claim Credibility Triage\nResearch prototype using supervised machine learning. It predicts dataset labels; it does not establish truth.")
    text=gr.Textbox(label="Factual statement",lines=4,placeholder="Enter a claim to assess")
    btn=gr.Button("Assess claim")
    result=gr.JSON(label="Prediction")
    btn.click(assess,inputs=text,outputs=result)
    gr.Markdown("**Limitations:** predictions may be wrong, out of date, biased, or poorly calibrated. Do not use for high-stakes decisions.")
if __name__=="__main__": demo.launch(server_name="0.0.0.0",server_port=int(os.getenv("PORT","7860")))
