"""Optional DistilBERT fine-tuning experiment for a stronger neural benchmark.
Run after installing requirements-advanced.txt. This can download model weights and is compute intensive.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from trustfabric.data import load_csv, download_hf_dataset
ROOT=Path(__file__).resolve().parents[1]
LABELS=["false","mixed","true"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',default=None); ap.add_argument('--model',default='distilbert/distilbert-base-uncased'); ap.add_argument('--epochs',type=int,default=2); ap.add_argument('--max-length',type=int,default=160); ap.add_argument('--max-rows',type=int,default=0); args=ap.parse_args()
    csv=Path(args.csv) if args.csv else ROOT/'data/raw/liar.csv'
    if not csv.exists(): csv=download_hf_dataset(ROOT/'data/raw/liar.csv')
    df=load_csv(csv)
    if args.max_rows and args.max_rows<len(df):
        n=max(1,args.max_rows//3); df=pd.concat([g.sample(min(len(g),n),random_state=42) for _,g in df.groupby('label')]).reset_index(drop=True)
    train,test=train_test_split(df,test_size=.2,random_state=42,stratify=df.label)
    label2id={x:i for i,x in enumerate(LABELS)}; id2label={i:x for x,i in label2id.items()}
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    def prep(batch): return tokenizer(batch['text'],truncation=True,max_length=args.max_length)
    train_ds=Dataset.from_pandas(train[['text','label']].assign(labels=train.label.map(label2id)).drop(columns='label'),preserve_index=False).map(prep,batched=True)
    test_ds=Dataset.from_pandas(test[['text','label']].assign(labels=test.label.map(label2id)).drop(columns='label'),preserve_index=False).map(prep,batched=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=3,id2label=id2label,label2id=label2id)
    def compute(eval_pred):
        logits,labels=eval_pred; pred=np.argmax(logits,axis=-1); p,r,f,_=precision_recall_fscore_support(labels,pred,average='macro',zero_division=0)
        return {'accuracy':accuracy_score(labels,pred),'precision_macro':p,'recall_macro':r,'f1_macro':f}
    out=ROOT/'outputs/models/trustfabric-distilbert'; out.mkdir(parents=True,exist_ok=True)
    training_args=TrainingArguments(output_dir=str(ROOT/'outputs/transformer_checkpoints'),num_train_epochs=args.epochs,learning_rate=2e-5,per_device_train_batch_size=8,per_device_eval_batch_size=16,weight_decay=.01,evaluation_strategy='epoch',save_strategy='no',logging_strategy='epoch',report_to='none',seed=42)
    trainer=Trainer(model=model,args=training_args,train_dataset=train_ds,eval_dataset=test_ds,tokenizer=tokenizer,data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=compute)
    start=time.perf_counter(); trainer.train(); elapsed=time.perf_counter()-start
    result=trainer.evaluate(); pred=trainer.predict(test_ds).predictions.argmax(axis=-1); actual=test_ds['labels']
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    report={'model':args.model,'dataset_rows':len(df),'train_rows':len(train),'test_rows':len(test),'epochs':args.epochs,'max_length':args.max_length,'training_seconds':round(elapsed,2),'test_metrics':{k:float(v) for k,v in result.items() if isinstance(v,(int,float,np.floating))},'label_order':LABELS,'confusion_matrix':confusion_matrix(actual,pred,labels=[0,1,2]).tolist(),'note':'Compare with classical models only when dataset mapping and split protocol match.'}
    (ROOT/'outputs/metrics/transformer_benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame({'text':test.text.values,'actual':[id2label[int(x)] for x in actual],'predicted':[id2label[int(x)] for x in pred],'correct':actual==pred}).to_csv(ROOT/'outputs/metrics/transformer_error_analysis.csv',index=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
