#!/usr/bin/env python
import argparse
import time
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor
)
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score

import config
from utils.dataset_loader import load_iemocap_metadata, get_fold


def prepare_dataset(df, processor, label2id):
    """
    Construye un Dataset con listas sin padding de input_values y attention_mask
    """
    examples = {
        "path": df["path"].tolist(),
        "label": [label2id[l] for l in df["label"].tolist()]
    }
    ds = Dataset.from_dict(examples)

    def preprocess(batch):
        import soundfile as sf, librosa
        speech_list = []
        for p in batch["path"]:
            audio, sr = sf.read(p)
            if sr != config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            speech_list.append(audio)
        proc_out = processor(
            speech_list,
            sampling_rate=config.SAMPLE_RATE,
            return_attention_mask=True,
            padding=False  # no pre-padding
        )
        # convertir a listas de floats/ints
        batch["input_values"] = [iv.tolist() for iv in proc_out["input_values"]]
        batch["attention_mask"] = [am.tolist() for am in proc_out["attention_mask"]]
        return batch

    return ds.map(preprocess, batched=True, remove_columns=["path"])


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "WA": accuracy_score(labels, preds),
        "UA": recall_score(labels, preds, average='macro'),
        "F1": f1_score(labels, preds, average='macro')
    }


def collate_fn(features):
    """
    Collate function que paddea manualmente secuencias de distinta longitud.
    """
    # crear tensores de input_values y masks
    iv_tensors = [torch.tensor(f["input_values"], dtype=torch.float32) for f in features]
    mask_tensors = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = torch.tensor([f["label"] for f in features], dtype=torch.long)

    # longitud máxima
    max_len = max(t.shape[-1] for t in iv_tensors)
    # pad y stack
    iv_padded = torch.stack([
        F.pad(t, (0, max_len - t.shape[-1]), value=0.0)
        for t in iv_tensors
    ])
    mask_padded = torch.stack([
        F.pad(m, (0, max_len - m.shape[-1]), value=0)
        for m in mask_tensors
    ])
    return {"input_values": iv_padded, "attention_mask": mask_padded, "labels": labels}


def fine_tune(epochs, fp16):
    print(f"Starting fine-tuning for wav2vec2 at {time.strftime('%Y-%m-%d %H:%M:%S')} epochs={epochs}, fp16={fp16}")
    start_time = time.time()

    emotions = config.EMOTIONS
    label2id = {l: i for i, l in enumerate(emotions)}
    id2label = {i: l for l, i in label2id.items()}

    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(emotions),
        label2id=label2id,
        id2label=id2label,
        gradient_checkpointing=True
    )

    df = load_iemocap_metadata()
    for fold in range(config.N_FOLDS):
        print(f"\n=== Fold {fold+1}/{config.N_FOLDS} ===")
        train_df, test_df = get_fold(df, fold)
        print(f"Train samples: {len(train_df)}, Eval samples: {len(test_df)}")

        print("Preparing datasets...")
        t_prep = time.time()
        train_ds = prepare_dataset(train_df, processor, label2id)
        eval_ds = prepare_dataset(test_df, processor, label2id)
        print(f"Datasets ready in {time.time() - t_prep:.2f}s")

        training_args = TrainingArguments(
            output_dir=f"ft_w2v2/fold{fold}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=100,
            num_train_epochs=epochs,
            fp16=fp16,
            learning_rate=3e-5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_UA",
            greater_is_better=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collate_fn,
            compute_metrics=compute_metrics
        )

        print("Starting training...")
        t_train = time.time()
        trainer.train()
        print(f"Training time: {time.time() - t_train:.2f}s")

        print("Evaluating...")
        t_eval = time.time()
        metrics = trainer.evaluate()
        print({k: v for k, v in metrics.items() if k.startswith("eval_")})
        print(f"Eval time: {time.time() - t_eval:.2f}s")

    total = time.time() - start_time
    print(f"\nTotal fine-tuning time: {total/3600:.2f}h ({total/60:.2f}m)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    fine_tune(args.epochs, args.fp16)
