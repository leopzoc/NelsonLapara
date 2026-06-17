"""
TESS Dataset Trainer — Fine-tune Wav2Vec2-BERT on the Toronto Emotional
Speech Set for 7-class SER, then export to ONNX + INT8 quantisation.

Run once on a machine with a GPU (not on the RPi itself):
    python train_ser.py --data_dir ./TESS --epochs 10 --batch 8
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample as scipy_resample
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

import config as cfg
from ser_engine import export_to_onnx, quantise_onnx

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

# ── Label parsing ───────────────────────────────────────────────────

TESS_LABELS = sorted(cfg.LABEL_MAP.values())   # alphabetical
LABEL_TO_ID = {l: i for i, l in enumerate(TESS_LABELS)}


def _parse_tess_label(filename: str) -> str:
    """
    TESS filenames follow:  OAF_back_angry.wav  /  YAF_bar_happy.wav
    The emotion is the last segment before the extension.
    """
    stem = Path(filename).stem
    emotion = stem.split("_")[-1].lower()
    # Normalise TESS label variants
    mapping = {
        "ps": "pleasant_surprise",
        "pleasantsurprise": "pleasant_surprise",
    }
    return mapping.get(emotion, emotion)


# ── Dataset ─────────────────────────────────────────────────────────

class TESSDataset(Dataset):
    def __init__(self, root_dir: str, processor: AutoFeatureExtractor):
        self.processor = processor
        self.samples: List[Dict] = []

        root = Path(root_dir)
        for wav_path in root.rglob("*.wav"):
            label_str = _parse_tess_label(wav_path.name)
            if label_str not in LABEL_TO_ID:
                continue
            self.samples.append({
                "path": str(wav_path),
                "label": LABEL_TO_ID[label_str],
            })

        log.info("TESS dataset: %d samples, %d classes", len(self.samples), len(LABEL_TO_ID))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        audio, sr = sf.read(entry["path"], dtype="float32")
        # If stereo, take first channel
        if audio.ndim > 1:
            audio = audio[:, 0]
        # Resample to 16 kHz if needed
        if sr != cfg.SAMPLE_RATE:
            num_samples = int(len(audio) * cfg.SAMPLE_RATE / sr)
            audio = scipy_resample(audio, num_samples).astype(np.float32)

        inputs = self.processor(
            audio,
            sampling_rate=cfg.SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            max_length=500,  # 5 seconds = 500 spectrogram frames (10ms hop)
            truncation=True,
        )
        return {
            "input_features": inputs["input_features"].squeeze(0),
            "label": torch.tensor(entry["label"], dtype=torch.long),
        }


# ── Training loop ──────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    processor = AutoFeatureExtractor.from_pretrained(cfg.MODEL_ID)
    dataset = TESSDataset(args.data_dir, processor)

    val_size = int(0.15 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = Wav2Vec2BertForSequenceClassification.from_pretrained(
        cfg.MODEL_ID, num_labels=cfg.NUM_LABELS,
    ).to(device)

    # Freeze feature encoder — only train classifier head + projection
    for name, param in model.named_parameters():
        if "classifier" not in name and "projector" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}"):
            feats = batch["input_features"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_features=feats, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_dl)

        # ── Validation ──────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_dl:
                feats = batch["input_features"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_features=feats).logits
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total else 0.0
        log.info("Epoch %d/%d — loss=%.4f  val_acc=%.2f%%", epoch, args.epochs, avg_loss, val_acc * 100)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(save_dir / "best"))
            processor.save_pretrained(str(save_dir / "best"))
            log.info("  ↳ Best model saved (acc=%.2f%%)", val_acc * 100)

    # ── Export ──────────────────────────────────────────────────────
    log.info("Exporting to ONNX …")
    export_to_onnx(model_id=str(save_dir / "best"))
    log.info("Quantising to INT8 …")
    quantise_onnx()
    log.info("Done. Models in %s and %s", cfg.ONNX_MODEL_PATH, cfg.ONNX_QUANTIZED_PATH)


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2-BERT on TESS")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to TESS dataset root")
    parser.add_argument("--output_dir", type=str, default="model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    train(parser.parse_args())
