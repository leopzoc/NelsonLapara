"""
Speech Emotion Recognition Engine — Wav2Vec2-BERT inference for RPi 5.

Supports three backends (selected via config):
  1. ONNX Runtime  (preferred — fastest on ARM64)
  2. PyTorch INT8   (dynamic quantisation fallback)
  3. PyTorch FP32   (baseline, no-grad, eval-mode)

Also includes an ONNX export + INT8 quantisation helper.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

import config as cfg

log = logging.getLogger(__name__)


# ── ONNX Export & Quantisation (offline, run once) ──────────────────

def export_to_onnx(
    model_id: str = cfg.MODEL_ID,
    num_labels: int = cfg.NUM_LABELS,
    out_path: str = cfg.ONNX_MODEL_PATH,
):
    """Export the fine-tuned Wav2Vec2-BERT classifier to ONNX."""
    import torch
    from transformers import Wav2Vec2BertForSequenceClassification, AutoFeatureExtractor

    model = Wav2Vec2BertForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    )
    model.eval()
    processor = AutoFeatureExtractor.from_pretrained(model_id)

    dummy = torch.randn(1, cfg.SAMPLE_RATE * 4)   # 4 s dummy
    inputs = processor(
        dummy.squeeze().numpy(),
        sampling_rate=cfg.SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        (inputs["input_features"],),
        out_path,
        input_names=["input_features"],
        output_names=["logits"],
        dynamic_axes={
            "input_features": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    log.info("ONNX model exported → %s", out_path)


def quantise_onnx(
    src: str = cfg.ONNX_MODEL_PATH,
    dst: str = cfg.ONNX_QUANTIZED_PATH,
):
    """Apply dynamic INT8 quantisation to the ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    log.info("Quantised ONNX model → %s", dst)


# ── Inference Backends ──────────────────────────────────────────────

class _ONNXBackend:
    """ONNX Runtime inference session (ARM64 CPU EP)."""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        log.info("ONNX session loaded: %s", model_path)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """features: (1, seq_len, feat_dim) float32 → logits (1, num_labels)."""
        (logits,) = self.session.run(None, {self.input_name: features})
        return logits


class _TorchBackend:
    """PyTorch backend with optional INT8 dynamic quantisation."""

    def __init__(self, model_id: str, quantise: bool = True):
        import torch
        from transformers import (
            Wav2Vec2BertForSequenceClassification,
        )

        self.torch = torch
        self.model = Wav2Vec2BertForSequenceClassification.from_pretrained(
            model_id, num_labels=cfg.NUM_LABELS,
        )
        self.model.eval()

        if quantise:
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8,
            )
            log.info("PyTorch dynamic INT8 quantisation applied")

    def __call__(self, features: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            tensor = torch.from_numpy(features)
            logits = self.model(input_features=tensor).logits
            return logits.cpu().numpy()


# ── Public SER Engine ───────────────────────────────────────────────

class SEREngine:
    """
    Stateless emotion classifier.

    Instantiation loads the model once; call `predict(audio)` to get
    the emotion label and arousal bucket.
    """

    def __init__(self):
        from transformers import AutoFeatureExtractor

        self.processor = AutoFeatureExtractor.from_pretrained(cfg.MODEL_ID)
        self._backend = self._load_backend()

    # ── backend selection ───────────────────────────────────────────

    @staticmethod
    def _load_backend():
        if cfg.USE_ONNX:
            q_path = cfg.ONNX_QUANTIZED_PATH
            f_path = cfg.ONNX_MODEL_PATH
            path = q_path if os.path.isfile(q_path) else f_path
            if os.path.isfile(path):
                return _ONNXBackend(path)
            log.warning("ONNX model not found at %s — falling back to PyTorch", path)

        return _TorchBackend(cfg.MODEL_ID, quantise=True)

    # ── inference ───────────────────────────────────────────────────

    def predict(
        self, audio: np.ndarray, sr: int = cfg.SAMPLE_RATE
    ) -> Tuple[str, str, float]:
        """
        Parameters
        ----------
        audio : 1-D float32 waveform (mono, 16 kHz).

        Returns
        -------
        emotion : str   — raw label (e.g. "angry")
        arousal : str   — simplified bucket ("calm" | "tense" | "agitated")
        confidence : float
        """
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="np",
            padding=True,
        )
        features = inputs["input_features"].astype(np.float32)
        logits = self._backend(features)

        probs = _softmax(logits[0])
        idx = int(np.argmax(probs))
        emotion = cfg.LABEL_MAP.get(idx, "unknown")
        arousal = cfg.AROUSAL_MAP.get(emotion, "calm")
        confidence = float(probs[idx])

        log.info(
            "SER → %s (arousal=%s, conf=%.2f)", emotion, arousal, confidence
        )
        return emotion, arousal, confidence


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
