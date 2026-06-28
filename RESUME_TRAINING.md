# 🔄 RESUME TRAINING — Guida per riprendere dopo il riavvio

> **Stato**: Training stoppato dopo Epoch 1/10 su CPU. Riavviare con CUDA (RTX 2060).
> **Data**: 2026-06-25
> **PC**: Pop!_OS con NVIDIA RTX 2060

---

## ✅ Cosa è già stato fatto

1. **Venv creato**: `venv_onix/` nella cartella del progetto
2. **Dipendenze installate**: torch, torchaudio, transformers, onnxruntime, onnx, numpy, scipy, soundfile, tqdm, kagglehub
3. **Dataset TESS scaricato** da Kaggle in:
   ```
   ~/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1
   ```
4. **Epoch 1 completata su CPU**: val_acc=46.79%, best model salvato in `model/best/`
5. **README.md creato**: guida completa per deployment su RPi 5

---

## 🔧 Cosa fare dopo il riavvio

### Step 1: Verificare che CUDA funzioni

```bash
nvidia-smi
```

Deve mostrare la RTX 2060. Se non funziona:
- Controlla che nelle impostazioni energetiche di Pop!_OS la GPU dedicata sia attiva
- Potresti dover fare: `sudo system76-power graphics nvidia`
- Poi riavvia

### Step 2: Verificare CUDA con PyTorch

```bash
cd /home/p_leonardo_c/Documenti/Github/NelsonLapara
source venv_onix/bin/activate
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Deve stampare: `CUDA: True` e `GPU: NVIDIA GeForce RTX 2060`

### Step 3: Eliminare il checkpoint CPU parziale (opzionale)

```bash
rm -rf model/best/ model/ser_w2v_bert.onnx model/ser_w2v_bert_q8.onnx
```

### Step 4: Lanciare il training con CUDA

```bash
cd /home/p_leonardo_c/Documenti/Github/NelsonLapara
source venv_onix/bin/activate

python train_ser.py \
  --data_dir "/home/p_leonardo_c/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1" \
  --epochs 10 \
  --batch 8
```

> **Nota**: Con la RTX 2060 (6 GB VRAM) puoi usare `--batch 8`.
> Se dà errore CUDA OOM, riduci a `--batch 4`.
> Tempo stimato: ~30-50 minuti totali (vs 20 ore su CPU).

### Step 5: Verificare l'output

Dopo il training, controlla che esistano:
```bash
ls -lh model/best/
ls -lh model/ser_w2v_bert.onnx
ls -lh model/ser_w2v_bert_q8.onnx
```

---

## 📂 Struttura progetto attuale

```
/home/p_leonardo_c/Documenti/Github/NelsonLapara/
├── venv_onix/          ← Virtual environment (già creato, dipendenze installate)
├── model/
│   └── best/           ← Checkpoint Epoch 1 CPU (val_acc=46.79%) — da sovrascrivere
├── train_ser.py        ← Script di training
├── ser_engine.py       ← Engine inferenza + export ONNX
├── config.py           ← Configurazione modello e costanti
├── README.md           ← Guida deployment RPi 5 (già creato)
├── requirements.txt    ← Dipendenze Python
└── RESUME_TRAINING.md  ← Questo file
```

---

## 🎯 Obiettivo finale

Dopo il training CUDA, i file modello (`model/best/`, `model/ser_w2v_bert.onnx`, `model/ser_w2v_bert_q8.onnx`) vanno trasferiti sul Raspberry Pi 5 come descritto nel README.md.
