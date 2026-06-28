# 🧠 ONIX — Closed-Loop Adaptive Emotional Intervention

> Sistema di intervento emotivo adattivo ad anello chiuso basato su **Wav2Vec2-BERT** per il riconoscimento vocale delle emozioni (**SER**), con retroazione cromatica in tempo reale tramite striscia LED NeoPixel WS2812B su **Raspberry Pi 5**.

---

## 📋 Indice

- [Panoramica](#-panoramica)
- [Architettura](#-architettura)
- [Requisiti Hardware](#-requisiti-hardware)
- [Modalità Operative](#-modalità-operative)
- [Setup — Training (PC/GPU)](#-setup--training-pcgpu)
- [Setup — Deployment su Raspberry Pi 5](#-setup--deployment-su-raspberry-pi-5)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Panoramica

ONIX analizza l'audio catturato dal microfono in finestre da 4 secondi, classifica le emozioni tramite un modello **Wav2Vec2-BERT** fine-tunato sul dataset **TESS** (Toronto Emotional Speech Set), e reagisce mostrando colori calmanti su una striscia LED NeoPixel da 144 LED, utilizzando un algoritmo di **hill-climbing** per ottimizzare l'intervento.

### Emozioni riconosciute (7 classi)
| Emozione | Arousal |
|----------|---------|
| 😠 Angry | Agitated |
| 🤢 Disgust | Tense |
| 😨 Fear | Tense |
| 😊 Happy | Calm |
| 😐 Neutral | Calm |
| 😮 Pleasant Surprise | Calm |
| 😢 Sad | Tense |

---

## 🏗 Architettura

```
Microfono USB ──► AudioStreamer ──► VAD Filter ──► SEREngine (Wav2Vec2-BERT)
                                                        │
                                                   emotion + arousal
                                                        │
                                                        ▼
                                              Main State Machine
                                              (LISTENING → MITIGATING → COOLDOWN)
                                                        │
                                                        ▼
                                              ColorMitigation (Hill-Climbing)
                                                        │
                                                        ▼
                                              NeoPixel Strip (144 LED)
```

### Pipeline di Inferenza (ordine di priorità)
1. **ONNX Runtime INT8** (`model/ser_w2v_bert_q8.onnx`) — ~1.2s per 4s audio su RPi5
2. **ONNX Runtime FP32** (`model/ser_w2v_bert.onnx`) — fallback
3. **PyTorch dynamic INT8** — auto-quantizza i layer `nn.Linear`
4. **PyTorch FP32** — baseline

---

## 🔧 Requisiti Hardware

### Raspberry Pi 5
- **Modello**: Raspberry Pi 5 — **8 GB RAM** (consigliato)
- **Storage**: MicroSD ≥ 32 GB (Class 10 / A2)
- **OS**: Raspberry Pi OS (64-bit Bookworm) o Ubuntu 24.04 LTS ARM64

### Componenti aggiuntivi
| Componente | Connessione | Note |
|------------|-------------|------|
| **Striscia LED NeoPixel WS2812B** (144 LED) | GPIO 18 (PWM0) | Alimentatore 5V esterno (≥ 10A) |
| **Microfono USB** | USB-A | Qualsiasi mic compatibile ALSA |
| **Pulsante 1** — Modalità Terapeutica | GPIO 17 → GND | Pull-up interno |
| **Pulsante 2** — Modalità Avion | GPIO 27 → GND | Pull-up interno |
| **Pulsante 3** — Modalità Circadiano | GPIO 22 → GND | Pull-up interno |

### ⚡ Schema Alimentazione NeoPixel
```
Alimentatore 5V/10A ──┬── 5V ──► Striscia NeoPixel VCC
                       └── GND ──┬──► Striscia NeoPixel GND
                                 └──► Raspberry Pi GND
                                 
Raspberry Pi GPIO 18 ──► Striscia NeoPixel Data IN
```

> ⚠️ **IMPORTANTE**: NON alimentare la striscia da 144 LED direttamente dal RPi. A piena luminosità assorbe ~8.6A (60mA × 144 LED). Usare sempre un alimentatore esterno e collegare i GND insieme.

---

## 🎮 Modalità Operative

### 1. 🟢 Terapeutica (Pulsante GPIO 17)
Intervento emotivo ad anello chiuso:
1. Ascolta in finestre da 4 secondi
2. VAD filtra il silenzio (soglia RMS = 0.015)
3. Wav2Vec2-BERT classifica emozione → arousal
4. Se agitato/teso: mostra colore calmante (verde)
5. Hill-climbing: desatura progressivamente, misura dB, rollback se peggiora

### 2. ✈️ Avion (Pulsante GPIO 27)
Simula l'illuminazione cabina del Boeing 737 Sky Interior:
- **Boarding**: ambra caldo → **Takeoff**: viola tenue → **Cruise Day**: azzurro cielo → **Cruise Night**: indaco profondo → **Meal**: bianco caldo → **Landing**: arancione tramonto
- Ogni fase 30s con cross-fade 3s

### 3. 🌅 Circadiano (Pulsante GPIO 22)
Segue il ritmo della luce naturale basato sull'ora locale (alba → tramonto → notte).

---

## 🖥 Setup — Training (PC/GPU)

Il training va eseguito su una macchina con GPU (NON sul Raspberry Pi).

### 1. Clona il repository
```bash
git clone https://github.com/<tuo-utente>/NelsonLapara.git
cd NelsonLapara
```

### 2. Crea il Virtual Environment
```bash
python3 -m venv venv_onix
source venv_onix/bin/activate
pip install --upgrade pip
```

### 3. Installa le dipendenze per il training
```bash
pip install torch torchaudio transformers onnxruntime onnx numpy scipy soundfile tqdm kagglehub
```

### 4. Scarica il dataset TESS da Kaggle
```bash
python -c "
import kagglehub
path = kagglehub.dataset_download('ejlok1/toronto-emotional-speech-set-tess')
print('Dataset scaricato in:', path)
"
```
> 📝 Il path di default sarà: `~/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1`

### 5. Lancia il training
```bash
python train_ser.py \
  --data_dir ~/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1 \
  --epochs 10 \
  --batch 8
```

> 💡 Se la GPU ha poca memoria, riduci `--batch` a 2 o 4.

### Output del training
Il training produce nella cartella `model/`:
```
model/
├── best/                      # Checkpoint migliore (PyTorch + tokenizer)
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── ser_w2v_bert.onnx          # Modello ONNX FP32
└── ser_w2v_bert_q8.onnx       # Modello ONNX INT8 quantizzato (per RPi5)
```

---

## 🍓 Setup — Deployment su Raspberry Pi 5

### 1. Prepara il Raspberry Pi

#### Installa Raspberry Pi OS (64-bit)
1. Scarica [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Seleziona **Raspberry Pi OS (64-bit)** — Bookworm
3. Abilita SSH e configura WiFi dal menu impostazioni di Imager
4. Flasha sulla microSD e avvia il RPi5

#### Aggiornamento sistema
```bash
sudo apt update && sudo apt upgrade -y
```

#### Dipendenze di sistema
```bash
sudo apt install -y \
  python3-venv python3-dev python3-pip \
  libsndfile1 portaudio19-dev \
  git
```

### 2. Clona il progetto sul Raspberry Pi
```bash
mkdir -p ~/intervention
cd ~/intervention
git clone https://github.com/<tuo-utente>/NelsonLapara.git .
```

### 3. Trasferisci il modello trainato dal PC

Dal **PC** dove hai eseguito il training:
```bash
scp -r model/ pi@<IP_RASPBERRY>:~/intervention/
```

Oppure, se hai tutto nello stesso repo git:
```bash
# Sul RPi
cd ~/intervention
git pull
```

### 4. Crea il Virtual Environment sul RPi
```bash
cd ~/intervention
python3 -m venv venv_rpi
source venv_rpi/bin/activate
pip install --upgrade pip
```

### 5. Installa le dipendenze RPi
```bash
pip install -r requirements.txt
```

> ⚠️ **Nota**: le librerie `rpi_ws281x` e `RPi.GPIO` richiedono `sudo` per l'accesso DMA/PWM. Se dai errore, installa con:
> ```bash
> sudo pip install rpi_ws281x RPi.GPIO
> ```

### 6. Configura i permessi

#### Accesso al framebuffer (per display)
```bash
sudo usermod -aG video $USER
```

#### Accesso audio (microfono USB)
```bash
sudo usermod -aG audio $USER
```

Fai logout e login per applicare i permessi.

### 7. Verifica il microfono USB
```bash
arecord -l   # Lista i dispositivi di registrazione
arecord -d 3 -f S16_LE -r 16000 test.wav   # Registra 3 secondi di test
aplay test.wav   # Riproduci per verificare
```

### 8. Avvia ONIX
```bash
# Attiva il venv
source ~/intervention/venv_rpi/bin/activate

# Avvia (sudo necessario per PWM DMA dei NeoPixel)
sudo $(which python) main.py
```

> 💡 **Perché `sudo $(which python)`?** I NeoPixel WS2812B richiedono accesso DMA/PWM, disponibile solo come root. Usando `$(which python)` ci assicuriamo di usare il Python del venv e non quello di sistema.

### 9. (Opzionale) Avvio automatico al boot

Crea un servizio systemd:
```bash
sudo tee /etc/systemd/system/onix.service << 'EOF'
[Unit]
Description=ONIX Emotional Intervention System
After=network.target sound.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/pi/intervention
ExecStart=/home/pi/intervention/venv_rpi/bin/python main.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable onix.service
sudo systemctl start onix.service
```

Per controllare i log:
```bash
sudo journalctl -u onix.service -f
```

---

## 📁 Struttura del Progetto

| File | Descrizione |
|------|-------------|
| `config.py` | Tutte le costanti: audio, modello, colori, GPIO, LED |
| `audio_pipeline.py` | Streamer `sounddevice`, RMS/dB, VAD ad energia |
| `ser_engine.py` | Inferenza Wav2Vec2-BERT: ONNX INT8 → PyTorch INT8 → FP32 |
| `color_mitigation.py` | Macchina a stati hill-climbing sulla saturazione HSL |
| `led_strip.py` | Driver NeoPixel WS2812B con cross-fade thread-safe |
| `buttons.py` | Controller pulsanti GPIO con interrupt e debounce |
| `mode_avion.py` | Ciclo illuminazione cabina Boeing 737 |
| `mode_circadian.py` | Ciclo luce circadiana basato sull'ora locale |
| `display.py` | Backend display: Framebuffer / Pygame / Tkinter |
| `main.py` | Orchestratore: gestione modalità e loop principale |
| `train_ser.py` | Training offline TESS + export ONNX (su GPU) |
| `requirements.txt` | Dipendenze Python per RPi5 |

---

## 🐛 Troubleshooting

### Il modello ONNX non si carica
```
ONNX model not found at model/ser_w2v_bert_q8.onnx — falling back to PyTorch
```
**Soluzione**: Assicurati che la cartella `model/` con i file `.onnx` sia presente nella directory del progetto. Ri-trasferisci dal PC se necessario:
```bash
scp -r model/ pi@<IP>:~/intervention/
```

### Errore permessi NeoPixel
```
Can't open /dev/mem: Permission denied
```
**Soluzione**: Esegui con `sudo`:
```bash
sudo $(which python) main.py
```

### Microfono non rilevato
```
sounddevice.PortAudioError: No input device
```
**Soluzione**: 
```bash
arecord -l  # Verifica che il mic sia rilevato
# Se serve, imposta il device di default:
echo 'defaults.pcm.card 1' >> ~/.asoundrc
```

### Out of Memory durante il training
**Soluzione**: Riduci la dimensione del batch:
```bash
python train_ser.py --data_dir ./TESS --epochs 10 --batch 2
```

### Inferenza lenta su RPi5
**Soluzione**: Assicurati di usare il modello ONNX INT8 quantizzato (`ser_w2v_bert_q8.onnx`). Se non esiste, eseguire la quantizzazione dal PC:
```python
from ser_engine import quantise_onnx
quantise_onnx()
```

### Il training non trova i file audio
**Soluzione**: Verifica che il path del dataset contenga i file `.wav` nelle sottocartelle:
```bash
find /path/to/TESS -name "*.wav" | head -5
```

---

## 📊 Performance Attese su RPi 5 (8 GB)

| Metrica | ONNX INT8 | ONNX FP32 | PyTorch INT8 | PyTorch FP32 |
|---------|-----------|-----------|--------------|--------------|
| Inferenza (4s audio) | ~1.2s | ~3.5s | ~4.0s | ~5.5s |
| RAM utilizzo | ~1.5 GB | ~2.5 GB | ~3.0 GB | ~3.5 GB |
| Dimensione modello | ~600 MB | ~2.3 GB | ~2.3 GB | ~2.3 GB |

---

## 📜 Licenza

Questo progetto è sviluppato come parte del progetto di ricerca Nelson-Lapara.

---

## 👤 Autore

Progetto a cura di Leonardo — Università degli Studi
