# 🆕 Resume Nuove Funzionalità — Integrazione Audio + 4 Pulsanti

> **Data**: 2026-07-13
> **Versione**: v2.0 — Aggiunta riproduzione audio per modalità + controlli fisici estesi

---

## 📋 Panoramica

Il sistema ONIX è stato esteso con:
1. **4 nuovi pulsanti fisici GPIO** per controllare lampada, riproduzione audio e volume
2. **Riproduzione audio MP3** dedicata per ogni modalità operativa
3. **Nuova modalità Autismo** — usa SER + ColorMitigation (come Terapeutica) con traccia audio di musicoterapia
4. **Nuovo modulo `audio_player.py`** — player MP3 basato su `pygame.mixer`

---

## 🎛 Nuovi Pulsanti GPIO

### Prima (v1.0) — 1 pulsante
| GPIO | Funzione |
|------|----------|
| 17 | Cicla modalità: Terapeutica → Avion → Circadiano |

### Dopo (v2.0) — 5 pulsanti
| GPIO | Pulsante | Funzione | Collegamento |
|------|----------|----------|--------------|
| 17 | Mode Cycle | Cicla modalità: Terapeutica → Avion → Circadiano → **Autismo** | GPIO 17 → GND |
| **5** | **Lamp On/Off** | Accende/spegne la striscia LED | GPIO 5 → GND |
| **6** | **Play/Stop** | Play/Pause della traccia audio corrente | GPIO 6 → GND |
| **13** | **Volume Up** | Aumenta volume (+10% per pressione) | GPIO 13 → GND |
| **19** | **Volume Down** | Diminuisce volume (−10% per pressione) | GPIO 19 → GND |

> ⚠️ Tutti i pulsanti usano la resistenza di pull-up interna (`PUD_UP`). Collegare ogni pulsante tra il pin GPIO e GND. La pressione porta il pin a LOW (FALLING edge).

### Schema cablaggio aggiornato

```
Raspberry Pi 5 (8 GB)
├── SPI0 (GPIO 10) ──── WS2812B NeoPixel Data (144 LED)
├── GPIO 17 ──── Pulsante: Mode Cycle
├── GPIO  5 ──── Pulsante: Lamp On/Off           ← NUOVO
├── GPIO  6 ──── Pulsante: Play/Stop Audio        ← NUOVO
├── GPIO 13 ──── Pulsante: Volume Up              ← NUOVO
├── GPIO 19 ──── Pulsante: Volume Down            ← NUOVO
├── USB    ──── Microfono
└── 3.5mm / HDMI / Bluetooth ──── Speaker (uscita audio)
```

---

## 🎵 Audio MP3 per Modalità

Ogni modalità ha la sua traccia audio dedicata. Le tracce partono **automaticamente al cambio di modalità** e si riproducono **anche a volume 0** (muto) — il volume controlla solo il livello di uscita, non la riproduzione.

| Modalità | Traccia Audio | File |
|----------|---------------|------|
| Terapeutica | ❌ Nessuna | — |
| Avion ✈️ | Suono cabina aereo (ASMR ruido blanco) | `Sonido de CABINA DE AVIÓN...mp3` |
| Circadiano 🌅 | White noise puro | `White Noise Puro (Ruido Blanco)...mp3` |
| Autismo 🧠 | Musicoterapia neuro-rilassamento | `🔵 Musicoterapia para AUTISMO...mp3` |

### Comportamento audio
- **Cambio modalità** → la traccia precedente si ferma, parte quella nuova (loop infinito)
- **Pulsante Play/Stop** → mette in pausa / riprende la traccia corrente
- **Volume Up/Down** → incrementi del 10%, range 0%–100%
- **Volume 0 (muto)** → l'audio continua a riprodursi, solo il livello è a zero

---

## 🧠 Modalità Autismo — Dettagli

La modalità Autismo **NON** ha un ciclo di colori proprio. Utilizza lo **stesso pipeline della modalità Terapeutica**:

```
Microfono → AudioStreamer → VAD → SEREngine (Wav2Vec2-BERT)
                                         │
                                    emotion + arousal
                                         │
                                         ▼
                               ColorMitigation (Hill-Climbing)
                                         │
                                         ▼
                               NeoPixel Strip (144 LED)
```

La differenza è che in parallelo viene riprodotta la traccia di **musicoterapia per autismo/Asperger** (neuro-rilassamento profondo).

---

## 💡 Pulsante Lampada — Come funziona

| Stato | LED | SER/Audio |
|-------|-----|-----------|
| **Lamp ON** (default) | Funzionano normalmente secondo la modalità | Tutto attivo |
| **Lamp OFF** | Tutti spenti (nero) | SER continua ad ascoltare, audio continua a suonare, hill-climbing continua a calcolare |

Quando si riaccende la lampada, il comportamento LED riprende immediatamente dalla modalità corrente.

---

## 📁 File Modificati / Creati

### Nuovo file

| File | Descrizione |
|------|-------------|
| `audio_player.py` | Player MP3 thread-safe via `pygame.mixer`. Gestisce play/pause/stop, volume up/down, caricamento tracce per modalità. |

### File modificati

| File | Modifiche |
|------|-----------|
| `config.py` | Aggiunto: `import os`, `_PROJECT_DIR`, 4 nuovi pin GPIO (`BTN_LAMP_TOGGLE=5`, `BTN_PLAY_STOP=6`, `BTN_VOL_UP=13`, `BTN_VOL_DOWN=19`), costanti audio (`AUDIO_DEFAULT_VOLUME`, `AUDIO_VOLUME_STEP`), 3 percorsi MP3 (`AUDIO_TRACK_AVION/CIRCADIAN/AUTISM`), colore feedback autismo (`MODE_COLOR_AUTISM`) |
| `buttons.py` | Aggiunto `AUTISM` al ciclo modalità. Aggiunto `MODE_AUDIO_TRACKS` (mappa modo→traccia). 4 nuovi handler GPIO con callback (`on_lamp_toggle`, `on_play_stop`, `on_volume_up`, `on_volume_down`). Debounce refactored in metodo condiviso `_debounced()`. |
| `main.py` | Integrato `AudioPlayer`. Aggiunto `_SER_MODES = {THERAPEUTIC, AUTISM}`. Callback per lampada/play/volume. Auto-play traccia al cambio modo. Rispetta stato lampada on/off nei `set_color()`. Shutdown chiude anche l'audio player. |
| `requirements.txt` | Aggiunto `pygame>=2.5` |

### File rimosso

| File | Motivo |
|------|--------|
| `mode_autism.py` | Non necessario: il modo autismo usa ColorMitigation (SER + hill-climbing), non un ciclo colori separato |

---

## 🔧 Costanti aggiunte in `config.py`

```python
# GPIO Buttons
BTN_LAMP_TOGGLE = 5           # GPIO 5  — turn lamp on/off
BTN_PLAY_STOP = 6             # GPIO 6  — play/pause audio
BTN_VOL_UP = 13               # GPIO 13 — volume up
BTN_VOL_DOWN = 19             # GPIO 19 — volume down

# Feedback colour
MODE_COLOR_AUTISM = "#4FC3F7"  # Soft blue → Autismo

# Audio
AUDIO_DEFAULT_VOLUME = 0.5    # 50%
AUDIO_VOLUME_STEP = 0.10      # ±10% per pressione

# MP3 paths (risolti con os.path.join dalla root del progetto)
AUDIO_TRACK_AVION     = os.path.join(_PROJECT_DIR, "Sonido de CABINA DE AVIÓN...mp3")
AUDIO_TRACK_CIRCADIAN = os.path.join(_PROJECT_DIR, "White Noise Puro...mp3")
AUDIO_TRACK_AUTISM    = os.path.join(_PROJECT_DIR, "🔵 Musicoterapia para AUTISMO...mp3")
```

---

## 📦 Dipendenza aggiunta

```
pygame>=2.5
```

Installare sul Raspberry Pi:
```bash
sudo pip install pygame>=2.5
# oppure
sudo pip install -r requirements.txt
```

---

## ✅ Checklist Deploy

- [ ] Collegare i 4 nuovi pulsanti (GPIO 5, 6, 13, 19) tra pin e GND
- [ ] Collegare un speaker / cuffie all'uscita audio del RPi (3.5mm jack o HDMI o Bluetooth)
- [ ] Copiare i 3 file MP3 nella cartella del progetto sul RPi
- [ ] Installare pygame: `sudo pip install pygame>=2.5`
- [ ] Trasferire i file aggiornati: `config.py`, `buttons.py`, `main.py`, `audio_player.py`
- [ ] Avviare: `sudo $(which python) main.py`
