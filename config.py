"""
Closed-Loop Adaptive Emotional Intervention — Configuration
Target: Raspberry Pi 5 (ARM64, 8GB)
Hardware: WS2812B NeoPixel strip (144 LED) + 7 GPIO buttons + USB mic + audio output
"""

import os

# ── Project root (for resolving MP3 paths) ──────────────────────────
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Audio Pipeline ──────────────────────────────────────────────────
SAMPLE_RATE = 16_000          # Wav2Vec2-BERT expects 16 kHz
CHANNELS = 1
DTYPE = "float32"

LISTEN_WINDOW_SEC = 4.0       # Chunked capture window
VAD_RMS_THRESHOLD = 0.015     # RMS below this → silence, skip inference
VAD_MIN_SPEECH_SEC = 0.3      # Minimum voiced duration within window

POST_INTERVENTION_LISTEN_SEC = 3.0   # Audio capture after showing color
INTERVENTION_SETTLE_SEC = 5.0        # Wait before re-measuring

# ── SER Model ───────────────────────────────────────────────────────
MODEL_ID = "facebook/w2v-bert-2.0"   # Base checkpoint; replace with your
                                      # fine-tuned path or HF repo id.
NUM_LABELS = 7                        # TESS: angry, disgust, fear, happy,
                                      #        neutral, pleasant_surprise, sad
LABEL_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "pleasant_surprise",
    6: "sad",
}

# Simplified arousal mapping used by the mitigation loop
AROUSAL_MAP = {
    "angry":              "agitated",
    "disgust":            "tense",
    "fear":               "tense",
    "happy":              "calm",
    "neutral":            "calm",
    "pleasant_surprise":  "calm",
    "sad":                "tense",
}

AROUSAL_TRIGGER_STATES = {"agitated", "tense"}   # States that fire mitigation

# ── ONNX / Quantization ────────────────────────────────────────────
USE_ONNX = True               # Prefer ONNX Runtime on RPi5
ONNX_MODEL_PATH = "model/ser_w2v_bert.onnx"
ONNX_QUANTIZED_PATH = "model/ser_w2v_bert_q8.onnx"

# ── NeoPixel LED Strip (WS2812B) ───────────────────────────────────
LED_COUNT = 144               # Number of LEDs on the strip
LED_PIN = 10                  # GPIO 10 (SPI0 MOSI) — REQUIRED FOR RASPBERRY PI 5!
LED_BRIGHTNESS = 180          # 0-255 global brightness
LED_FREQ_HZ = 800_000         # WS2812B signal frequency
LED_DMA = 10                  # DMA channel (ignored when using SPI)
LED_INVERT = False            # Invert signal (for level-shifter)
LED_CHANNEL = 0               # PWM channel

# Dead LED zone — these LEDs are skipped (kept off)
LED_SKIP_START = 22           # First LED to skip (inclusive, index 22 = 23rd LED)
LED_SKIP_END = 36             # Last LED to skip (inclusive, index 36 = 37th LED)
                              # → LEDs 23..37 always off, all others active

# ── GPIO Buttons ────────────────────────────────────────────────────
BTN_MODE_CYCLE = 17           # GPIO 17 — cycles modes: Therapeutic → Avion → Circadian → Autism
BTN_LAMP_TOGGLE = 5           # GPIO 5  — turn lamp (LED strip) on/off
BTN_PLAY_STOP = 6             # GPIO 6  — play/pause audio track
BTN_VOL_UP = 13               # GPIO 13 — volume up
BTN_VOL_DOWN = 19              # GPIO 19 — volume down
BTN_DEBOUNCE_MS = 300         # Debounce time in milliseconds

# Feedback colours — LED flash for 1 s when switching mode
MODE_COLOR_THERAPEUTIC = "#00FF66"   # Bright green  → Terapeutica
MODE_COLOR_AVION       = "#C4A8FF"   # Soft violet   → Avion
MODE_COLOR_CIRCADIAN   = "#FF8C42"   # Warm orange   → Circadiano
MODE_COLOR_AUTISM      = "#4FC3F7"   # Soft blue     → Autismo

# ── Audio Player ────────────────────────────────────────────────────
AUDIO_DEFAULT_VOLUME_DB = -25.0  # Decibels (0.0 is max, -60.0 is mute)
AUDIO_VOLUME_STEP_DB = 3.0       # Decibels increment/decrement per button press
AUDIO_SESSION_SEC = 15 * 60   # 15 minutes — auto-pause after this duration

# MP3 tracks per mode (only modes with audio)
AUDIO_TRACK_AVION = os.path.join(
    _PROJECT_DIR,
    "Sonido de CABINA DE AVIÓN Pantalla Negra Para Dormir ✈️ [ASMR Ruido Blanco] 10 Horas [U7D7tJJe0IY].mp3",
)
AUDIO_TRACK_CIRCADIAN = os.path.join(
    _PROJECT_DIR,
    "White Noise Puro (Ruido Blanco) 12 horas continuas [hSERj4HYyxg].mp3",
)
AUDIO_TRACK_AUTISM = os.path.join(
    _PROJECT_DIR,
    "🔵 Musicoterapia para AUTISMO, ASPERGER ｜►【Neuro-Relajación Profunda】Terapia Sensorial Relajante [tN9vJnGyHEc].mp3",
)

# ── Color Mitigation (Hill-Climbing) — Therapeutic Mode ────────────
# Base palette – calming greens, ordered full-sat → desaturated
BASE_COLORS_HEX = [
    "#6F9C6B",   # Full saturation green
    "#5B9570",
    "#ADBE93",
    "#B5C4A8",
    "#C8D5BC",
    "#DAE3D1",
    "#ECF0E7",   # Near-white (maximum desaturation)
]

DESAT_STEP = 0.10             # Saturation decrement per hill-climbing step
MIN_SATURATION = 0.05         # Floor
MAX_ROLLBACKS = 3             # Consecutive worsening before full reset

# ── Avion Mode — Boeing 737 Cabin Lighting Palette ─────────────────
# Inspired by Boeing Sky Interior LED ceiling wash colors
AVION_COLORS = [
    {"name": "boarding",     "hex": "#FFD580", "duration": 0},    # Warm amber
    {"name": "takeoff",      "hex": "#C4A8FF", "duration": 0},    # Soft violet
    {"name": "cruise_day",   "hex": "#87CEEB", "duration": 0},    # Sky blue
    {"name": "cruise_night", "hex": "#1A1A4E", "duration": 0},    # Deep indigo
    {"name": "meal",         "hex": "#FFE4B5", "duration": 0},    # Warm white
    {"name": "landing",      "hex": "#FFB347", "duration": 0},    # Sunset orange
]
AVION_TRANSITION_SEC = 3.0    # Cross-fade duration between phases
AVION_HOLD_SEC = 30.0         # Hold each phase before transitioning

# ── Circadian Mode — Daylight Rhythm Cycle ─────────────────────────
# Colors follow the natural light temperature across the day
CIRCADIAN_SCHEDULE = [
    # (hour_start, hour_end, hex_color, description)
    (6,   8,  "#FF8C42", "Alba — luce calda arancione"),
    (8,  10,  "#FFD700", "Mattina — giallo dorato"),
    (10, 14,  "#F5F5DC", "Mezzogiorno — bianco naturale"),
    (14, 17,  "#87CEEB", "Pomeriggio — azzurro cielo"),
    (17, 19,  "#FF6347", "Tramonto — rosso corallo"),
    (19, 21,  "#8B4513", "Sera — ambra scuro"),
    (21, 23,  "#2C1A4E", "Notte — indaco profondo"),
    (23,  6,  "#0D0D2B", "Notte fonda — blu notte quasi spento"),
]
CIRCADIAN_TRANSITION_SEC = 60.0   # Slow cross-fade between time blocks
