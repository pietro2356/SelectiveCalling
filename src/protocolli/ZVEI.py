import numpy as np

# ==========================
#  DEFINIZIONI ZVEI-1
# ==========================
ZVEI1_FREQS = {
    "1": 1060, "2": 1160, "3": 1270, "4": 1400, "5": 1530,
    "6": 1670, "7": 1830, "8": 2000, "9": 2200, "0": 2400,
    "A": 2800, "B": 810,  "C": 970, "D": 886,  "E": 2600
}
ZVEI1_SYMBOLS = list(ZVEI1_FREQS.keys())
ZVEI1_VALUES = np.array(list(ZVEI1_FREQS.values()), dtype=float)


# ==========================
#  DEFINIZIONI ZVEI-2
# ==========================
ZVEI2_FREQS = {
    "1": 970, "2": 1060, "3": 1160, "4": 1270, "5": 1400,
    "6": 1530, "7": 1670, "8": 1830, "9": 2000, "0": 2200,
    "A": 2600, "B": 2800,  "C": 810, "D": 886,  "E": 2400
}
ZVEI2_SYMBOLS = list(ZVEI2_FREQS.keys())
ZVEI2_VALUES = np.array(list(ZVEI2_FREQS.values()), dtype=float)


# ==========================
#  DEFINIZIONI ZVEI-3
# ==========================
ZVEI3_FREQS = {
    "1": 970, "2": 1060, "3": 1160, "4": 1270, "5": 1400,
    "6": 1530, "7": 1670, "8": 1830, "9": 2000, "0": 2200,
    "A": 886, "B": 810,  "C": 740, "D": 680,  "E": 2400
}
ZVEI3_SYMBOLS = list(ZVEI3_FREQS.keys())
ZVEI3_VALUES = np.array(list(ZVEI3_FREQS.values()), dtype=float)

# ==========================
#  DEFINIZIONE LUNGHEZZA TONO
# ==========================
ZVEI_TONE_MS = 100  # Durata standard del tono ZVEI in millisecondi