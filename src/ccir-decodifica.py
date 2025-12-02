import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================
#  DEFINIZIONI CCIR-7
# ==========================
CCIR7_FREQS = {
    "1": 1124, "2": 1197, "3": 1275, "4": 1358, "5": 1446,
    "6": 1540, "7": 1640, "8": 1747, "9": 1860, "0": 1981,
    "A": 2400, "B": 930, "C": 2246, "D": 991, "E": 2110
}

CCIR7_SYMBOLS = list(CCIR7_FREQS.keys())
CCIR7_VALUES = np.array(list(CCIR7_FREQS.values()))


# -------------------------------
#  ALGORITMO GOERTZEL - IMPLEMENTAZIONE
# -------------------------------
def goertzel(samples, sample_rate, freq):
    """Implementazione del filtro di Goertzel per una singola frequenza."""
    n = len(samples)
    k = int(0.5 + (n * freq) / sample_rate)
    omega = (2.0 * np.pi * k) / n
    coeff = 2.0 * np.cos(omega)
    s_prev = 0
    s_prev2 = 0

    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return power


def goertzel_band(samples, center_freq, fs, band=10):
    """Calcola la potenza massima in una piccola banda intorno alla frequenza centrale."""
    freqs = np.linspace(center_freq - band, center_freq + band, 5)
    powers = [goertzel(samples, f, fs) for f in freqs]
    return max(powers)

# -------------------------------
#  Decodifica CCIR con Goertzel
# -------------------------------
def decode_ccir_goertzel(file, tone_ms=100):
    audio, fs = sf.read(file)

    # stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    tone_len = int(fs * (tone_ms / 1000))
    result = ""

    for pos in range(0, len(audio) - tone_len, tone_len):
        segment = audio[pos:pos + tone_len]

        # calcola l’energia Goertzel Band per ogni frequenza CCIR
        magnitudes = [goertzel_band(segment, f, fs, band=10) for f in CCIR7_VALUES]

        # prendi la frequenza più intensa
        idx = np.argmax(magnitudes) # ?????
        best_freq = magnitudes[idx] # ?????


        if best_freq < 1000: # soglia di rumore. Se sotto, considera vuoto
            result += "-" # FIXME: Per debug stampiamo un trattino. In prod non stamperemo nulla
        else:
            result += CCIR7_SYMBOLS[idx]

        # if max(magnitudes.values()) > 12000:
        #     # prendi la frequenza più intensa
        #     best_freq = max(magnitudes, key=magnitudes.get)
        #     print(max(magnitudes.values()))
        #     # print(best_freq)
        #     result += CCIR7_SYMBOLS[list(CCIR7_VALUES).index(best_freq)]



    return result

# -------------------------------
#  MAIN
# -------------------------------
if __name__ == "__main__":
    print("Decodifica selettiva CCIR-7 con Goertzel")
    file = "./selettive_audio/00532.wav"
    code = decode_ccir_goertzel(file)
    print("Decodifica:", code)
