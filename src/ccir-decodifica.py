import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, butter, filtfilt

CCIR1_FREQS = {
    1124: "1", 1197: "2", 1275: "3", 1358: "4", 1446: "5",
    1540: "6", 1640: "7", 1747: "8", 1860: "9", 1981: "0",
    2400: "A", 930: "B", 2247: "C", 991: "D", 2110: "E"
}

def closest_freq(f):
    return min(CCIR1_FREQS.keys(), key=lambda x: abs(x - f))

def bandpass(audio, fs, low=900, high=3000):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, audio)

def detect_tones(audio, fs):
    # envelope per segmentare i toni
    env = np.abs(audio)
    env = np.convolve(env, np.ones(int(fs*0.02))/int(fs*0.02), mode="same")

    # trova i picchi (inizio tono)
    peaks, _ = find_peaks(env, height=np.mean(env)*1.5, distance=fs*0.05)
    return peaks

def decode_ccir1(file):
    audio, fs = sf.read(file)
    if audio.ndim > 1:   # stereo → mono
        audio = audio.mean(axis=1)

    audio = bandpass(audio, fs)

    peaks = detect_tones(audio, fs)
    tone_len = int(fs * 0.07)  # circa 100 ms (accettato anche se non perfetto)

    digits = ""

    for p in peaks:
        segment = audio[p:p+tone_len]
        if len(segment) < tone_len:
            continue

        # FFT
        fft = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1/fs)

        peak_freq = freqs[np.argmax(fft)]
        matched = closest_freq(peak_freq)
        digits += CCIR1_FREQS[matched]

    return digits

if __name__ == "__main__":
    decoded = decode_ccir1("../selettive_audio/00259.wav")
    print("Codice:", decoded)
