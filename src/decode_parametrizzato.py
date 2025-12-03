import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# ==========================
#  DEFINIZIONI CCIR
# ==========================
CCIR_FREQS = {
    "1": 1124, "2": 1197, "3": 1275, "4": 1358, "5": 1446,
    "6": 1540, "7": 1640, "8": 1747, "9": 1860, "0": 1981,
    "A": 2400, "B": 930,  "C": 2246, "D": 991,  "E": 2110
}
CCIR_SYMBOLS = list(CCIR_FREQS.keys())
CCIR_VALUES = np.array(list(CCIR_FREQS.values()), dtype=float)


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
#  DEFINIZIONI PCCIR
# ==========================
PCCIR_FREQS = {
    "1": 1124, "2": 1197, "3": 1275, "4": 1358, "5": 1446,
    "6": 1540, "7": 1640, "8": 1747, "9": 1860, "0": 1981,
    "A": 1050, "B": 930,  "C": 2400, "D": 991,  "E": 2110
}
PCCIR_SYMBOLS = list(PCCIR_FREQS.keys())
PCCIR_VALUES = np.array(list(PCCIR_FREQS.values()), dtype=float)

# ===============================
#  DEFINIZIONI TONI PER SELETTIVA
# ===============================
DEFAULT_TONE_MS = {
    "CCIR-1": 100.0,
    "CCIR-2": 70.0,
    "CCIR-7": 70.0,
    "PCCIR": 100.0,
    "ZVEI-1": 70.0,
    "ZVEI-2": 100.0,
    "ZVEI-3": 70.0
}

# -------------------------------
#  FILTRO BANDPASS (butterworth)
# -------------------------------
def bandpass_filter(signal, fs, lowcut=700.0, highcut=2500.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# -------------------------------
#  GOERTZEL - IMPLEMENTAZIONE CORRETTA
# -------------------------------
def goertzel(samples, sample_rate, freq):
    """
    Goertzel implementation returning power for frequency `freq` (Hz)
    samples: 1D numpy array (float)
    """
    n = len(samples)
    if n == 0:
        return 0.0
    # k (bin) must be float to compute omega, but we use integer k as in classic Goertzel
    k = int(0.5 + (n * freq) / sample_rate)
    omega = (2.0 * np.pi * k) / n
    coeff = 2.0 * np.cos(omega)

    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    # power (magnitude squared)
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    # numerical safety
    return float(np.abs(power))


def goertzel_band(samples, center_freq, fs, band=8, steps=5):
    """
    Evaluate a small frequency band around center_freq and return the maximal power.
    Helps to handle slight detuning.
    """
    freqs = np.linspace(center_freq - band, center_freq + band, steps)
    powers = [goertzel(samples, fs, f) for f in freqs]
    return max(powers)


# -------------------------------
#  DECISIONE PER SINGOLO FRAME
# -------------------------------
def detect_symbol_for_frame(frame, fs,
                            freq_list=CCIR_VALUES,
                            symbol_list=CCIR_SYMBOLS,
                            band=8,
                            ratio_threshold=3.0):
    """
    Restituisce (symbol, max_power, second_power, idx)
    symbol = '-' se nessuna decisione.
    L'algoritmo usa sia soglia assoluta (da fuori) che ratio primo/secondo per robustezza.
    """
    # finestrare il frame (riduce leakage)
    w = np.hamming(len(frame))
    frame_win = frame * w

    powers = np.array([goertzel_band(frame_win, f, fs, band=band) for f in freq_list])
    if powers.size == 0:
        return "-", 0.0, 0.0, -1

    idx = int(np.argmax(powers))
    max_p = float(powers[idx])
    # seconda potenza massima
    powers[idx] = 0.0
    second_p = float(np.max(powers))

    # restore (non necessario dopo)
    # powers[idx] = max_p

    # ratio check
    ratio = (max_p / (second_p + 1e-12)) if second_p > 0 else np.inf

    return (symbol_list[idx], max_p, second_p, idx) if ratio >= ratio_threshold else ("-", max_p, second_p, idx)


# -------------------------------
#  DECODER PRINCIPALE
# -------------------------------
def decode_ccir(file,
                tone_ms=100.0,
                overlap=0.5,
                prebandpass=True,
                bp_low=700.0,
                bp_high=2500.0,
                band=8,
                ratio_threshold=3.0,
                noise_factor=5.0,
                min_abs_power=None,
                debug=False,
                plot=False,
                cod="CCIR-1"):
    """
    Decodifica selettiva scelta di un file .wav.
    Parametri principali:
      - tone_ms: lunghezza di frame (ms) usata per campionare (default 100)
      - overlap: frazione di overlap tra frame (0..0.9), default 0.5 (50% overlap)
      - prebandpass: applica filtro band-pass prima dell'analisi
      - band: banda di ricerca intorno alla freq centrale (Hz)
      - ratio_threshold: primo/secondo rapporto minimo per accettare una decisione
      - noise_factor: fattore per ricavare soglia assoluta (soglia = median_noise * noise_factor)
      - min_abs_power: se fornito, forza una soglia minima assoluta oltre la quale accettare toni
      - debug: stampa informazioni per frame
      - plot: disegna grafici diagnostici
      - cod: tipo di selettiva
    Restituisce: (final_string, frames_info)
      frames_info = list di tuple (t_start_sec, symbol, max_power, second_power, idx)
    """


    # ==========================
    #    SELEZIONE DINAMICA
    # ==========================
    cod = cod.upper()

    # group_size = length_cod if length_cod is not None else 5

    ###--------------------------------------------------------------------------------------------------------###
    ###PROBLEMA CHE NON CAMPIONA COME DOVREBBE??, NON CAPISCO COME MAI DATO CHE IL TONO E' IMPOSTATO correttamente
    ###--------------------------------------------------------------------------------------------------------###

    #imposto il tone_ms in base alla selettiva scelta
    tone_ms = DEFAULT_TONE_MS.get(cod) #se non trovo il corrispondente imposto 100ms
    print("tone usato: ",  DEFAULT_TONE_MS.get(cod)) #per debug


    if cod in ("CCIR-1", "CCIR-2", "CCIR-7"):
        freq_list = CCIR_VALUES
        symbol_list = CCIR_SYMBOLS

    elif cod == "ZVEI-1":
        freq_list = ZVEI1_VALUES
        symbol_list = ZVEI1_SYMBOLS

    elif cod == "ZVEI-2":
        freq_list = ZVEI2_VALUES
        symbol_list = ZVEI2_SYMBOLS

    elif cod == "ZVEI-3":
        freq_list = ZVEI3_VALUES
        symbol_list = ZVEI3_SYMBOLS

    elif cod == "PCCIR":
        freq_list = PCCIR_VALUES
        symbol_list = PCCIR_SYMBOLS

    else:
        raise ValueError(f"Codifica selettiva sconosciuta: {cod}")
    

    # ==========================
    #    LETTURA FILE
    # ==========================
    audio, fs = sf.read(file)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(float)

    # prefilter (opzionale)
    if prebandpass:
        audio = bandpass_filter(audio, fs, lowcut=bp_low, highcut=bp_high, order=4)

    tone_len = int(fs * (tone_ms / 1000.0))
    hop = int(tone_len * (1.0 - overlap))
    if hop <= 0:
        hop = max(1, tone_len // 2)

    # prepara frames
    frames = []
    starts = list(range(0, max(0, len(audio) - tone_len + 1), hop))
    for s in starts:
        frames.append(audio[s:s + tone_len])

    # raccogli potenze (per calcolare soglia adattiva)
    frame_powers = []
    frames_info = []
    for i, frame in enumerate(frames):
        # usare detect_symbol_for_frame che ritorna massimo e secondo
        symbol, max_p, second_p, idx = detect_symbol_for_frame(
            frame, fs,
            freq_list=freq_list,
            symbol_list=symbol_list,
            band=band,
            ratio_threshold=ratio_threshold)
        
        frame_powers.append(max_p)
        t_start = starts[i] / fs
        frames_info.append((t_start, symbol, max_p, second_p, idx))

    # soglia adattiva: usa mediana dei frame come rumore di fondo
    median_noise = float(np.median(frame_powers))
    adaptive_threshold = median_noise * noise_factor
    if min_abs_power is not None:
        adaptive_threshold = max(adaptive_threshold, float(min_abs_power))

    if debug:
        print(f"Fs={fs}, tone_len={tone_len} samples ({tone_ms} ms), hop={hop} samples")
        print(f"Frames: {len(frames)}, median_noise={median_noise:.3e}, adaptive_threshold={adaptive_threshold:.3e}")

    # applica soglia e ratio per ottenere simbolo netto per frame
    final_frames = []
    for (t, symbol, max_p, second_p, idx) in frames_info:
        # se la potenza non supera la soglia adattiva -> vuoto
        if max_p < adaptive_threshold:
            final_frames.append((t, "-", max_p, second_p, idx))
        else:
            # ratio già applicata dentro detect_symbol_for_frame; se symbol=='-' potrà rimanere '-'
            final_frames.append((t, symbol, max_p, second_p, idx))

    # POST-PROCESSING: compressione e rimozione duplicati vicini
    # 1) elimina simboli '-' (silenzio) o li trasforma in separatori
    # 2) riduce sequenze ripetute consecutive a singolo simbolo
    seq = []
    for t_start, symbol, max_p, second_p, idx in final_frames:
        seq.append(symbol)

    # compress consecutive duplicates (es. ['5','5','-','-','5'] -> ['5','-','5'])
    compressed = []
    for s in seq:
        if not compressed or s != compressed[-1]:
            compressed.append(s)
        # if not compressed:
        #     compressed.append(s)
        # else:
        #     if s == compressed[-1]:
        #         continue
        #     compressed.append(s)

    # opzione: rimuovere '-' del tutto (si possono interpretare come "silenzio")
    # ma per sicurezza lasciamo solo come separatore e poi rimuoviamo leading/trailing
    cleaned = [c for c in compressed if c != '-']

    final_string = "".join(cleaned)

    if debug:
        print("Compressed frames (unique consecutive):", compressed)
        print("Cleaned (no '-'): ", cleaned)
        print("Final decoded string:", final_string)

    if plot:
        # plot potenze e decisioni
        times = [t for (t, _, _, _, _) in final_frames]
        mags = [p for (_, _, p, _, _) in final_frames]
        symb = [s for (_, s, _, _, _) in final_frames]
        plt.figure(figsize=(12, 4))
        plt.plot(times, mags, "-o", label="max power per frame")
        plt.axhline(adaptive_threshold, color='r', linestyle='--', label="adaptive threshold")
        for i, (t, s) in enumerate(zip(times, symb)):
            if s != '-':
                plt.text(t, mags[i], s, fontsize=9, ha='center', va='bottom')
        plt.xlabel("Time (s)")
        plt.ylabel("Power")
        plt.title("CCIR detection (frame max power and decisions)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # opzionale: histogram of powers
        plt.figure(figsize=(6,3))
        plt.hist(frame_powers, bins=60)
        plt.title("Histogram of frame max powers")
        plt.show()

    # restituisce anche le informazioni per frame per debugging
    return final_string, final_frames

def split_hex(hex_string, group_size):

    #Divide una stringa in gruppi di ma prima taglia la stessa dopo il pattern desiderato se presente

    hex_string = hex_string.upper()

    pattern = "4E4E"

    # Taglia tutto dopo il pattern
    idx = hex_string.find(pattern)
    if idx != -1:
        hex_string = hex_string[:idx + len(pattern)] #se c'e' pattern taglia

    hex_list = list(hex_string)  # converto in lista per modifiche

    for i in range(1, len(hex_list)):
        if hex_list[i] == 'E':
            hex_list[i] = hex_list[i-1]


    #se l'utente non specifica la lunghezza di codifica, usa default 5
    group_size = group_size if group_size is not None else 5  

    # Divide in gruppi
    groups = [hex_list[i:i+group_size] for i in range(0, len(hex_list), group_size)]

    # Trasforma ogni sottolista in stringa e formatta con parentesi tonde
    groups_str = ["(" + "".join(g) + ")" for g in groups]

    # Unisce con trattino
    return "-".join(groups_str)


# -------------------------------
#  ESEMPIO DI USO DA CLI
# -------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description= ("Decoder con supporto per selettive CCIR-1/CCIR-2/CCIR-7, PCCIR, ZVEI-1/ZVEI-2/ZVEI-3. "
                      "Con lunghezza di codifica configurabile."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # <-- mostra i default
    )
    parser.add_argument("file", nargs="?", default="./selettive_audio/00532.wav",
                        help="Percorso al file .wav (default: ./selettive_audio/00532.wav)")
    parser.add_argument("--tone-ms", type=float, default=100.0, help="Lunghezza frame in ms")
    parser.add_argument("--overlap", type=float, default=0.5, help="Frazione overlap (0..0.9)")
    parser.add_argument("--plot", action="store_true", help="Mostra grafici diagnostici")
    parser.add_argument("--debug", action="store_true", help="Stampa debug")
    parser.add_argument("--noise-factor", type=float, default=5.0, help="Fattore per soglia adattiva")
    parser.add_argument("--cod", type=str, default="CCIR-1", help="Selettiva per decodificare il file: CCIR-1, CCIR-2, CCIR-7, PCCIR, ZVEI-1, ZVEI-2, ZVEI-3")
    parser.add_argument("--length-cod", type=int, default=5, help="Lunghezza di codifica")

    args = parser.parse_args()

    decoded, frames = decode_ccir(args.file,
                                tone_ms=args.tone_ms,
                                overlap=args.overlap,
                                noise_factor=args.noise_factor,
                                debug=args.debug,
                                plot=args.plot,
                                cod=args.cod
                                )

    # print("Decoded:", decoded)
    print("Decoded:", split_hex(decoded, args.length_cod))   
