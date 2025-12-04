import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import argparse

from protocolli.CCIR import (
    CCIR_SYMBOLS,
    CCIR_VALUES,
    PCCIR_VALUES,
    PCCIR_SYMBOLS,
    CCIR_CODE_LEN_MS,
    CCIR_TONE_CH_PAUSE,
    CCIR_TONE_CH_REPEATER
)
from protocolli.ZVEI import (
    ZVEI1_VALUES,
    ZVEI1_SYMBOLS,
    ZVEI2_VALUES,
    ZVEI2_SYMBOLS,
    ZVEI_TONE_MS,
    ZVEI_TONE_CH_PAUSE,
    ZVEI_TONE_CH_REPEATER
)

DEBUG_ENABLED = False


# -------------------------------
#  BANDPASS FILTER (butterworth)
# -------------------------------
def bandpass_filter(signal, fs, lowcut=700.0, highcut=2500.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# -------------------------------
#  GOERTZEL FILTER
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
#  DETECT SYMBOL FOR FRAME
# -------------------------------
def detect_symbol_for_frame(frame, fs,
                            freq_list=CCIR_VALUES,
                            symbol_list=CCIR_SYMBOLS,
                            band=8,
                            ratio_threshold=3.0):
    """
    Returns (symbol, max_power, second_power, idx)
    symbol = '-' if no decision.
    The algorithm uses both absolute threshold (from outside) and first/second ratio for robustness.
    """
    # windowing
    w = np.hamming(len(frame))
    frame_win = frame * w

    powers = np.array([goertzel_band(frame_win, f, fs, band=band) for f in freq_list])
    if powers.size == 0:
        return "-", 0.0, 0.0, -1

    idx = int(np.argmax(powers))
    max_p = float(powers[idx])
    # remove max to find second
    powers[idx] = 0.0
    second_p = float(np.max(powers))

    ratio = (max_p / (second_p + 1e-12)) if second_p > 0 else np.inf

    return (symbol_list[idx], max_p, second_p, idx) if ratio >= ratio_threshold else ("-", max_p, second_p, idx)

# -------------------------------
#  SET TONE MS BASED ON PROTOCOL
# -------------------------------
def set_tone_ms_for_protocol(protocollo):
    protocollo = protocollo.upper()
    if protocollo in ["CCIR-1", "PCCIR", "CCIR-2", "CCIR-7"]:
        return CCIR_CODE_LEN_MS.get(protocollo)
    elif protocollo in ["ZVEI-1", "ZVEI-2", "ZVEI-3"]:
        return ZVEI_TONE_MS
    else:
        return 100.0  # default fallback


# -------------------------------
#  DEBUG LOGGING
# -------------------------------
def debug_log(*args):
    if DEBUG_ENABLED:
        print("[DEBUG]: ", *args)

# -------------------------------
#  DECODER FUNCTION
# -------------------------------
def decode(file,
           tone_ms=None,
           overlap=0.5,
           prebandpass=True,
           bp_low=700.0,
           bp_high=2500.0,
           band=8,
           ratio_threshold=3.0,
           noise_factor=5.0,
           min_abs_power=None,
           plot=False,
           protocollo="CCIR-1"):
    """
    Selective decoding of a .wav file.
    Main parameters:
      - tone_ms: frame length (ms) used for sampling (default None)
      - overlap: fraction of overlap between frames (0..0.9), default 0.5 (50% overlap)
      - prebandpass: applies band-pass filter before analysis
      - band: search band around center frequency (Hz)
      - ratio_threshold: minimum first/second ratio to accept a decision
      - noise_factor: factor to obtain absolute threshold (threshold = median_noise * noise_factor)
      - min_abs_power: if provided, forces a minimum absolute threshold above which tones are accepted
      - debug: prints information per frame
      - plot: draws diagnostic graphs
      - cod: type of selective filter
    Returns: (final_string, frames_info)
      frames_info = list of tuples (t_start_sec, symbol, max_power, second_power, idx)
    """

    protocollo = protocollo.upper()

    match protocollo:
        case "CCIR-1" | "CCIR-2" | "CCIR-7":
            freq_list = CCIR_VALUES
            symbol_list = CCIR_SYMBOLS
            if tone_ms is None:
                tone_ms = set_tone_ms_for_protocol(protocollo)
        case "ZVEI-1":
            freq_list = ZVEI1_VALUES
            symbol_list = ZVEI1_SYMBOLS
            if tone_ms is None:
                tone_ms = set_tone_ms_for_protocol(protocollo)
        case "ZVEI-2":
            freq_list = ZVEI2_VALUES
            symbol_list = ZVEI2_SYMBOLS
            if tone_ms is None:
                tone_ms = set_tone_ms_for_protocol(protocollo)
        case "PCCIR":
            freq_list = PCCIR_VALUES
            symbol_list = PCCIR_SYMBOLS
            if tone_ms is None:
                tone_ms = set_tone_ms_for_protocol(protocollo)
        case _:
            raise ValueError(f"Codifica selettiva sconosciuta: {protocollo}")

    debug_log(f"Decodifica con protocollo: {protocollo}, lunghezza tono impostata a: {tone_ms} ms")
    

    # ==========================
    #  FILE READING
    # ==========================
    audio, fs = sf.read(file)
    # Stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(float)

    # Prefilter (optional)
    if prebandpass:
        audio = bandpass_filter(audio, fs, lowcut=bp_low, highcut=bp_high, order=4)

    tone_len = int(fs * (tone_ms / 1000.0))
    hop = int(tone_len * (1.0 - overlap))
    if hop <= 0:
        hop = max(1, tone_len // 2)

    # Frame splitting
    frames = []
    starts = list(range(0, max(0, len(audio) - tone_len + 1), hop))
    for s in starts:
        frames.append(audio[s:s + tone_len])

    #  FRAME ANALYSIS
    frame_powers = []
    frames_info = []
    for i, frame in enumerate(frames):
        # usare detect_symbol_for_frame che ritorna massimo e secondo
        symbol, max_p, second_p, idx = detect_symbol_for_frame(
            frame, fs,
            freq_list=freq_list,
            symbol_list=symbol_list,
            band=band,
            ratio_threshold=ratio_threshold
        )
        
        frame_powers.append(max_p)
        t_start = starts[i] / fs
        frames_info.append((t_start, symbol, max_p, second_p, idx))

    # Adaptive thresholding based on median noise
    median_noise = float(np.median(frame_powers))
    adaptive_threshold = median_noise * noise_factor
    if min_abs_power is not None:
        adaptive_threshold = max(adaptive_threshold, float(min_abs_power))

    debug_log(f"Fs={fs}, tone_len={tone_len} samples ({tone_ms} ms), hop={hop} samples")
    debug_log(f"Frames: {len(frames)}, median_noise={median_noise:.3e}, adaptive_threshold={adaptive_threshold:.3e}")

    # APPLY THRESHOLDING
    final_frames = []
    for (t, symbol, max_p, second_p, idx) in frames_info:
        # If below threshold, set symbol to '-'
        if max_p < adaptive_threshold:
            final_frames.append((t, "-", max_p, second_p, idx))
        else:
            # keep detected symbol
            final_frames.append((t, symbol, max_p, second_p, idx))

    # POST-PROCESSING: compression and removal of nearby duplicates
    # 1) removes ‘-’ symbols (silence) or transforms them into separators
    # 2) reduces consecutive repeated sequences to a single symbol
    seq = []
    for t_start, symbol, max_p, second_p, idx in final_frames:
        seq.append(symbol)

    # Compress consecutive duplicates (es. ['5','5','-','-','5'] -> ['5','-','5'])
    compressed = []
    for s in seq:
        if not compressed or s != compressed[-1]:
            compressed.append(s)

    # option: remove ‘-’ altogether (they can be interpreted as “silence”)
    # but to be on the safe side, let's just leave them as separators and then remove leading/trailing ones
    cleaned = [c for c in compressed if c != '-']

    final_string = "".join(cleaned)

    debug_log("Compressed frames (unique consecutive):", compressed)
    debug_log("Cleaned (no '-'): ", cleaned)
    debug_log("Final decoded string:", final_string)

    if plot:
        # plot max power per frame with decisions
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

        # histogram of frame powers (optional)
        plt.figure(figsize=(6,3))
        plt.hist(frame_powers, bins=60)
        plt.title("Histogram of frame max powers")
        plt.show()

    # RETURN FINAL STRING AND FRAMES INFO
    return final_string, final_frames

# -------------------------------
#  SELECTIVE FORMATTING
# -------------------------------
def selective_formatter(selective_string, group_size, protocol="ZVEI", format_output="MINIMAL"):
    # Pattern normalization and trimming
    selective_string = selective_string.upper()
    pattern = "4E4E"
    idx = selective_string.find(pattern)
    if idx != -1:
        selective_string = selective_string[:idx + len(pattern)]

    hex_list = list(selective_string)
    debug_log("Initial hex list:", hex_list)

    # default group_size
    group_size = group_size if group_size is not None else 5

    pause_char = None
    repeat_char = None

    match protocol:
        case "ZVEI":
            pause_char = ZVEI_TONE_CH_PAUSE
            repeat_char = ZVEI_TONE_CH_REPEATER
        case "CCIR":
            pause_char = CCIR_TONE_CH_PAUSE
            repeat_char = CCIR_TONE_CH_REPEATER
        case _:
            pause_char = ""
            repeat_char = "E"

    debug_log("Using pause char:", pause_char, "and repeat char:", repeat_char)

    new_list = []

    for i, char in enumerate(hex_list):
        debug_log("Processing char:", char, "at index:", i)
        if char == repeat_char or char == pause_char:
            # se il carattere di pausa specifico è esattamente al confine => pausa (skip)
            if pause_char and char == pause_char and i != 0 and (i % group_size) == 0:
                #debug_log("Skipping pause char at index:", i)
                continue
            # altrimenti è ripetizione: uso l'ultimo carattere valido già inserito
            if new_list:
                #debug_log("Repeating last valid char:", new_list[-1])
                new_list.append(new_list[-1])
            else:
                prev = hex_list[i - 1] if i - 1 >= 0 else ''
                #debug_log("No previous valid char, using previous from original list:", prev)
                new_list.append(prev)
        else:
            #debug_log("Adding char to new list:", char)
            new_list.append(char)

    # Grouping
    groups = [new_list[i:i + group_size] for i in range(0, len(new_list), group_size)]

    groups_str = ["".join(g) for g in groups]

    if format_output == "MINIMAL":
        return "-".join(groups_str)
    else:
        sel_src = ""
        sel_dest = ""

        if len(groups) >= 1:
            sel_src = "".join(groups[0])
        if len(groups) >= 2:
            sel_dest = "".join(groups[1])

        print("Protocol:", protocol)
        print("Pause char:", pause_char)
        print("Source:", sel_src, "(len=%d)" % len(sel_src))
        print("Dest:", sel_dest, "(len=%d)" % len(sel_dest))
        return "-".join(groups_str)


# -------------------------------
#  MAIN PROGRAM
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="[SCPD] - Selective Calling Protocol Decoder",
        usage="%(prog)s [options] <file.wav>",
        description="Selective protocol decoders CCIR-1/CCIR-2/CCIR-7, PCCIR, ZVEI-1/ZVEI-2/ZVEI-3 from .wav files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "file",
        nargs="?",
        default="./selettive_audio/00532.wav",
        help="Path to the .wav file (default: ./selettive_audio/00532.wav)",
    )

    parser.add_argument(
        "-p",
        type=str,
        default="CCIR-1",
        help="Protocol for selective decoding: CCIR-1, CCIR-2 (CCIR-7), PCCIR, ZVEI-1, ZVEI-2",
        required=True,
        choices=["CCIR-1", "CCIR-2", "CCIR-7", "PCCIR", "ZVEI-1", "ZVEI-2"],
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="MINIMAL",
        help="Format for selective formatting: Only MINIMAL (default) or COMPLETE with pauses and repetitions",
        required=True,
        choices=["MINIMAL", "COMPLETE"],
    )

    parser.add_argument(
        "-l",
        type=int,
        default=5,
        help="Selective coding length",
        required=True,
        choices=[3, 4, 5, 6],
    )

    parser.add_argument(
        "-tm",
        type=float,
        default=None,
        help="Frame length in ms. If not present, the default value for the selective choice is used.",
        choices=[70, 100]
    )

    parser.add_argument(
        "-o",
        type=float,
        default=0.5,
        help="Fractional overlap (0..0.9)",
        choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    parser.add_argument(
        "-pl",
        "--plot",
        default=False,
        action="store_true",
        help="Display diagnostic charts"
    )

    parser.add_argument(
        "-d",
        "--debug",
        default=False,
        action="store_true",
        help="Display debugging information"
    )
    parser.add_argument(
        "-nf",
        "--noise-factor",
        type=float,
        default=5.0,
        help="Adaptive noise threshold factor"
    )

    args = parser.parse_args()

    print(args)

    if args.debug:
        DEBUG_ENABLED = True

    decoded, frames = decode(
        args.file,
        tone_ms=args.tm,
        overlap=args.o,
        noise_factor=args.noise_factor,
        plot=args.plot,
        protocollo=args.p
    )

    print("Original:", decoded)
    print("Decoded:", selective_formatter(decoded, args.l, format_output=args.format))

# OK: Aggiustare frequenza basate su Gazzetta Ufficiale
# OK: Pulire codice
#           - Pulire nomi parametri
# TODO: Test Selettive registrate
# OK: Commentare codice in inglese
# OK: Correggere funzione di sostituzione tono ripetitore in base a codifica
# TODO: Blocchi GNURadio
# TODO: Codifica


