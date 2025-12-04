from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from ..protocolli.CCIR import (
    CCIR_SYMBOLS,
    CCIR_VALUES,
    PCCIR_VALUES,
    PCCIR_SYMBOLS,
    CCIR_CODE_LEN_MS,
    CCIR_TONE_CH_PAUSE,
    CCIR_TONE_CH_REPEATER
)
from ..protocolli.ZVEI import (
    ZVEI1_VALUES,
    ZVEI1_SYMBOLS,
    ZVEI2_VALUES,
    ZVEI2_SYMBOLS,
    ZVEI_TONE_MS,
    ZVEI_TONE_CH_PAUSE,
    ZVEI_TONE_CH_REPEATER
)


class SelectiveCalling:
    def __init__(self, debug: bool = False):
        self.debug = debug

    # -------------------------------
    #  DEBUG
    # -------------------------------
    def _debug(self, *args):
        if self.debug:
            print("[DEBUG]:", *args)

    # -------------------------------
    #  BANDPASS FILTER (butterworth)
    # -------------------------------
    @staticmethod
    def bandpass_filter(signal: np.ndarray, fs: int, lowcut=700.0, highcut=2500.0, order=4) -> np.ndarray:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    # -------------------------------
    #  GOERTZEL
    # -------------------------------
    @staticmethod
    def goertzel(samples: np.ndarray, sample_rate: int, freq: float) -> float:
        n = len(samples)
        if n == 0:
            return 0.0
        k = int(0.5 + (n * freq) / sample_rate)
        omega = (2.0 * np.pi * k) / n
        coeff = 2.0 * np.cos(omega)
        s_prev = 0.0
        s_prev2 = 0.0
        for x in samples:
            s = x + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
        return float(np.abs(power))

    @classmethod
    def goertzel_band(cls, samples: np.ndarray, center_freq: float, fs: int, band=8, steps=5) -> float:
        freqs = np.linspace(center_freq - band, center_freq + band, steps)
        powers = [cls.goertzel(samples, fs, f) for f in freqs]
        return max(powers)

    # -------------------------------
    #  SYMBOL DETECTION
    # -------------------------------
    def detect_symbol_for_frame(self, frame: np.ndarray, fs: int,
                                freq_list: List[float],
                                symbol_list: List[str],
                                band=8,
                                ratio_threshold=3.0) -> Tuple[str, float, float, int]:
        w = np.hamming(len(frame))
        frame_win = frame * w
        powers = np.array([self.goertzel_band(frame_win, f, fs, band=band) for f in freq_list])
        if powers.size == 0:
            return "-", 0.0, 0.0, -1
        idx = int(np.argmax(powers))
        max_p = float(powers[idx])
        powers[idx] = 0.0
        second_p = float(np.max(powers))
        ratio = (max_p / (second_p + 1e-12)) if second_p > 0 else np.inf
        return (symbol_list[idx], max_p, second_p, idx) if ratio >= ratio_threshold else ("-", max_p, second_p, idx)

    # -------------------------------
    #  TONE LENGTH PER PROTOCOL
    # -------------------------------
    @staticmethod
    def set_tone_ms_for_protocol(protocollo: str):
        protocollo = protocollo.upper()
        if protocollo in ["CCIR-1", "PCCIR", "CCIR-2", "CCIR-7"]:
            return CCIR_CODE_LEN_MS.get(protocollo)
        elif protocollo.startswith("ZVEI"):
            return ZVEI_TONE_MS
        else:
            return 100.0

    # -------------------------------
    #  DECODE WAV FILE
    # -------------------------------
    def decode(self, file: str,
               tone_ms: Optional[float] = None,
               overlap: float = 0.5,
               prebandpass: bool = True,
               bp_low: float = 700.0,
               bp_high: float = 2500.0,
               band: int = 8,
               ratio_threshold: float = 3.0,
               noise_factor: float = 5.0,
               min_abs_power: Optional[float] = None,
               plot: bool = False,
               protocollo: str = "CCIR-1"):
        protocollo = protocollo.upper()

        if protocollo in ["CCIR-1", "CCIR-2", "CCIR-7"]:
            freq_list = CCIR_VALUES
            symbol_list = CCIR_SYMBOLS
        elif protocollo == "ZVEI-1":
            freq_list = ZVEI1_VALUES
            symbol_list = ZVEI1_SYMBOLS
        elif protocollo == "ZVEI-2":
            freq_list = ZVEI2_VALUES
            symbol_list = ZVEI2_SYMBOLS
        elif protocollo == "PCCIR":
            freq_list = PCCIR_VALUES
            symbol_list = PCCIR_SYMBOLS
        else:
            raise ValueError(f"Codifica selettiva sconosciuta: {protocollo}")

        if tone_ms is None:
            tone_ms = self.set_tone_ms_for_protocol(protocollo)

        self._debug(f"Protocol: {protocollo}, tone_ms: {tone_ms}")

        audio, fs = sf.read(file)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(float)

        if prebandpass:
            audio = self.bandpass_filter(audio, fs, lowcut=bp_low, highcut=bp_high, order=4)

        tone_len = int(fs * (tone_ms / 1000.0))
        hop = int(tone_len * (1.0 - overlap))
        if hop <= 0:
            hop = max(1, tone_len // 2)

        frames = []
        starts = list(range(0, max(0, len(audio) - tone_len + 1), hop))
        for s in starts:
            frames.append(audio[s:s + tone_len])

        frame_powers = []
        frames_info = []
        for i, frame in enumerate(frames):
            symbol, max_p, second_p, idx = self.detect_symbol_for_frame(
                frame, fs,
                freq_list=freq_list,
                symbol_list=symbol_list,
                band=band,
                ratio_threshold=ratio_threshold
            )
            frame_powers.append(max_p)
            t_start = starts[i] / fs
            frames_info.append((t_start, symbol, max_p, second_p, idx))

        median_noise = float(np.median(frame_powers))
        adaptive_threshold = median_noise * noise_factor
        if min_abs_power is not None:
            adaptive_threshold = max(adaptive_threshold, float(min_abs_power))

        self._debug(f"Fs={fs}, tone_len={tone_len}, hop={hop}, frames={len(frames)}, median_noise={median_noise:.3e}")

        final_frames = []
        for (t, symbol, max_p, second_p, idx) in frames_info:
            if max_p < adaptive_threshold:
                final_frames.append((t, "-", max_p, second_p, idx))
            else:
                final_frames.append((t, symbol, max_p, second_p, idx))

        # compress consecutive duplicates and remove '-' separators
        seq = [s for (_, s, _, _, _) in final_frames]
        compressed = []
        for s in seq:
            if not compressed or s != compressed[-1]:
                compressed.append(s)
        cleaned = [c for c in compressed if c != '-']
        final_string = "".join(cleaned)

        # optional plotting
        if plot:
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
            plt.title("Selective detection (frame max power and decisions)")
            plt.legend()
            plt.grid(True)
            plt.show()

        return final_string, final_frames

    # -------------------------------
    #  SELECTIVE FORMATTING
    # -------------------------------
    def selective_formatter(self, selective_string: str, group_size: Optional[int], protocol: str = "ZVEI", format_output: str = "MINIMAL") -> str:
        selective_string = selective_string.upper()
        # trim after known terminator pattern if present
        pattern = "4E4E"
        idx = selective_string.find(pattern)
        if idx != -1:
            selective_string = selective_string[:idx + len(pattern)]

        hex_list = list(selective_string)
        self._debug("Initial hex list:", hex_list)

        group_size = group_size if group_size is not None else 5

        # decide pause and repeat chars based on protocol name prefix
        p = protocol.upper()
        if p.startswith("ZVEI"):
            pause_char = ZVEI_TONE_CH_PAUSE
            repeat_char = ZVEI_TONE_CH_REPEATER
        elif p.startswith("CCIR") or p == "PCCIR":
            pause_char = CCIR_TONE_CH_PAUSE
            repeat_char = CCIR_TONE_CH_REPEATER
        else:
            pause_char = ""
            repeat_char = "E"

        # if protocol defines empty pause char, also consider space as pause candidate
        pause_candidates = {pause_char} if pause_char else {'' , ' '}
        # normalize empty-string candidate removal for direct comparisons
        if '' in pause_candidates:
            pause_candidates.discard('')
        self._debug("Pause candidates:", pause_candidates, "Repeat char:", repeat_char)

        new_list = []
        for i, char in enumerate(hex_list):
            self._debug("Processing char:", char, "index:", i)
            # if char equals repeater or matches a pause candidate
            if char == repeat_char or char in pause_candidates:
                # if char is a pause candidate and is located exactly at a group boundary -> skip (separator)
                if (char in pause_candidates) and i != 0 and (i % group_size) == 0:
                    self._debug("Skipping pause char at group boundary index:", i)
                    continue
                # special rule: if repeater char equals 'E' (or same as pause) but it's at boundary => treat as pause (skip)
                if char == repeat_char and (i != 0 and (i % group_size) == 0):
                    self._debug("Repeater at group boundary treated as pause, skipping index:", i)
                    continue
                # otherwise it's a repetition: repeat last valid char (or fallback to previous original)
                if new_list:
                    new_list.append(new_list[-1])
                else:
                    prev = hex_list[i - 1] if i - 1 >= 0 else ''
                    new_list.append(prev)
            else:
                new_list.append(char)

        # grouping into chunks of group_size
        groups = [new_list[i:i + group_size] for i in range(0, len(new_list), group_size)]
        groups_str = ["".join(g) for g in groups]

        if format_output == "MINIMAL":
            return "-".join(groups_str)
        else:
            sel_src = "".join(groups[0]) if len(groups) >= 1 else ""
            sel_dest = "".join(groups[1]) if len(groups) >= 2 else ""
            print("Protocol:", protocol)
            print("Pause char(s):", pause_candidates)
            print("Source:", sel_src, "(len=%d)" % len(sel_src))
            print("Dest:", sel_dest, "(len=%d)" % len(sel_dest))
            return "-".join(groups_str)
