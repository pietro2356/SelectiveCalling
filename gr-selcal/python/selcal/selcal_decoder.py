"""
Embedded Python Block: SelCall Decoder
"""

import numpy as np
from gnuradio import gr
import pmt
import time

# Importiamo la logica esistente (assumendo che i file siano nella stessa cartella o nel path)
# Se usati in un modulo OOT standard, l'import potrebbe variare (es: from my_module import ...)
from .SelectiveCalling import SelectiveCalling
from .protocolli.CCIR import (
    CCIR_VALUES,
    CCIR_SYMBOLS,
    CCIR_CODE_LEN_MS,
    PCCIR_VALUES,
    PCCIR_SYMBOLS,
    CCIR_TONE_CH_PAUSE,
    CCIR_TONE_CH_REPEATER
)
from .protocolli.ZVEI import (
    ZVEI1_VALUES,
    ZVEI1_SYMBOLS,
    ZVEI2_VALUES,
    ZVEI2_SYMBOLS,
    ZVEI_TONE_MS,
    ZVEI_TONE_CH_PAUSE,
    ZVEI_TONE_CH_REPEATER
)


class selcal_decoder(gr.sync_block):
    """
    SelCall Decoder Block per GNU Radio.
    Decodifica CCIR/ZVEI in tempo reale e controlla un gate audio.
    """

    def __init__(self,
                 sample_rate=48000,
                 protocol="ZVEI-1",
                 target_code="50101",
                 code_length=5,
                 tone_duration_ms=0.0,
                 debug=False):

        # Inizializzazione blocco: 1 input float, 1 output float
        gr.sync_block.__init__(
            self,
            name='SelCall Decoder',
            in_sig=[np.float32],
            out_sig=[np.float32]
        )

        # --- Parametri Utente ---
        self.fs = sample_rate
        self.protocol = protocol
        self.target_code = target_code
        self.code_length = code_length
        self.user_tone_ms = tone_duration_ms
        self.debug_mode = debug

        # --- OTTIMIZZAZIONE: Decimazione ---
        # Analizziamo l'audio a 8kHz invece che 48kHz.
        # Fattore 6: 48000 / 6 = 8000 Hz.
        self.decim_factor = 6
        self.fs_analysis = self.fs / self.decim_factor

        # --- Setup Logica Selettiva ---
        self.decoder_lib = SelectiveCalling(debug=debug)
        self.freq_list = []
        self.symbol_list = []
        self.tone_ms = 100.0  # Default
        self._configure_protocol()

        # --- Message Port ---
        self.message_port_name = pmt.intern("selcall_out")
        self.message_port_register_out(self.message_port_name)

        # --- Audio Gate State ---
        self.gate_open = False
        self.gate_timer_samples = 0
        self.gate_duration_samples = int(20.0 * self.fs)  # 20 secondi

        # Buffer per l'analisi Goertzel
        # La finestra di analisi deve essere grande circa quanto un tono
        self.samples_per_tone = int(self.fs * (self.tone_ms / 1000.0))
        # Hop size: ogni quanto facciamo l'analisi (es. 50% overlap)
        self.hop_size = max(1, self.samples_per_tone // 2)

        # Buffer interno per accumulare i campioni
        self.internal_buffer = np.array([], dtype=np.float32)

        # --- State Machine Decodifica ---
        self.detected_symbols_history = []  # Lista di (simbolo, potenza)
        self.last_valid_sequence = ""

        # Soglia rumore adattiva (Running Average)
        self.avg_noise_power = 10.0  # Valore iniziale arbitrario

        if self.debug_mode:
            print(f"[SelCall] Init: Protocol={protocol}, Tone={self.tone_ms}ms, Gate={self.target_code}")

    def _configure_protocol(self):
        """ Configura le tabelle di frequenza in base al protocollo scelto """
        p = self.protocol.upper()

        if p == "ZVEI-1":
            self.freq_list = ZVEI1_VALUES
            self.symbol_list = ZVEI1_SYMBOLS
            base_ms = ZVEI_TONE_MS
        elif p == "ZVEI-2":
            self.freq_list = ZVEI2_VALUES
            self.symbol_list = ZVEI2_SYMBOLS
            base_ms = ZVEI_TONE_MS
        elif p in ["CCIR-1", "CCIR-2", "CCIR-7"]:
            self.freq_list = CCIR_VALUES
            self.symbol_list = CCIR_SYMBOLS
            base_ms = CCIR_CODE_LEN_MS.get(p, 100)
        elif p == "PCCIR":
            self.freq_list = PCCIR_VALUES
            self.symbol_list = PCCIR_SYMBOLS
            base_ms = 100
        else:
            # Fallback
            self.freq_list = ZVEI1_VALUES
            self.symbol_list = ZVEI1_SYMBOLS
            base_ms = 70

        # Override se l'utente ha specificato una durata custom (>0)
        if self.user_tone_ms > 0:
            self.tone_ms = self.user_tone_ms
        else:
            self.tone_ms = base_ms

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        n_samples = len(in0)

        # 1. Accumulo Dati (No Filtering Interno)
        # Assumiamo in0 già filtrato dal blocco Bandpass Filter (C++) precedente
        # Concateniamo direttamente l'input grezzo (che ora è già filtrato)
        self.internal_buffer = np.concatenate((self.internal_buffer, in0))

        # Processiamo finché abbiamo abbastanza dati per una finestra
        while len(self.internal_buffer) >= self.samples_per_tone:
            # Estrai finestra a piena risoluzione (48k)
            frame_48k = self.internal_buffer[:self.samples_per_tone]

            # --- OTTIMIZZAZIONE QUI ---
            # Creiamo una versione "leggera" per l'analisi: prendiamo 1 campione ogni 6
            frame_analysis = frame_48k[::self.decim_factor]

            # Esegui Goertzel (riutilizzando la logica della classe SelectiveCalling)
            symbol, max_p, second_p, idx = self.decoder_lib.detect_symbol_for_frame(
                frame_analysis,
                self.fs_analysis,
                freq_list=self.freq_list,
                symbol_list=self.symbol_list,
                band=8,
                ratio_threshold=2.5  # Leggermente più basso per real-time
            )

            # Aggiornamento soglia rumore (filtro esponenziale lento)
            # Se la potenza è bassa, probabilmente è rumore
            if max_p < (self.avg_noise_power * 20):
                self.avg_noise_power = 0.95 * self.avg_noise_power + 0.05 * max_p

            adaptive_thresh = self.avg_noise_power * 8.0  # Soglia attivazione

            valid_symbol = "-"
            if max_p > adaptive_thresh and max_p > 100.0:  # Hard floor minimo
                valid_symbol = symbol

            # Aggiungi alla storia dei simboli rilevati
            self._process_symbol_stream(valid_symbol, max_p)

            # Shift del buffer (rimane basato sui campioni reali a 48k)
            self.internal_buffer = self.internal_buffer[self.hop_size:]

        # --- 3. Gestione Audio Gate ---
        if self.gate_open:
            if self.gate_timer_samples > 0:
                # Copia input in output
                out0[:] = in0[:]
                self.gate_timer_samples -= n_samples
            else:
                # Tempo scaduto
                self.gate_open = False
                out0[:] = 0.0  # Mute
                if self.debug_mode:
                    print("[SelCall] Gate chiuso (timeout).")
        else:
            # Mute
            out0[:] = 0.0

        return n_samples

    def _process_symbol_stream(self, symbol, power):
        """ 
        Logica per ricostruire la stringa dai simboli grezzi (debouncing)
        e verificare il target code.
        """
        # Aggiungiamo il simbolo raw alla lista temporanea
        self.detected_symbols_history.append(symbol)

        # Manteniamo la lista corta (es. ultimi 50 frame) per non saturare memoria
        if len(self.detected_symbols_history) > 100:
            self.detected_symbols_history.pop(0)

        # Logica semplificata di "End of Sequence":
        # Se riceviamo silenzio ("-") per un po', proviamo a parsare ciò che abbiamo accumulato.
        # Oppure se il buffer è pieno.

        # Controlliamo se gli ultimi N simboli sono silenzio -> fine trasmissione probabile
        suffix_len = 4
        if len(self.detected_symbols_history) > suffix_len:
            last_n = self.detected_symbols_history[-suffix_len:]
            if all(s == "-" for s in last_n):
                # Tentativo di estrazione stringa
                raw_seq = [s for s in self.detected_symbols_history if s != "-"]

                if not raw_seq:
                    return  # Niente da processare

                # Compressione (Run Length Encoding implicito)
                # Uniamo simboli uguali adiacenti
                compressed = []
                if raw_seq:
                    prev = raw_seq[0]
                    compressed.append(prev)
                    for s in raw_seq[1:]:
                        if s != prev:
                            compressed.append(s)
                            prev = s

                final_str = "".join(compressed)

                # Filtro lunghezza minima (es. almeno 3 caratteri per essere una selettiva valida)
                if len(final_str) >= 3:
                    # Evita di riprocessare la stessa stringa identica consecutiva troppe volte
                    if final_str != self.last_valid_sequence:
                        self.last_valid_sequence = final_str
                        self._analyze_sequence(final_str)

                        # Reset history dopo un match/process valido
                        self.detected_symbols_history = []

    def _analyze_sequence(self, decoded_string):
        """ 
        Analizza la stringa compressa, applica la logica del formatter 
        e decide se aprire il gate.
        """
        # Usa il formatter della libreria per gestire ripetizioni (E) e pause
        # Nota: selective_formatter ritorna stringa formattata (es. 12345-50101)
        formatted_str = self.decoder_lib.selective_formatter(
            decoded_string,
            group_size=self.code_length,
            protocol=self.protocol,
            format_output="MINIMAL"
        )

        # Rimuoviamo il separatore "-" per il controllo semplice
        clean_str = formatted_str.replace("-", "")
        target = self.target_code.upper()

        match_found = False

        # Logica di Match:
        # La selettiva è tipicamente SRC + DEST + EOS.
        # Controlliamo se il TARGET è presente nella stringa decodificata
        if target in clean_str:
            match_found = True
            self.gate_open = True
            self.gate_timer_samples = self.gate_duration_samples  # Reset timer (Retrigger)
            if self.debug_mode:
                print(f"[SelCall] MATCH! Target {target} trovato in {clean_str}. Gate APERTO.")

        # Invio Messaggio PMT
        self._send_message(formatted_str, match_found)

    def _send_message(self, decoded_code, match_status):
        """ Invia un messaggio asincrono sulla porta 'measurements' """
        timestamp = time.time()

        # Costruzione dizionario metadati
        meta = pmt.make_dict()
        meta = pmt.dict_add(meta, pmt.intern("timestamp"), pmt.from_double(timestamp))
        meta = pmt.dict_add(meta, pmt.intern("protocol"), pmt.intern(self.protocol))
        meta = pmt.dict_add(meta, pmt.intern("gate_active"), pmt.from_bool(match_status))
        meta = pmt.dict_add(meta, pmt.intern("code"), pmt.intern(decoded_code))

        # PMT Pair: (Code_String, Metadata_Dict) o viceversa, standard GNU Radio è spesso un cons
        # Qui inviamo una coppia (stringa_codice, metadati)
        msg = pmt.cons(pmt.intern(decoded_code), meta)

        self.message_port_pub(self.message_port_name, msg)