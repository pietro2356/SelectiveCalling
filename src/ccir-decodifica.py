import numpy as np
from scipy.io import wavfile
import math

class GoertzelFilter:
    def __init__(self, target_freq, sample_rate, num_samples):
        """
        Pre-calcola i coefficienti di Goertzel per efficienza.
        """
        self.target_freq = target_freq
        self.k = int(0.5 + ((num_samples * target_freq) / sample_rate))
        self.omega = (2 * math.pi * self.k) / num_samples
        self.coeff = 2 * math.cos(self.omega)
        
    def process(self, samples):
        """
        Esegue l'algoritmo di Goertzel su un blocco di campioni.
        Ritorna la "potenza" (magnitude squared) della frequenza target.
        """
        s_prev = 0.0
        s_prev2 = 0.0
        
        for sample in samples:
            s = sample + (self.coeff * s_prev) - s_prev2
            s_prev2 = s_prev
            s_prev = s
            
        power = s_prev2**2 + s_prev**2 - self.coeff * s_prev * s_prev2
        return power

# --- CONFIGURAZIONE ---

# Frequenze CCIR Standard
CCIR_FREQS = {
    '1': 1124, '2': 1197, '3': 1275, '4': 1358, '5': 1446,
    '6': 1540, '7': 1640, '8': 1747, '9': 1860, '0': 1981,
    'A': 2400, 'B': 930,  'C': 2247, 'D': 991,  'E': 2110
}

FILE_PATH = './selettive_audio/00532.wav' #Inserisci il tuo file qui
TONE_DURATION_MS = 100 
WINDOW_MS = 40          # Analizziamo finestre più piccole del tono (40ms)
OVERLAP = 0.5           # Sovrapposizione del 50% per non perdere attacchi

# --- MAIN ---

def decode_ccir_goertzel():
    # 1. Caricamento Audio
    fs, data = wavfile.read(FILE_PATH)
    
    # Converti in mono e normalizza tra -1 e 1
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data / np.max(np.abs(data))

    # 2. Calcolo dimensioni finestre
    N = int(fs * (WINDOW_MS / 1000)) # Numero campioni per blocco Goertzel
    step = int(N * (1 - OVERLAP))
    
    print(f"Sample Rate: {fs} Hz | Block Size: {N} samples")

    # 3. Inizializzazione Filtri Goertzel
    # Creiamo un banco di filtri, uno per ogni tono CCIR
    filters = {}
    for char, freq in CCIR_FREQS.items():
        filters[char] = GoertzelFilter(freq, fs, N)

    decoded_sequence = []
    last_detected = None
    consecutive_hits = 0
    REQUIRED_HITS = 2  # Quante finestre consecutive devono validare il tono?

    # 4. Loop di Analisi (Sliding Window)
    for i in range(0, len(data) - N, step):
        chunk = data[i : i + N]
        
        max_power = 0
        best_char = None
        
        # Eseguiamo tutti i filtri sul blocco corrente
        for char, filter_obj in filters.items():
            power = filter_obj.process(chunk)
            if power > max_power:
                max_power = power
                best_char = char
        
        # Soglia dinamica o fissa (regolare questo valore sperimentale)
        # Il valore di Goertzel scala con N^2, quindi sarà alto.
        THRESHOLD = 10000000 # Esempio per segnale normalizzato a N=1764 (40ms @ 44.1k)
        # Se usi normalizzazione 0-1, la soglia potrebbe essere molto più bassa, es. 100
        # Qui usiamo una soglia relativa dinamica per robustezza:
        
        if max_power > 1000 : # Soglia minima assoluta "silenzio"
            if best_char == last_detected:
                consecutive_hits += 1
            else:
                consecutive_hits = 1
                last_detected = best_char
            
            # Se abbiamo trovato lo stesso tono per abbastanza finestre consecutive
            # e non lo abbiamo appena aggiunto alla sequenza finale
            if consecutive_hits == REQUIRED_HITS:
                # Gestione ripetizione (CCIR usa 'E' per ripetere l'ultimo)
                token = best_char
                
                # Se l'ultimo aggiunto è lo stesso, ignoralo (è ancora lo stesso tono da 100ms)
                # A meno che non sia passato molto tempo (qui semplificato)
                if not decoded_sequence or decoded_sequence[-1] != token:
                     # Logica specifica per il tono 'E' (Repeat)
                    if token == 'E' and decoded_sequence:
                        actual_token = decoded_sequence[-1]
                        decoded_sequence.append(actual_token)
                        print(f"Tono Rilevato: {token} (Ripetizione di {actual_token})")
                    else:
                        decoded_sequence.append(token)
                        print(f"Tono Rilevato: {token} (Power: {max_power:.0f})")

        else:
            consecutive_hits = 0 # Reset su silenzio
            
    print("" * 30)
    print(f"RISULTATO: {''.join(decoded_sequence)}")

if __name__ == "__main__":
    try:
        decode_ccir_goertzel()
    except FileNotFoundError:
        print("Errore: File non trovato. Modifica 'FILE_PATH' nello script.")
    except Exception as e:
        print(f"Errore: {e}")