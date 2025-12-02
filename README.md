# 📡 Selective Calling
Implementazione **sistema di codifica e decodifica CCIR-7** (Selective Calling) utilizzato in ambito radio–emergenza.

## 🎯 Obiettivo del progetto

Lo scopo del progetto è fornire una pipeline completa per gestire selettive CCIR-7:

1. **Generare** file audio che rappresentano selettive CCIR-7 con 10 toni (5 destinazione + pausa + 5 mittente).
2. **Analizzare** file WAV reali, estrarre la sequenza di toni e convertirli nei caratteri CCIR-7.
3. **Correggere** eventuali errori o ripetizioni tramite interpretazione del carattere `E`.
4. **Rendere il sistema robusto** alla presenza di rumore tramite:

   * tolleranza ±10 Hz
   * filtro Goertzel a banda
   * soglia minima di potenza per filtrare i toni falsi

---

## 🛠️ Funzionalità Attuali

### ✔ Codifica CCIR-7

Lo script encoder:

* genera i 10 toni CCIR-7 (70ms ciascuno, senza spaziatura)
* rispetta le frequenze ufficiali di ogni simbolo
* salva un file `.wav` contenente la selettiva completa

---

### ✔ Decodifica CCIR-7 da file WAV

Lo script decoder è in grado di:

* scansionare l’intera traccia e cercare sequenze compatibili con CCIR-7
* stimare la frequenza dominante tramite **Goertzel Band-pass** ±10Hz
* scartare automaticamente i toni con intensità troppo bassa
* ricostruire la sequenza originale di simboli
* applicare la regola: `carattere E = ripetizione del tono precedente`
* ricostruire la selettiva finale corretta (es. `1E501 → 11501`)

---

### ✔ Robustezza al rumore

Il decoder:

* Utilizza **Goertzel band-pass** con tolleranza ±10Hz
* Usa una **soglia minima di potenza** per discriminare segnali reali da rumore
* Impedisce riconoscimenti errati in presenza di interferenze o audio di bassa qualità

---

## 📂 Struttura del progetto

```
📁 SelectiveCalling/
 ├── selettive_audio            # Cartella contenente file WAV di selettive di esempio
 │    ├── 00259.wav
 │    ├── 00529.wav
 │    ├── 00532.wav
 │    ├── 00841.wav
 │    └── 23533.wav
 └── src
     ├── ccir-codifica.py       # Script per generare selettive CCIR-7 (WAV)
     ├── ccir-decodifica.py     # Script per rilevare e decodificare selettive da un file audio
     └── py5mon.py
```

---

## ▶️ Uso del Codificatore: TODO

```bash
python3 encoder.py --code 2353E --output selettiva.wav
```

Il file prodotto conterrà la selettiva CCIR-7 per il codice `2353E`.

---

## ▶️ Uso del Decodificatore: TODO

```bash
python3 decoder.py --input registrazione.wav
```

Output tipico:

```
Selettiva rilevata!
Chiamante: 11501
Chiamato: 90515
```

---

## 🚀 Possibili Sviluppi Futuri

* [ ] Analisi multi-standard (ZVEI-1, ZVEI-2, CCIR-1)
* [ ] Creazione blocco GNURadio per decodifica in tempo reale
---

## 📜 Licenza

Questo progetto è rilasciato sotto la licenza MIT. Vedi il file [LICENSE](LICENSE) per i dettagli.


## Autori
- Pietro Rocchio
- Andrea De Cao
- Juri Marku
