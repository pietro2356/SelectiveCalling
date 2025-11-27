# CHANGELOG
File contenente i cambiamenti significativi apportati al progetto e note operative.
> Formato data: `YYYY-MM-DD`

## [0.0.1] - 2025-11-27
### Aggiunto
- Implementazione filtro Goertzel con banda per rilevamento frequenze specifiche.

### Modificato
- Oggetto `CCIR7_FREQS` contentente le i codici associati alle frequenze standard CCIR.

### Miglioramenti da fare
- Separare la lettura del file wav e l'elaborazione del segnale in funzioni distinti.
- Utilizzare una libbreria diversa per la lettura del file wav per migliorare la compatibilità.
    - Per esempio, `scipy.io.wavfile`.
- Visualizzare un grafico delle frequenze rilevate nel tempo.

### Problemi noti
- La funzione di lettura del file wav e la sua separazione in blocchi di tot millisec non è ottimale e/o sta causando problemi.