import argparse
from src.core.SelectiveCalling import SelectiveCalling

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

    if args.debug:
        DEBUG_ENABLED = True

    selcal = SelectiveCalling(debug=args.debug)

    decoded, frames = selcal.decode(
        args.file,
        tone_ms=args.tm,
        overlap=args.o,
        noise_factor=args.noise_factor,
        plot=args.plot,
        protocollo=args.p
    )

    print("Original:", decoded)
    print("Decoded:", selcal.selective_formatter(decoded, args.l, format_output=args.format))

# OK: Aggiustare frequenza basate su Gazzetta Ufficiale
# OK: Pulire codice
#           - Pulire nomi parametri
# TODO: Test Selettive registrate
# OK: Commentare codice in inglese
# OK: Correggere funzione di sostituzione tono ripetitore in base a codifica
# TODO: Blocchi GNURadio
# TODO: Codifica


