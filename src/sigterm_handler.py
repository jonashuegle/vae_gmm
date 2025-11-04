import signal, sys
def handler(signum, frame):
    print("SIGUSR1 empfangen, speichere Checkpoint…")
    # hier Trainer.save_checkpoint(...) aufrufen, falls nötig
    sys.exit(0)
signal.signal(signal.SIGUSR1, handler)
