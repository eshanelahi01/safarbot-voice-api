def transcribe_audio(_: bytes) -> str:
    raise NotImplementedError(
        "STT is optional in this service. Add faster-whisper and wire audio handling when needed."
    )
