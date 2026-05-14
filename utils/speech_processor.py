
"""
Speech-to-text transcription using Google Cloud Speech-to-Text.

Requirements:
    pip install google-cloud-speech

Set GOOGLE_APPLICATION_CREDENTIALS in .env pointing to the service account JSON key.
"""

import os
from google.cloud import speech
from google.oauth2 import service_account

# MIME type → Google Speech encoding
_MIME_ENCODING = {
    "audio/webm": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
    "audio/webm;codecs=opus": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
    "audio/ogg": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
    "audio/ogg;codecs=opus": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
    "audio/mp3": speech.RecognitionConfig.AudioEncoding.MP3,
    "audio/mpeg": speech.RecognitionConfig.AudioEncoding.MP3,
    "audio/flac": speech.RecognitionConfig.AudioEncoding.FLAC,
    "audio/wav": speech.RecognitionConfig.AudioEncoding.LINEAR16,
    "audio/x-wav": speech.RecognitionConfig.AudioEncoding.LINEAR16,
    "audio/l16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
}

# MIME type → default sample rate
_MIME_SAMPLE_RATE = {
    "audio/webm": 48000,
    "audio/webm;codecs=opus": 48000,
    "audio/ogg": 48000,
    "audio/ogg;codecs=opus": 48000,
    "audio/mp3": 16000,
    "audio/mpeg": 16000,
    "audio/flac": 16000,
    "audio/wav": 16000,
    "audio/x-wav": 16000,
    "audio/l16": 16000,
}


def transcribe_audio_bytes(
    audio_bytes: bytes,
    mime_type: str = "audio/webm",
    language_code: str = "en-US",
    sample_rate: int = None,
) -> str:
    """
    Transcribe uploaded audio bytes using Google Cloud Speech-to-Text.

    Args:
        audio_bytes: Raw audio file contents.
        mime_type: MIME type reported by the upload (e.g. "audio/webm").
        language_code: BCP-47 language tag (e.g. "en-US", "si-LK").
        sample_rate: Override sample rate; auto-selected from mime_type when None.

    Returns:
        Transcribed text, or empty string when nothing was detected.
    """
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        credentials = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = speech.SpeechClient(credentials=credentials)
    else:
        client = speech.SpeechClient()

    key = mime_type.lower().strip()
    encoding = _MIME_ENCODING.get(key, speech.RecognitionConfig.AudioEncoding.WEBM_OPUS)
    if sample_rate is None:
        sample_rate = _MIME_SAMPLE_RATE.get(key, 48000)

    config = speech.RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=sample_rate,
        language_code=language_code,
        enable_automatic_punctuation=True,
    )

    response = client.recognize(
        config=config,
        audio=speech.RecognitionAudio(content=audio_bytes),
    )

    return " ".join(r.alternatives[0].transcript for r in response.results)
