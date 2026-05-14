
"""
Text-to-speech synthesis supporting two backends:

  TTS_PROVIDER=cloud  (default) — Google Cloud Text-to-Speech, produces MP3.
  TTS_PROVIDER=gemini            — Google Gemini TTS, produces WAV.

Cloud requirements:
    pip install google-cloud-texttospeech
    Set GOOGLE_APPLICATION_CREDENTIALS in .env pointing to the service account JSON key.

Gemini requirements:
    pip install google-genai
    Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env.
"""

import asyncio
import os
import pathlib
import re
import wave

from utils.logger import get_debug_logger

logger = get_debug_logger(
    "tts_processor",
    pathlib.Path(__file__).parent.resolve() / "../logs/server.log",
)

# language_code (from UI dropdown) → (cloud_lang, cloud_voice, gemini_voice)
_LANGUAGE_VOICE_MAP = {
    "en-US": ("en-US", "en-US-Neural2-F",  "Kore"),
    "si-LK": ("si-LK", "si-LK-Standard-A", "Kore"),
    "ta-LK": ("ta-IN", "ta-IN-Neural2-A",  "Kore"),
    "ta-IN": ("ta-IN", "ta-IN-Neural2-A",  "Kore"),
    "zh":    ("cmn-CN", "cmn-CN-Wavenet-A", "Kore"),
    "zh-CN": ("cmn-CN", "cmn-CN-Wavenet-A", "Kore"),
    "de-DE": ("de-DE", "de-DE-Neural2-F",  "Kore"),
    "fr-FR": ("fr-FR", "fr-FR-Neural2-C",  "Kore"),
    "ja-JP": ("ja-JP", "ja-JP-Neural2-B",  "Kore"),
}


def _write_wav(filename: str, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def _clean_for_tts(text: str) -> str:
    """Strip markdown/HTML so the TTS model receives plain speakable text."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#~]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def _synthesize_cloud(
    text: str,
    output_file: str,
    language_code: str,
    speaking_rate: float,
    pitch: float,
) -> str:
    from google.cloud import texttospeech
    from google.oauth2 import service_account

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        credentials = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = texttospeech.TextToSpeechAsyncClient(credentials=credentials)
    else:
        client = texttospeech.TextToSpeechAsyncClient()

    lang, voice_name, _ = _LANGUAGE_VOICE_MAP.get(language_code, ("en-US", "en-US-Neural2-F", "Kore"))
    logger.info(f"Cloud TTS | lang={lang} voice={voice_name} chars={len(text)} output={output_file}")

    response = await client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(language_code=lang, name=voice_name),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
            pitch=pitch,
        ),
    )

    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(response.audio_content)

    logger.info(f"Cloud TTS | saved {len(response.audio_content)} bytes → {output_file}")
    return output_file


async def _synthesize_gemini(
    text: str,
    output_file: str,
    language_code: str,
) -> str:
    from google import genai
    from google.genai import types

    clean_text = _clean_for_tts(text)
    if not clean_text:
        raise ValueError("Text is empty after cleaning; cannot synthesize speech.")

    base, _ = os.path.splitext(output_file)
    wav_file = base + ".wav"

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    logger.info(f"Gemini TTS | chars={len(clean_text)} output={wav_file}")

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=clean_text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Leda')
                )
            ),
        ),
    )

    pcm = response.candidates[0].content.parts[0].inline_data.data

    out_dir = os.path.dirname(os.path.abspath(wav_file))
    os.makedirs(out_dir, exist_ok=True)
    _write_wav(wav_file, pcm)

    logger.info(f"Gemini TTS | saved {len(pcm)} bytes → {wav_file}")
    return wav_file


async def synthesize_text(
    text: str,
    output_file: str,
    language_code: str = "si-LK",
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
) -> str:
    """
    Async: synthesize text to an audio file.

    Backend selected by TTS_PROVIDER env var:
      - "cloud"  (default): Google Cloud TTS → MP3
      - "gemini":           Google Gemini TTS → WAV
    """
    provider = os.environ.get("TTS_PROVIDER", "cloud").lower()
    logger.debug(f"synthesize_text | provider={provider} lang={language_code} output={output_file}")

    if provider == "gemini":
        return await _synthesize_gemini(text, output_file, language_code)

    if provider == "cloud":
        return await _synthesize_cloud(text, output_file, language_code, speaking_rate, pitch)

    logger.error(f"synthesize_text | unknown TTS_PROVIDER={provider!r}, falling back to cloud")
    return await _synthesize_cloud(text, output_file, language_code, speaking_rate, pitch)
