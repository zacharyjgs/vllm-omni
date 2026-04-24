"""Streaming SSE endpoint that returns TTS audio interleaved with word-level
timing events.

The Qwen3-TTS Talker front-loads text-token consumption (one token per decode
step while text vectors remain, then pad embeddings).  Token consumption rate
does NOT correspond to audible speech rate -- the model generates the full
utterance across all frames after consuming the text.

To produce accurate word-level timing we therefore:

1.  Stream audio chunks to the client immediately as they arrive.
2.  Accumulate the full PCM waveform.
3.  After generation completes, analyse audio energy to detect the speech
    onset / offset (stripping leading/trailing silence).
4.  Distribute words within the detected speech region using character-
    weighted proportional timing (longer words → more time).

SSE protocol:
    event: audio   -- base64-encoded PCM chunk (streamed in real time)
    event: word    -- word boundary with offset_ms / duration_ms (emitted
                      after all audio, before ``done``)
    event: done    -- stream complete with total_duration_ms
    event: error   -- generation failure
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import unicodedata
from typing import Any

import numpy as np
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

logger = init_logger(__name__)

_PCM_BYTES_PER_SAMPLE = 2
_DEFAULT_CODEC_FPS = 12.5

# Energy-based speech detection parameters
_ENERGY_FRAME_MS = 20
_ENERGY_THRESHOLD_RATIO = 0.04  # fraction of peak RMS to treat as speech
_MIN_SPEECH_FRAMES = 3  # require N consecutive frames above threshold


def _is_punctuation(text: str) -> bool:
    return all(unicodedata.category(ch).startswith("P") for ch in text if ch.strip())


def _build_word_boundaries(
    text: str,
    tokenizer: AutoTokenizer,
) -> list[dict[str, Any]]:
    """Map text to words with token-index boundaries.

    Returns ``[{text, first_token, last_token, type}, ...]``.
    """
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    offsets = encoding.get("offset_mapping")

    if not token_ids:
        return []

    if offsets is None:
        return [
            {
                "text": tokenizer.decode([tid]),
                "first_token": i,
                "last_token": i,
                "type": "punc" if _is_punctuation(tokenizer.decode([tid])) else "word",
            }
            for i, tid in enumerate(token_ids)
        ]

    words: list[dict[str, Any]] = []
    cur_char_start = offsets[0][0]
    cur_char_end = offsets[0][1]
    cur_first_token = 0

    for i in range(1, len(token_ids)):
        char_start, char_end = offsets[i]
        if char_start == char_end == 0:
            continue
        gap = text[cur_char_end:char_start]
        has_gap_ws = any(c in gap for c in " \n\t")
        has_leading_ws = char_start < len(text) and text[char_start] in " \n\t"

        if has_gap_ws or has_leading_ws:
            word_text = text[cur_char_start:cur_char_end].strip()
            if word_text:
                words.append({
                    "text": word_text,
                    "first_token": cur_first_token,
                    "last_token": i - 1,
                    "type": "punc" if _is_punctuation(word_text) else "word",
                })
            new_start = char_start
            while new_start < char_end and new_start < len(text) and text[new_start] in " \n\t":
                new_start += 1
            cur_char_start = new_start if new_start < char_end else char_start
            cur_first_token = i
        cur_char_end = max(cur_char_end, char_end)

    word_text = text[cur_char_start:cur_char_end].strip()
    if word_text:
        words.append({
            "text": word_text,
            "first_token": cur_first_token,
            "last_token": len(token_ids) - 1,
            "type": "punc" if _is_punctuation(word_text) else "word",
        })
    return words


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
#  Audio energy analysis
# ---------------------------------------------------------------------------

def _compute_rms_frames(pcm: np.ndarray, sample_rate: int, frame_ms: int = _ENERGY_FRAME_MS) -> np.ndarray:
    """Return per-frame RMS energy for *pcm* (float32, mono)."""
    frame_size = int(sample_rate * frame_ms / 1000)
    if frame_size <= 0 or len(pcm) < frame_size:
        return np.array([], dtype=np.float32)
    n_frames = len(pcm) // frame_size
    trimmed = pcm[: n_frames * frame_size].reshape(n_frames, frame_size)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def _detect_speech_bounds_ms(
    pcm: np.ndarray,
    sample_rate: int,
    frame_ms: int = _ENERGY_FRAME_MS,
) -> tuple[float, float]:
    """Return (speech_start_ms, speech_end_ms) from energy thresholding.

    Uses a conservative threshold to find the first and last regions of
    sustained energy, which approximates where audible speech begins and
    ends.
    """
    rms = _compute_rms_frames(pcm, sample_rate, frame_ms)
    if rms.size == 0:
        total_ms = len(pcm) / sample_rate * 1000.0
        return 0.0, total_ms

    peak = float(rms.max())
    if peak < 1e-8:
        total_ms = len(pcm) / sample_rate * 1000.0
        return 0.0, total_ms

    threshold = peak * _ENERGY_THRESHOLD_RATIO
    above = rms > threshold

    # Find first run of _MIN_SPEECH_FRAMES consecutive active frames.
    start_idx = 0
    count = 0
    for i, v in enumerate(above):
        if v:
            count += 1
            if count >= _MIN_SPEECH_FRAMES:
                start_idx = i - _MIN_SPEECH_FRAMES + 1
                break
        else:
            count = 0

    # Find last run of _MIN_SPEECH_FRAMES consecutive active frames.
    end_idx = len(rms) - 1
    count = 0
    for i in range(len(above) - 1, -1, -1):
        if above[i]:
            count += 1
            if count >= _MIN_SPEECH_FRAMES:
                end_idx = i + _MIN_SPEECH_FRAMES - 1
                break
        else:
            count = 0

    start_ms = float(start_idx * frame_ms)
    end_ms = float((end_idx + 1) * frame_ms)
    return start_ms, end_ms


# ---------------------------------------------------------------------------
#  Word timing computation
# ---------------------------------------------------------------------------

_PUNCT_STRIP = set(".,!?;:'\"-()[]{}…")


def _word_weight(text: str) -> float:
    """Estimate relative speaking duration from word text.

    Uses cleaned character count as a simple but non-uniform proxy for
    phonetic duration.  Punctuation-only tokens get a small fixed weight
    representing a natural pause.
    """
    cleaned = "".join(ch for ch in text if ch not in _PUNCT_STRIP)
    if not cleaned:
        return 0.5  # pause weight for standalone punctuation
    return float(max(1, len(cleaned)))


def _compute_word_timings(
    word_boundaries: list[dict[str, Any]],
    speech_start_ms: float,
    speech_end_ms: float,
) -> list[dict[str, Any]]:
    """Distribute words across the detected speech region by character weight."""
    speech_dur = max(0.0, speech_end_ms - speech_start_ms)
    if not word_boundaries or speech_dur <= 0:
        return []

    weights = [_word_weight(wb["text"]) for wb in word_boundaries]
    total_weight = sum(weights) or 1.0

    events: list[dict[str, Any]] = []
    cursor_ms = speech_start_ms
    for i, wb in enumerate(word_boundaries):
        dur_ms = speech_dur * (weights[i] / total_weight)
        events.append({
            "text": wb["text"],
            "offset_ms": round(cursor_ms),
            "duration_ms": round(dur_ms),
            "type": wb["type"],
        })
        cursor_ms += dur_ms
    return events


# ---------------------------------------------------------------------------
#  SSE generator
# ---------------------------------------------------------------------------

async def _aligned_sse_generator(
    speech_service: OmniOpenAIServingSpeech,
    request: OpenAICreateSpeechRequest,
    word_boundaries: list[dict[str, Any]],
    codec_fps: float,
    sample_rate: int = 24000,
) -> Any:
    """Yield SSE strings: audio chunks streamed in real time, word events
    emitted after full audio analysis for accurate alignment."""
    request.stream = True
    request.response_format = "pcm"
    request.non_streaming_mode = False

    try:
        request_id, generator, _ = await speech_service._prepare_speech_generation(request)
    except Exception as e:
        yield _sse("error", {"message": f"Failed to prepare generation: {e}"})
        return

    prev_audio_count = 0
    all_pcm_chunks: list[np.ndarray] = []

    try:
        async for res in generator:
            audio_output, audio_key = speech_service._extract_audio_output(res)
            if audio_key is None:
                continue

            sr_raw = audio_output.get("sr")
            if sr_raw is not None:
                sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
                sample_rate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

            audio_val = audio_output[audio_key]
            if isinstance(audio_val, list):
                new_chunks = audio_val[prev_audio_count:]
                prev_audio_count = len(audio_val)
            else:
                new_chunks = [audio_val] if audio_val is not None else []
                prev_audio_count += len(new_chunks)

            for chunk_tensor in new_chunks:
                if chunk_tensor is None:
                    continue
                if hasattr(chunk_tensor, "numel") and chunk_tensor.numel() == 0:
                    continue

                wav_np = chunk_tensor.detach().cpu().float().numpy()
                pcm_int16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
                pcm_bytes = pcm_int16.tobytes()

                all_pcm_chunks.append(wav_np)

                b64 = base64.b64encode(pcm_bytes).decode("ascii")
                yield _sse("audio", {"chunk": b64})

    except asyncio.CancelledError:
        logger.info("Aligned speech stream cancelled by client")
        yield _sse("error", {"message": "Client disconnected"})
        return
    except Exception as e:
        logger.exception("Aligned speech generation failed: %s", e)
        yield _sse("error", {"message": f"Generation failed: {e}"})
        return

    # -- Post-generation: analyse audio and compute word timings --
    if all_pcm_chunks:
        full_pcm = np.concatenate(all_pcm_chunks)
    else:
        full_pcm = np.array([], dtype=np.float32)

    total_samples = len(full_pcm)
    total_ms = round(total_samples / sample_rate * 1000.0) if sample_rate > 0 else 0

    if word_boundaries and total_samples > 0:
        speech_start, speech_end = _detect_speech_bounds_ms(full_pcm, sample_rate)
        word_events = _compute_word_timings(word_boundaries, speech_start, speech_end)
        for evt in word_events:
            yield _sse("word", evt)
    else:
        speech_start = 0.0
        speech_end = float(total_ms)

    yield _sse("done", {
        "total_duration_ms": total_ms,
        "speech_start_ms": round(speech_start),
        "speech_end_ms": round(speech_end),
    })


def _ensure_tokenizer(speech_service: OmniOpenAIServingSpeech) -> AutoTokenizer:
    if speech_service._tts_tokenizer is None:
        model_name = speech_service.engine_client.model_config.model
        speech_service._tts_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left",
        )
    return speech_service._tts_tokenizer


async def create_aligned_speech(
    speech_service: OmniOpenAIServingSpeech,
    request: OpenAICreateSpeechRequest,
) -> StreamingResponse:
    """Entry point for the /v1/audio/speech/aligned endpoint."""
    tokenizer = _ensure_tokenizer(speech_service)
    codec_fps = speech_service._codec_frame_rate or _DEFAULT_CODEC_FPS

    word_boundaries = _build_word_boundaries(
        text=request.input,
        tokenizer=tokenizer,
    )

    return StreamingResponse(
        _aligned_sse_generator(
            speech_service, request, word_boundaries,
            codec_fps=codec_fps,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
