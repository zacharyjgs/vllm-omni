"""Streaming SSE endpoint that returns TTS audio interleaved with word-level
timing events.

Uses precomputed word alignment from the tokenizer and codec frame rate.
Word timings are estimated based on token positions and scaled to match
the actual audio duration once generation completes.

SSE protocol (mirrors tts-generation-backend Azure pattern):
    event: word    -- word/punc boundary with offset_ms, duration_ms
    event: audio   -- base64-encoded PCM chunk
    event: done    -- stream complete
    event: error   -- generation failure
"""

from __future__ import annotations

import asyncio
import base64
import json
import unicodedata
from typing import Any

from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

logger = init_logger(__name__)

_DEFAULT_CODEC_FPS = 12.5
_PCM_BYTES_PER_SAMPLE = 2


def _is_punctuation(text: str) -> bool:
    return all(unicodedata.category(ch).startswith("P") for ch in text if ch.strip())


def precompute_word_alignments(
    text: str,
    tokenizer: AutoTokenizer,
    codec_fps: float,
) -> list[dict[str, Any]]:
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoding["input_ids"]
    offsets = encoding.get("offset_mapping")

    if not token_ids:
        return []

    ms_per_step = 1000.0 / codec_fps if codec_fps > 0 else 80.0

    if offsets is None:
        alignments = []
        for i, tid in enumerate(token_ids):
            decoded = tokenizer.decode([tid])
            alignments.append({
                "text": decoded,
                "offset_ms": round(i * ms_per_step),
                "duration_ms": round(ms_per_step),
                "type": "punc" if _is_punctuation(decoded) else "word",
            })
        return alignments

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
                start_ms = round(cur_first_token * ms_per_step)
                dur_ms = round((i - cur_first_token) * ms_per_step)
                words.append({
                    "text": word_text,
                    "offset_ms": start_ms,
                    "duration_ms": dur_ms,
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
        start_ms = round(cur_first_token * ms_per_step)
        dur_ms = round((len(token_ids) - cur_first_token) * ms_per_step)
        words.append({
            "text": word_text,
            "offset_ms": start_ms,
            "duration_ms": dur_ms,
            "type": "punc" if _is_punctuation(word_text) else "word",
        })
    return words


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _aligned_sse_generator(
    speech_service: OmniOpenAIServingSpeech,
    request: OpenAICreateSpeechRequest,
    word_alignments: list[dict[str, Any]],
    sample_rate: int = 24000,
) -> Any:
    request.stream = True
    request.response_format = "pcm"

    try:
        request_id, generator, _ = await speech_service._prepare_speech_generation(request)
    except Exception as e:
        yield _sse("error", {"message": f"Failed to prepare generation: {e}"})
        return

    word_cursor = 0
    cumulative_samples = 0

    try:
        async for pcm_chunk in speech_service._generate_pcm_chunks(generator, request_id):
            chunk_samples = len(pcm_chunk) // _PCM_BYTES_PER_SAMPLE
            chunk_end_ms = (cumulative_samples + chunk_samples) / sample_rate * 1000.0

            while word_cursor < len(word_alignments):
                wa = word_alignments[word_cursor]
                if wa["offset_ms"] < chunk_end_ms:
                    yield _sse("word", wa)
                    word_cursor += 1
                else:
                    break

            b64 = base64.b64encode(pcm_chunk).decode("ascii")
            yield _sse("audio", {"chunk": b64})
            cumulative_samples += chunk_samples

    except asyncio.CancelledError:
        logger.info("Aligned speech stream cancelled by client")
        yield _sse("error", {"message": "Client disconnected"})
        return
    except Exception as e:
        logger.exception("Aligned speech generation failed: %s", e)
        yield _sse("error", {"message": f"Generation failed: {e}"})
        return

    while word_cursor < len(word_alignments):
        yield _sse("word", word_alignments[word_cursor])
        word_cursor += 1

    total_ms = round(cumulative_samples / sample_rate * 1000.0)
    yield _sse("done", {"total_duration_ms": total_ms})


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

    word_alignments = precompute_word_alignments(
        text=request.input,
        tokenizer=tokenizer,
        codec_fps=codec_fps,
    )

    return StreamingResponse(
        _aligned_sse_generator(speech_service, request, word_alignments),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
