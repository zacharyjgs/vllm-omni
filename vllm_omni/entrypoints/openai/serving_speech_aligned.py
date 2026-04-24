"""Streaming SSE endpoint that returns TTS audio interleaved with word-level
timing events derived from **actual** text token consumption during generation.

The Qwen3-TTS Talker consumes exactly one text conditioning vector per decode
step.  The instrumented Talker emits a ``text_token_cursor`` alongside each
codec frame, and the pipeline threads it through to the serving layer.  This
module uses those cursor values to determine the exact audio timestamp at which
each word boundary is crossed, then emits SSE ``word`` events interleaved with
base64-encoded PCM ``audio`` events.

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

_PCM_BYTES_PER_SAMPLE = 2


def _is_punctuation(text: str) -> bool:
    return all(unicodedata.category(ch).startswith("P") for ch in text if ch.strip())


def _build_word_boundaries(
    text: str,
    tokenizer: AutoTokenizer,
) -> list[dict[str, Any]]:
    """Map text to words with token-index boundaries (no time yet).

    Returns a list of ``{text, first_token, last_token, type}`` dicts.
    Token indices are 0-based and correspond to the tokenizer output
    *without* special tokens (matching the Talker's ``input_ids[:, 3:-5]``
    for the first token and ``4:-5`` for the rest in CustomVoice streaming
    mode, plus the final EOS).
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


async def _aligned_sse_generator(
    speech_service: OmniOpenAIServingSpeech,
    request: OpenAICreateSpeechRequest,
    word_boundaries: list[dict[str, Any]],
    sample_rate: int = 24000,
) -> Any:
    """Async generator yielding SSE strings with interleaved word + audio events.

    Word events are emitted when the ``text_token_cursor`` metadata from the
    engine crosses a word boundary, giving exact audio-time alignment.
    """
    request.stream = True
    request.response_format = "pcm"

    try:
        request_id, generator, _ = await speech_service._prepare_speech_generation(request)
    except Exception as e:
        yield _sse("error", {"message": f"Failed to prepare generation: {e}"})
        return

    word_cursor = 0
    cumulative_samples = 0
    # Track the last seen text_token_cursor to detect transitions.
    last_seen_ttc = -1
    # Map from token index to the audio sample offset where it was first seen.
    token_start_samples: dict[int, int] = {}

    try:
        async for pcm_bytes, ttc_list in speech_service._generate_pcm_chunks_with_cursors(
            generator, request_id
        ):
            chunk_samples = len(pcm_bytes) // _PCM_BYTES_PER_SAMPLE

            # Record token cursor transitions within this chunk.
            if ttc_list:
                for ttc_val in ttc_list:
                    if ttc_val >= 0 and ttc_val not in token_start_samples:
                        token_start_samples[ttc_val] = cumulative_samples

            # Check if any word boundaries have been crossed.
            max_cursor = max(ttc_list) if ttc_list else last_seen_ttc
            while word_cursor < len(word_boundaries):
                wb = word_boundaries[word_cursor]
                # Emit word event when the first token of this word has been
                # seen (its audio start time is known).
                ft = wb["first_token"]
                # The Talker cursor starts at 1 (first text token consumed in
                # prefill). Token index 0 in the word boundaries corresponds
                # to cursor value 1. So map: cursor = token_index + 1.
                cursor_for_ft = ft + 1
                if cursor_for_ft in token_start_samples:
                    start_samples = token_start_samples[cursor_for_ft]
                    offset_ms = round(start_samples / sample_rate * 1000.0)
                    # Duration is estimated to the next word or chunk end.
                    next_ft = (
                        word_boundaries[word_cursor + 1]["first_token"] + 1
                        if word_cursor + 1 < len(word_boundaries)
                        else None
                    )
                    if next_ft is not None and next_ft in token_start_samples:
                        dur_ms = round(
                            (token_start_samples[next_ft] - start_samples) / sample_rate * 1000.0
                        )
                    else:
                        dur_ms = round(
                            (cumulative_samples + chunk_samples - start_samples) / sample_rate * 1000.0
                        )
                    yield _sse("word", {
                        "text": wb["text"],
                        "offset_ms": offset_ms,
                        "duration_ms": max(dur_ms, 0),
                        "type": wb["type"],
                    })
                    word_cursor += 1
                else:
                    break

            if ttc_list:
                last_seen_ttc = max(max_cursor, last_seen_ttc)

            b64 = base64.b64encode(pcm_bytes).decode("ascii")
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

    # Flush remaining word events with best-effort timing.
    total_ms = round(cumulative_samples / sample_rate * 1000.0)
    while word_cursor < len(word_boundaries):
        wb = word_boundaries[word_cursor]
        ft = wb["first_token"] + 1
        start_samples = token_start_samples.get(ft, cumulative_samples)
        offset_ms = round(start_samples / sample_rate * 1000.0)
        yield _sse("word", {
            "text": wb["text"],
            "offset_ms": offset_ms,
            "duration_ms": max(total_ms - offset_ms, 0),
            "type": wb["type"],
        })
        word_cursor += 1

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

    word_boundaries = _build_word_boundaries(
        text=request.input,
        tokenizer=tokenizer,
    )

    return StreamingResponse(
        _aligned_sse_generator(speech_service, request, word_boundaries),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
