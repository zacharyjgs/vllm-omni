"""Streaming SSE endpoint that returns TTS audio interleaved with word-level
timing events derived from codec frame positions during generation.

The Qwen3-TTS Talker consumes exactly one text conditioning vector per decode
step, producing one codec frame per step.  In streaming mode
(non_streaming_mode=False), the first text token is consumed in prefill and
subsequent tokens are consumed one-per-step.  The Talker tracks a
``text_token_cursor`` (total tokens consumed so far) which propagates through
the stage processor and Code2Wav as ``chunk_cursor_start`` /
``chunk_cursor_end`` scalar tensors.

The aligned endpoint forces streaming mode and maps cursor values to tokenizer
offset-mapping word boundaries for precise word-level timing.

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

import numpy as np
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

logger = init_logger(__name__)

_PCM_BYTES_PER_SAMPLE = 2
_DEFAULT_CODEC_FPS = 12.5


def _is_punctuation(text: str) -> bool:
    return all(unicodedata.category(ch).startswith("P") for ch in text if ch.strip())


def _build_word_boundaries(
    text: str,
    tokenizer: AutoTokenizer,
) -> list[dict[str, Any]]:
    """Map text to words with token-index boundaries.

    Returns ``[{text, first_token, last_token, type}, ...]``.
    Token indices are 0-based corresponding to the tokenizer output without
    special tokens.
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


def _extract_scalar(audio_output: dict, key: str, index: int) -> int | None:
    """Extract an int scalar from a cumulative list of tensors at *index*."""
    val = audio_output.get(key)
    if not isinstance(val, list) or index >= len(val):
        return None
    v = val[index]
    return int(v.item()) if hasattr(v, "item") else int(v)


async def _aligned_sse_generator(
    speech_service: OmniOpenAIServingSpeech,
    request: OpenAICreateSpeechRequest,
    word_boundaries: list[dict[str, Any]],
    codec_fps: float,
    sample_rate: int = 24000,
) -> Any:
    """Async generator yielding SSE strings with interleaved word + audio events.

    Iterates over the raw engine generator to extract both audio tensors and
    cursor metadata from Code2Wav's multimodal_outputs.
    """
    request.stream = True
    request.response_format = "pcm"
    request.non_streaming_mode = False

    try:
        request_id, generator, _ = await speech_service._prepare_speech_generation(request)
    except Exception as e:
        yield _sse("error", {"message": f"Failed to prepare generation: {e}"})
        return

    word_cursor = 0
    cumulative_samples = 0
    cumulative_frames = 0
    prev_audio_count = 0
    samples_per_frame = sample_rate / codec_fps if codec_fps > 0 else sample_rate / 12.5
    has_cursor_metadata = False

    try:
        async for res in generator:
            audio_output, audio_key = speech_service._extract_audio_output(res)
            if audio_key is None:
                continue

            sr_raw = audio_output.get("sr")
            if sr_raw is not None:
                sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
                sample_rate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
                samples_per_frame = sample_rate / codec_fps

            audio_val = audio_output[audio_key]
            if isinstance(audio_val, list):
                new_chunks = audio_val[prev_audio_count:]
                new_start_idx = prev_audio_count
                prev_audio_count = len(audio_val)
            else:
                new_chunks = [audio_val] if audio_val is not None else []
                new_start_idx = max(0, prev_audio_count - 1)
                prev_audio_count += len(new_chunks)

            for ci, chunk_tensor in enumerate(new_chunks):
                if chunk_tensor is None:
                    continue
                if hasattr(chunk_tensor, "numel") and chunk_tensor.numel() == 0:
                    continue

                chunk_idx = new_start_idx + ci

                fs = _extract_scalar(audio_output, "chunk_frame_start", chunk_idx)
                fe = _extract_scalar(audio_output, "chunk_frame_end", chunk_idx)
                cs = _extract_scalar(audio_output, "chunk_cursor_start", chunk_idx)
                ce = _extract_scalar(audio_output, "chunk_cursor_end", chunk_idx)

                wav_np = chunk_tensor.detach().cpu().float().numpy()
                pcm_int16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
                pcm_bytes = pcm_int16.tobytes()
                chunk_samples = len(pcm_bytes) // _PCM_BYTES_PER_SAMPLE

                if fs is not None and fe is not None and fe > 0:
                    has_cursor_metadata = True
                    chunk_frame_start = fs
                    chunk_frame_end = fe
                    cursor_end = ce if ce is not None else 0
                else:
                    chunk_frames = max(1, round(chunk_samples / samples_per_frame))
                    chunk_frame_start = cumulative_frames
                    chunk_frame_end = cumulative_frames + chunk_frames
                    cursor_end = 0

                while word_cursor < len(word_boundaries):
                    wb = word_boundaries[word_cursor]
                    ft = wb["first_token"]

                    if has_cursor_metadata and cursor_end > 0:
                        # cursor_end = total text tokens consumed after this chunk.
                        # A word starting at token ft is spoken once cursor > ft.
                        if cursor_end <= ft:
                            break
                    else:
                        # Fallback: frame K corresponds to text token K+1
                        # (token 0 consumed in prefill, before decode frame 0).
                        speak_frame = max(0, ft - 1)
                        if speak_frame >= chunk_frame_end:
                            break

                    if has_cursor_metadata and cursor_end > 0:
                        offset_frames = max(0, ft - 1)
                    else:
                        offset_frames = chunk_frame_start + max(0, ft - chunk_frame_start)

                    offset_ms = round(offset_frames * samples_per_frame / sample_rate * 1000.0)

                    next_ft = (
                        word_boundaries[word_cursor + 1]["first_token"]
                        if word_cursor + 1 < len(word_boundaries)
                        else chunk_frame_end
                    )
                    dur_samples = (next_ft - ft) * samples_per_frame
                    dur_ms = round(dur_samples / sample_rate * 1000.0)

                    yield _sse("word", {
                        "text": wb["text"],
                        "offset_ms": offset_ms,
                        "duration_ms": max(dur_ms, 0),
                        "type": wb["type"],
                    })
                    word_cursor += 1

                cumulative_frames = chunk_frame_end

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

    # Flush remaining word events.
    total_ms = round(cumulative_samples / sample_rate * 1000.0)
    while word_cursor < len(word_boundaries):
        wb = word_boundaries[word_cursor]
        ft = wb["first_token"]
        offset_ms = round(max(0, ft - 1) * samples_per_frame / sample_rate * 1000.0)
        yield _sse("word", {
            "text": wb["text"],
            "offset_ms": min(offset_ms, total_ms),
            "duration_ms": max(total_ms - offset_ms, 0),
            "type": wb["type"],
        })
        word_cursor += 1

    yield _sse("done", {
        "total_duration_ms": total_ms,
        "cursor_metadata_available": has_cursor_metadata,
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
