"""
==============================================================================
Colab GPU Inference Server — FastAPI + Qwen2.5 with 4-bit quantization
==============================================================================

THIS FILE RUNS ON GOOGLE COLAB (not locally).

It provides a FastAPI HTTP server that:
    1. Loads Qwen2.5-1.5B-Instruct with 4-bit quantization (~1.2 GB VRAM)
    2. Exposes /generate endpoint for text generation
    3. Supports both streaming (SSE) and non-streaming responses
    4. Uses torch.inference_mode() for maximum speed
    5. Includes /health endpoint for connection checking

HOW TO USE:
    1. Open the colab_gpu_server.ipynb notebook in Google Colab
    2. Run all cells — it installs dependencies, loads the model, and
       starts the server with an ngrok tunnel
    3. Copy the ngrok URL into the Paper Tutor UI
    4. Done — all LLM inference now runs on Colab GPU

VRAM USAGE:
    - Qwen2.5-1.5B 4-bit: ~1.2 GB VRAM
    - Colab free T4 GPU: 15 GB VRAM
    - Headroom: ~13.8 GB (plenty for inference)
==============================================================================
"""

import os
import sys
import time
import json
import logging
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_MODEL_LEN = 2048  # Context window

# 4-bit quantization config for BitsAndBytes
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,    # Double quantization for extra savings
    bnb_4bit_quant_type="nf4",         # NormalFloat4 — best for LLMs
)


# ===========================================================================
# Global model state
# ===========================================================================

model = None
tokenizer = None
device = None


def load_model():
    """
    Load the quantized model and tokenizer.

    WHY 4-bit quantization:
        - Qwen2.5-1.5B at FP16 needs ~3 GB VRAM
        - At 4-bit, it needs only ~1.2 GB VRAM
        - Quality loss is minimal for instruction following
        - Fits comfortably on free Colab T4 (15 GB)
    """
    global model, tokenizer, device

    logger.info(f"Loading model: {MODEL_ID}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = "cpu"
        logger.warning("No GPU found — running on CPU (will be slow)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }

    if device == "cuda":
        load_kwargs["quantization_config"] = QUANTIZATION_CONFIG
        load_kwargs["device_map"] = "auto"
    else:
        # CPU fallback — no quantization
        load_kwargs["torch_dtype"] = torch.float32

    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    load_time = time.perf_counter() - start

    model.eval()

    # Log memory usage
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Model loaded in {load_time:.1f}s — VRAM: {allocated:.2f} GB")
    else:
        logger.info(f"Model loaded in {load_time:.1f}s (CPU mode)")


# ===========================================================================
# FastAPI Application
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield


app = FastAPI(
    title="Paper Tutor GPU Inference Server",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS from any origin (ngrok tunnels use different domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Request / Response Models
# ===========================================================================


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""
    prompt: str = Field(..., description="The full prompt text to generate from")
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    stream: bool = Field(default=False, description="Stream tokens via SSE")


class GenerateResponse(BaseModel):
    """Response body for non-streaming generation."""
    response: str
    tokens_generated: int
    generation_time_ms: float
    model: str


# ===========================================================================
# Endpoints
# ===========================================================================


@app.get("/health")
async def health_check():
    """
    Health check endpoint — used by the local client to verify connectivity.

    Returns model name, device, and memory usage.
    """
    info = {
        "status": "healthy",
        "model": MODEL_ID,
        "device": device,
        "quantization": "4-bit NF4" if device == "cuda" else "none (CPU)",
    }

    if device == "cuda":
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1e9, 2
        )

    return info


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate text from a prompt.

    Supports two modes:
        - stream=false: Returns complete response as JSON
        - stream=true: Returns Server-Sent Events (SSE) stream
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if request.stream:
        return StreamingResponse(
            _stream_generate(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        return _batch_generate(request)


# ===========================================================================
# Generation Logic
# ===========================================================================


def _batch_generate(request: GenerateRequest) -> GenerateResponse:
    """
    Non-streaming generation — returns the complete response at once.
    """
    start = time.perf_counter()

    # Tokenize the prompt
    inputs = tokenizer(
        request.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_LEN - request.max_tokens,
    )

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    # Generate
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=max(request.temperature, 0.01),  # Avoid div-by-zero
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the NEW tokens (exclude the prompt)
    new_tokens = outputs[0][input_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    gen_time = (time.perf_counter() - start) * 1000

    logger.info(
        f"Generated {len(new_tokens)} tokens in {gen_time:.0f}ms "
        f"({len(new_tokens) / (gen_time / 1000):.0f} tok/s)"
    )

    return GenerateResponse(
        response=response_text.strip(),
        tokens_generated=len(new_tokens),
        generation_time_ms=round(gen_time, 1),
        model=MODEL_ID,
    )


async def _stream_generate(request: GenerateRequest):
    """
    Streaming generation — yields tokens as SSE events.

    Uses HuggingFace's TextIteratorStreamer which runs generation
    in a background thread and yields tokens via a queue.
    """
    # Tokenize
    inputs = tokenizer(
        request.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_LEN - request.max_tokens,
    )

    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Create streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Generation kwargs
    gen_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "temperature": max(request.temperature, 0.01),
        "top_p": request.top_p,
        "repetition_penalty": request.repetition_penalty,
        "do_sample": request.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    # Run generation in a background thread (TextIteratorStreamer requires this)
    thread = Thread(target=_run_generation, args=(gen_kwargs,))
    thread.start()

    # Yield tokens as SSE events
    for token in streamer:
        if token:
            yield f"data: {json.dumps({'token': token})}\n\n"

    # Signal end of stream
    yield "data: [DONE]\n\n"

    thread.join()


def _run_generation(gen_kwargs: dict):
    """Run model.generate() in a thread (for streaming)."""
    with torch.inference_mode():
        model.generate(**gen_kwargs)


# ===========================================================================
# Entry point (for local testing — in Colab, uvicorn is started manually)
# ===========================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
