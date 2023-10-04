#!/bin/env python3
import argparse
import json
import os
from typing import AsyncGenerator

from fastapi import FastAPI, APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.engine.arg_utils import EngineArgs, AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

import kai
from drivers import InferenceEngine, DummyInferenceEngine, VLLMInferenceEngine, \
    TextGenerationInferenceEngine
from util import BridgeException

app = FastAPI(title="tgi-kai-bridge")
api = APIRouter()

INFERENCE_ENGINE_STR = os.environ.get("KAI_BRIDGE_ENGINE", "tgi")
TGI_ENDPOINT = os.environ.get("TGI_ENDPOINT", "http://127.0.0.1:3000")
TGI_MODE = os.environ.get("TGI_MODE", "")
TGI_MODEL = os.environ.get("TGI_MODEL", "")

INFERENCE_ENGINE: InferenceEngine = DummyInferenceEngine()


@api.get("/info/version")
def get_version() -> kai.BasicResultInner:
    """ Impersonate KAI """
    return kai.BasicResultInner(result="1.2.4")


@api.get("/model")
def get_model() -> kai.BasicResultInner:
    """ Get current model """
    return kai.BasicResultInner(result=INFERENCE_ENGINE.get_model())


@api.get("/config/soft_prompts_list")
def get_available_softprompts() -> kai.SoftPromptsList:
    """ stub for AI-Horde-Worker compatibility """
    return kai.SoftPromptsList(values=[])


@api.get("/config/soft_prompt")
def get_current_softprompt() -> kai.SoftPromptSetting:
    """ stub for AI-Horde-Worker compatibility """
    return kai.SoftPromptSetting(value="")


@api.put("/config/soft_prompt")
def set_current_softprompt():
    """ stub for AI-Horde-Worker compatibility """
    return kai.Empty()


@api.post("/generate")
async def generate(kai_payload: kai.GenerationInput) -> kai.GenerationOutput:
    """ Generate text """
    return await INFERENCE_ENGINE.generate(kai_payload)


@app.post("/api/extra/generate/stream")
async def generate_stream(request: Request) -> StreamingResponse:
    """ KoboldCpp streaming """
    json_payload = await request.json()
    kai_payload = kai.GenerationInput.model_construct(**json_payload)

    result = INFERENCE_ENGINE.generate_stream(kai_payload)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for token in result:
            yield b"event: message\n"
            yield f"data: {json.dumps({'token': token})}\n\n".encode()

    return StreamingResponse(stream_results(),
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                             media_type='text/event-stream')


@app.post("/api/extra/abort")
def abort_generation():
    """ stub for compatibility """
    return kai.Empty()


app.include_router(api, prefix="/api/v1")
app.include_router(api, prefix="/api/latest", include_in_schema=False)


@app.exception_handler(BridgeException)
def exception_handler(_, exc: BridgeException):
    return JSONResponse(status_code=400, content=jsonable_encoder(exc.model))


@app.get("/api/extra/version")
def get_extra_version():
    """ Impersonate KoboldCpp with streaming support """
    return {"result": "KoboldCpp", "version": "1.30"}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="tgi")
    parser.add_argument("--endpoint", type=str, default="http://localhost:3000")
    parser.add_argument("--mode", type=str, default=None, help="information to add to"
                                                             " the model string that describes"
                                                             " the model in use, such as whether"
                                                             " it is quantized to a lower"
                                                             " precision")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # the host/port that this server will bind to, not the tgi api server. that's
    # --endpoint in the parser.
    host = os.environ.get("KAI_HOST", "127.0.0.1")
    port = int(os.environ.get("KAI_PORT", 5000))

    if args.type == "vllm":
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        INFERENCE_ENGINE = VLLMInferenceEngine(engine, mode=args.mode, model=args.model)
    else:
        INFERENCE_ENGINE = TextGenerationInferenceEngine(args.endpoint, TGI_MODEL, TGI_MODE)

    if os.environ.get("DEBUG"):
        uvicorn.run("main:app", reload=True, host=host, port=port, log_level="debug")
    else:
        uvicorn.run(app, host=host, port=port)
