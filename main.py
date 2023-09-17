#!/bin/env python3

import os
import json

from typing import Iterator

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder

import requests

import kai
import tgi

app = FastAPI(title="tgi-kai-bridge")
api = APIRouter()

TGI_ENDPOINT = os.environ.get("TGI_ENDPOINT", "http://127.0.0.1:3000")
TGI_MODE = os.environ.get("TGI_MODE", "")
TGI_MODEL = os.environ.get("TGI_MODEL", "")

def translate_payload(kai_payload: kai.GenerationInput) -> tgi.GenerateRequest:
    """ Translate KoboldAI GenerationInput to TGI GenerateRequest """

    tgi_parameters = tgi.GenerateParameters.model_construct(do_sample=True, \
                            truncate=max(1, kai_payload.max_context_length - kai_payload.max_length), \
                            max_new_tokens=kai_payload.max_length)

    if kai_payload.temperature:
        tgi_parameters.temperature = max(kai_payload.temperature, 0.001)
    if kai_payload.top_p:
        tgi_parameters.top_p = min(max(kai_payload.top_p, 0.001), 0.999)
    if kai_payload.top_k:
        tgi_parameters.top_k = max(kai_payload.top_k, 0.001)
    if kai_payload.rep_pen:
        tgi_parameters.repetition_penalty = max(kai_payload.rep_pen, 0.001)
    if kai_payload.sampler_seed:
        tgi_parameters.seed = kai_payload.sampler_seed

    return tgi.GenerateRequest(inputs=kai_payload.prompt, parameters=tgi_parameters)

@api.post("/generate")
def generate(kai_payload: kai.GenerationInput) -> kai.GenerationOutput:
    """ Generate text """

    tgi_payload = translate_payload(kai_payload)
    r = requests.post(TGI_ENDPOINT + "/generate", json=tgi_payload.model_dump(exclude_none=True), headers={"Content-Type": "application/json"})

    if r.status_code != 200:
        raise BridgeException(kai.BasicError(msg=r.text, type="Error"))

    result = tgi.GenerateResponse(**r.json()).generated_text
    return kai.GenerationOutput(results=[kai.GenerationResult(text=result)])

@api.get("/info/version")
def get_version() -> kai.BasicResultInner:
    """ Impersonate KAI """
    return kai.BasicResultInner(result="1.2.4")

@api.get("/model")
def get_model() -> kai.BasicResultInner:
    """ Get current model """
    model = TGI_MODEL or tgi.Info(**requests.get(TGI_ENDPOINT + "/info").json()).model_id
    model_name = "tgi" \
        + (f"-{TGI_MODE}" if TGI_MODE else "") \
        + "/" + model
    return kai.BasicResultInner(result=model_name)

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

class BridgeException(Exception):
    def __init__(self, model: kai.BasicError):
        self.model = model

def stream_from_tgi(iter_sse_lines: Iterator[bytes]) -> Iterator[str]:
    """ Produce tokens streamed from TGI SSE byte stream """

    generated_text = ""
    for line in iter_sse_lines:
        data = str(line, "utf-8")

        if not data.startswith("data:"):
            continue

        json_data = json.loads(data.lstrip("data:"))
        if json_data["generated_text"] is not None and len(json_data["generated_text"]) > 0:
            generated_text = json_data["generated_text"]

        if json_data["token"]["special"]:
            continue

        yield json_data["token"]["text"]

    return generated_text

def stream_kobold(iter_tokens: Iterator[str]) -> Iterator[bytes]:
    """ Produce Kobold SSE byte stream from strings """

    generated_text = ""
    for token in iter_tokens:
        yield b"event: message\n"
        yield f"data: {json.dumps({'token': token})}\n\n".encode()
        generated_text += token

    return generated_text

@app.post("/api/extra/generate/stream")
def generate_stream(kai_payload: kai.GenerationInput) -> StreamingResponse:
    """ KoboldCpp streaming """

    r = requests.post(TGI_ENDPOINT + "/generate_stream", \
                      json=translate_payload(kai_payload).model_dump(exclude_none=True), \
                      headers={"Content-Type": "application/json"}, \
                      stream=True)

    if r.status_code != 200:
        raise BridgeException(kai.BasicError(msg=r.text, type="Error"))

    return StreamingResponse(stream_kobold(stream_from_tgi(r.iter_lines())), \
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}, \
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

    host = os.environ.get("KAI_HOST", "127.0.0.1")
    port = int(os.environ.get("KAI_PORT", 5000))

    if os.environ.get("DEBUG"):
        uvicorn.run("main:app", reload=True, host=host, port=port, log_level="debug")
    else:
        uvicorn.run(app, host=host, port=port)
