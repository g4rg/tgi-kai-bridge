#!/bin/env python3

from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

import requests

import kai
import tgi

api = APIRouter()

TGI_ENDPOINT = "http://127.0.0.1:3000"

@api.post("/generate")
def generate(input: kai.GenerationInput) -> kai.GenerationOutput:
    """ Generate text """
    tgi_parameters = tgi.GenerateParameters.model_construct(do_sample=True, truncate=input.max_context_length-input.max_length)
    if input.max_length:
        tgi_parameters.max_new_tokens = input.max_length
    if input.temperature:
        tgi_parameters.temperature = max(input.temperature, 0.001)
    if input.top_p:
        tgi_parameters.top_p = min(max(input.top_p, 0.001), 0.999)
    if input.top_k:
        tgi_parameters.top_k = max(input.top_k, 0.001)
    if input.rep_pen:
        tgi_parameters.repetition_penalty = max(input.rep_pen, 0.001)
    if input.sampler_seed:
        tgi_parameters.seed = input.sampler_seed
    tgi_request = tgi.GenerateRequest(inputs=input.prompt, parameters=tgi_parameters)

    r = requests.post(TGI_ENDPOINT + "/generate", json=tgi_request.model_dump(exclude_none=True), headers={"Content-Type": "application/json"})
    if r.status_code != 200:
        raise BridgeException(kai.BasicError(msg=r.text, type="Error"))
    tgi_result = tgi.GenerateResponse(**r.json())
    return kai.GenerationOutput(results=[kai.GenerationResult(text=tgi_result.generated_text)])

@api.get("/info/version")
def get_version() -> kai.BasicResult:
    """ Get API version """
    return kai.BasicResult(result=kai.BasicResultInner(result="1.2.3"))

@api.get("/model")
def get_model() -> kai.BasicResultInner:
    """ Get current model """
    tgi_info = tgi.Info(**requests.get(TGI_ENDPOINT + "/info").json())
    return kai.BasicResultInner(result=f"{tgi_info.model_id}")

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

app = FastAPI(title="tgi-kai-bridge")
app.include_router(api, prefix="/api/v1")
app.include_router(api, prefix="/api/latest", include_in_schema=False)

@app.exception_handler(BridgeException)
def exception_handler(_, exc: BridgeException):
    return JSONResponse(status_code=400, content=jsonable_encoder(exc.model))

@app.get("/api/extra/version")
def get_extra_version():
    return {"result": "tgi-kai-bridge", "version": "0.0.1"}

if __name__ == "__main__":
    import uvicorn, os

    host = os.environ.get("KAI_HOST", "0.0.0.0")
    port = int(os.environ.get("KAI_PORT", 5000))

    tgi_ep = os.environ.get("TGI_ENDPOINT")
    if tgi_ep:
        TGI_ENDPOINT = tgi_ep

    if os.environ.get("DEBUG"):
        uvicorn.run("main:app", reload=True, host=host, port=port)
    else:
        uvicorn.run(app, host=host, port=port, log_level="warning")
