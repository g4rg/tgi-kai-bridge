import json
from abc import ABC, abstractmethod
from typing import Iterator, AsyncGenerator

import aiohttp
import requests
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

import kai
import tgi
import util


class InferenceEngine(ABC):

    async def generate(self, payload: kai.GenerationInput) -> kai.GenerationOutput:
        result = "".join([item async for item in self.generate_stream(payload)])

        return kai.GenerationOutput(results=[kai.GenerationResult(text=result)])

    @abstractmethod
    async def generate_stream(self, payload: kai.GenerationInput) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    def get_model(self) -> str:
        pass


class TextGenerationInferenceEngine(InferenceEngine):

    def __init__(self, endpoint: str, mode=None, model=None):
        self.endpoint = endpoint
        self.model = model if model is not None and len(model) > 0 else None
        self.mode = mode if mode is not None and len(mode) > 0 else None

    async def generate_stream(self, payload: kai.GenerationInput) -> AsyncGenerator[str, None]:
        tgi_payload = TextGenerationInferenceEngine.translate_payload(payload)
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint + '/generate_stream',
                                    json=tgi_payload.model_dump(exclude_none=True),
                                    headers={"Content-Type": "application/json"}) as response:
                if response.status == 200:
                    async for line in response.content.iter_any():
                        # Parse each line as JSON
                        try:
                            data = json.loads(line.decode('utf-8').strip("data:"))
                            token = data.get('token', {})
                            text = token.get('text')
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            pass
                else:
                    raise util.BridgeException(kai.BasicError(msg="couldn't parse streaming response", type="Error"))

    def get_model(self) -> str:
        model = self.model or tgi.Info(**requests.get(self.endpoint + "/info").json()).model_id
        model_name = "tgi" \
                     + (f"-{self.mode}" if self.mode else "") \
                     + "/" + model

        return model_name

    @staticmethod
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

    @staticmethod
    def translate_payload(kai_payload: kai.GenerationInput) -> tgi.GenerateRequest:
        """ Translate KoboldAI GenerationInput to TGI GenerateRequest """

        tgi_parameters = tgi.GenerateParameters.model_construct(do_sample=True,
                                                                truncate=max(1,
                                                                             kai_payload.max_context_length - kai_payload.max_length),
                                                                max_new_tokens=kai_payload.max_length)

        if kai_payload.temperature is not None:
            tgi_parameters.temperature = max(kai_payload.temperature, 0.001)
        if kai_payload.top_p is not None and kai_payload.top_p != 1:
            tgi_parameters.top_p = min(max(kai_payload.top_p, 0.001), 0.999)
        if kai_payload.top_k is not None and kai_payload.top_k != 0:
            tgi_parameters.top_k = max(kai_payload.top_k, 1)
        if kai_payload.rep_pen is not None and kai_payload.rep_pen != 1:
            tgi_parameters.repetition_penalty = max(kai_payload.rep_pen, 0.001)
        if kai_payload.sampler_seed is not None:
            tgi_parameters.seed = kai_payload.sampler_seed

        return tgi.GenerateRequest(inputs=kai_payload.prompt, parameters=tgi_parameters)


class VLLMInferenceEngine(InferenceEngine):

    def __init__(self, engine: AsyncLLMEngine, mode=None, model=None):
        self.engine = engine
        self.model = model if model is not None and len(model) > 0 else None
        self.mode = mode if mode is not None and len(mode) > 0 else None

    async def generate_stream(self, payload: kai.GenerationInput) -> AsyncGenerator[str, None]:
        sampling_params = VLLMInferenceEngine.sampling_params(payload)
        request_id = random_uuid()

        results_generator = self.engine.generate(payload.prompt, sampling_params, request_id)

        result_list = []
        common_prefix = ""

        async for request_output in results_generator:
            string = request_output.outputs[0].text
            if string.startswith(common_prefix):
                new_token = string[len(common_prefix):]
                result_list.append(new_token)
                common_prefix = string
                yield new_token
            else:
                result_list.append(string)
                common_prefix = string
                yield string

    def get_model(self) -> str:
        model = self.model
        model_name = "vllm" \
                     + (f"-{self.mode}" if self.mode else "") \
                     + "/" + model

        return model_name


    @staticmethod
    def sampling_params(kai_payload: kai.GenerationInput) -> SamplingParams:
        return SamplingParams(
            top_p=min(max(kai_payload.top_p, 0.001), 0.999) if kai_payload.top_p is not None else 0.999,
            top_k=max(kai_payload.top_k, 1) if kai_payload.top_k is not None else 1,
            temperature=max(kai_payload.temperature, 0.001) if kai_payload.temperature is not None else 0.001,
            # I'm not sure how to map repetition penalty to presence_penalty
            # or frequency_penalty. I think it's frequency_penalty.
            frequency_penalty=max(kai_payload.rep_pen, 0.001) if kai_payload.rep_pen is not None else 0.001,
            max_tokens=kai_payload.max_length,
        )


class DummyInferenceEngine(InferenceEngine):
    def generate_stream(self, payload: kai.GenerationInput) -> Iterator[str]:
        raise util.BridgeException(kai.BasicError(msg="TGI or vLLM wasn't selected correctly!",
                                                  type="Error"))

    def generate(self, payload: kai.GenerationInput) -> kai.GenerationOutput:
        raise util.BridgeException(kai.BasicError(msg="TGI or vLLM wasn't selected correctly!",
                                                  type="Error"))

    def get_model(self) -> str:
        raise util.BridgeException(kai.BasicError(msg="TGI or vLLM wasn't selected correctly!",
                                                  type="Error"))
