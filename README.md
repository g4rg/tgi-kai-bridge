
# tgi-kai-bridge

Minimal API translation layer to make [text-generation-inference](https://github.com/huggingface/text-generation-inference) accessible to KoboldAI clients including [KoboldAI](https://github.com/KoboldAI/KoboldAI-Client), [TavernAI](https://github.com/TavernAI/TavernAI), [SillyTavern](https://github.com/SillyTavern/SillyTavern) and [AI-Horde-Worker](https://github.com/Haidra-Org/AI-Horde-Worker)

Dockerfile (not tested) includes TGI and connects it to the [AI Horde](https://aihorde.net/)

## Configuration

Environment Variables:

`KAI_PORT` - port to listen on for KAI clients (default `5000`)  
`KAI_HOST` - hostname to listen on (default `127.0.0.1`)  
`TGI_ENDPOINT` - URL to TGI REST API (default `http://127.0.0.1:3000`)  
`TGI_MODE` - additional information to add to the model name  
`TGI_MODEL` - model name override

## Limitations

- only supports `temperature`, `top_p`, `top_k` and `rep_pen` sampler settings
- no (EOS) token ban
