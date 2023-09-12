
# tgi-kai-bridge

Minimal API translation layer to make [text-generation-inference](https://github.com/huggingface/text-generation-inference) accessible to KoboldAI clients including [KoboldAI](https://github.com/KoboldAI/KoboldAI-Client), [SillyTavern](https://github.com/SillyTavern/SillyTavern) and [AI-Horde-Worker](https://github.com/Haidra-Org/AI-Horde-Worker)

Dockerfile (not tested) includes TGI and connects it to the [AI Horde](https://aihorde.net/)

## Configuration

Environment Variables:

`KAI_PORT` - port to listen on for KAI clients (default `5000`)  
`KAI_HOST` - hostname to listen on (default `127.0.0.1`)  
`TGI_ENDPOINT` - URL to TGI REST API (default `http://127.0.0.1:3000`)  
`TGI_MODE` - additional information to add to the model name (default `""`)

## Known Issues

- some options of KAI are not supported by TGI API
- no EOS token ban
- outputs get trimmed and can cause words to get lumped together (e.g. when doing story writing)
