# Livekit Murf

![Murf AI Logo](https://murf.ai/public-assets/home/Murf_Logo.png)


[Murf AI](https://murf.ai/api) text-to-speech and speech-to-text for [Livekit Agents](https://github.com/livekit/agents).

> **Note**: This integration is maintained by Murf AI.

Use this package for both speech and transcription. `murf.TTS` follows the same Murf requests as [LiveKit's Murf TTS plugin](https://docs.livekit.io/agents/models/tts/murf/): Falcon by default, `api-key` header, and the same voice, style, speed, pitch, and locale fields. Do not install `livekit-plugins-murf` in the same environment. Both packages provide `livekit.plugins.murf`, and the later install replaces the other.

## Installation

```bash
pip install livekit-murf
```

```python
from livekit.plugins import murf

session = AgentSession(
    stt=murf.STT(model="finch_indic_1.0", language="hi,te"),
    tts=murf.TTS(voice="Pooja", style="Conversational", model="FALCON"),
)
```

## Livekit Starter Project

```python
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess, get_job_context
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import murf, openai, silero

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a voice agent build using Murf",
            stt=murf.STT(model="finch_indic_1.0", language="hi,te"),
            llm=openai.LLM(model="gpt-4o"),
            tts=murf.TTS(voice="Pooja", style="Conversational", model="FALCON"),
            vad=get_job_context().proc.userdata["vad"],
        )

    async def on_enter(self):
        await self.session.say("Hi, I am a voice agent powered by Murf, how can I help you?")



async def entrypoint(ctx: JobContext):
    await ctx.connect()
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession()

    await session.start(
        agent=MyAgent(),
        room=ctx.room
    )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
```

## Speech to Text

`murf.STT` streams PCM audio to Murf's [streaming transcription](https://murf.ai/api/docs/speech-to-text/streaming) WebSocket. Partials are interim transcripts. Murf sends a final when the caller pauses. LiveKit's turn detection decides when the agent replies.

```python
stt = murf.STT(model="finch_indic_1.0", language="hi,te")
```

### Parameters

- **`model`** (optional) — Default: `finch_global_1.0`. `finch_global_1.0` (`us-east.api.murf.ai`) transcribes English and European languages. `finch_indic_1.0` (`in.api.murf.ai`) transcribes English (India) and Indic languages.
- **`language`** (optional) — Default: `auto`, which detects among the languages of the connected model. A comma-separated string or a list of codes, such as `"hi,te"` or `["hi", "te"]`, detects only from those languages. BCP-47 tags such as `en-US` or `hi-IN` are mapped to Murf's codes.
- **`sample_rate`** (optional) — Default: `16000`. Incoming audio is resampled to this rate. One of `8000`, `12000`, `16000`, `24000`, `32000`, `44100`, or `48000`.
- **`keywords`** (optional) — Words or phrases to boost. Sent as Murf's comma-separated `keywords` query parameter.
- **`alpha`** (optional) — Default: `0.5`. How strongly to favour `keywords`, from `0.0` to `1.0`.
- **`text_formatting`** (optional) — Default: `False`. Format numbers and similar terms. Applies to Hindi and English (India) only.
- **`bvc`** (optional) — Default: `False`. Remove background voices before transcription.

The API key is read from `MURF_API_KEY` when `api_key` is omitted.

## Text to Speech

`murf.TTS` synthesizes speech with Murf. It uses the same request fields as [LiveKit's Murf TTS plugin](https://docs.livekit.io/agents/models/tts/murf/). Audio streams over a WebSocket. Set `streaming=False` to synthesize over HTTP instead.

```python
tts = murf.TTS(
    voice="Pooja",
    style="Conversational",
    model="FALCON",
)
```

### Parameters

- **`model`** (optional) — Default: `FALCON`. Murf TTS model. Valid values are `FALCON` and `GEN2`.
- **`voice`** (optional) — Default: `Gordon`. Voice ID from [Murf's voice library](https://murf.ai/api/docs/voices-styles/voice-library).
- **`style`** (optional) — Default: `Conversation`. Voice style, for example `Conversational`, `Narration`, or `Promo`. See the [voice library](https://murf.ai/api/docs/voices-styles/voice-library) for the styles each voice supports.
- **`locale`** (optional) — Locale for synthesis, for example `en-US` or `hi-IN`. When omitted, the locale is inferred from the voice.
- **`speed`** (optional) — Speech speed. Positive values are faster and negative values are slower. Valid range: `-50` to `50`.
- **`pitch`** (optional) — Speech pitch. Positive values are higher and negative values are lower. Valid range: `-50` to `50`.

`sample_rate` defaults to `24000` and `encoding` defaults to `pcm`. The API key is read from `MURF_API_KEY` when `api_key` is omitted. `update_options()` changes `locale`, `voice`, `style`, `speed`, and `pitch` for later audio.

## Pre-requisites

You'll need an API key from [Murf AI](https://murf.ai/api). It can be set as an environment variable: `MURF_API_KEY` 

> **Compatibility**: Requires livekit-agents >= 1.5.2.