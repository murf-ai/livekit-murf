# Livekit Murf

![Murf AI Logo](https://murf.ai/public-assets/home/Murf_Logo.png)


[Murf AI](https://murf.ai/api) text-to-speech and speech-to-text for [Livekit Agents](https://github.com/livekit/agents).

> **Note**: This integration is maintained by Murf AI.

Use this package for both speech and transcription. `murf.TTS` follows the same Murf requests as [LiveKit's Murf TTS plugin](https://docs.livekit.io/agents/models/tts/murf/): Falcon by default, `api-key` header, and the same voice, style, rate, pitch, and locale fields. Do not install `livekit-plugins-murf` in the same environment. Both packages provide `livekit.plugins.murf`, and the later install replaces the other.

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

`murf.STT` streams PCM audio to Murf's [streaming transcription](https://murf.ai/api/docs/speech-to-text/streaming) WebSocket.

`finch_global_1.0` (`us-east.api.murf.ai`) transcribes English and European languages. `finch_indic_1.0` (`in.api.murf.ai`) transcribes English (India) and Indic languages. `language="auto"` detects among the languages of the connected model. A comma-separated list, or a list of codes, detects only from those languages.

```python
stt=murf.STT(model="finch_indic_1.0", language="hi,te")
```

Partials are interim transcripts. Murf sends a final when the caller pauses. LiveKit's turn detection decides when the agent replies.

## Pre-requisites

You'll need an API key from [Murf AI](https://murf.ai/api). It can be set as an environment variable: `MURF_API_KEY` 

> **Compatibility**: Requires livekit-agents >= 1.4.4.