# Livekit Murf

![Murf AI Logo](https://murf.ai/public-assets/home/Murf_Logo.png)


Official [Murf AI](https://murf.ai/api) Text-to-Speech integration for [Livekit Agents](https://github.com/livekit/agents) - a framework for building voice and multimodal conversational AI applications.

> **Note**: This integration is maintained by Murf AI. As the official provider of the TTS service, we are committed to actively maintaining and updating this integration.


## Installation

```bash
pip install livekit-murf
```

## Livekit Starter Project 

```python
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess, get_job_context
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import murf, openai, silero, deepgram

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a voice agent build using Murf TTS",
            stt=deepgram.STT(model="nova-3"),
            llm=openai.LLM(model="gpt-4o"),
            tts=murf.TTS(voice="en-US-matthew", style="Conversation"),
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

## Pre-requisites

You'll need an API key from [Murf AI](https://murf.ai/api). It can be set as an environment variable: `MURF_API_KEY` 

> **Compatibility**: Requires livekit-agents >= 1.2.18.