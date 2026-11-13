from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlencode

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from livekit import rtc

from .log import logger
from .models import STTLanguages, STTModels

API_AUTH_HEADER = "api-key"
STREAM_PATH = "/v1/transcription/stream-input"
SUPPORTED_SAMPLE_RATES = (8000, 12000, 16000, 24000, 32000, 44100, 48000)
_MODEL_ORIGINS = {
    "finch_indic_1.0": "wss://in.api.murf.ai",
    "finch_global_1.0": "wss://us-east.api.murf.ai",
}
_INDIC_LANGUAGES = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}
_GLOBAL_LANGUAGES = {"en", "de", "es", "fr", "it", "pt", "nl"}
_RETRYABLE_CLOSE_CODES = {4004, 4008, 4010, 4011, 1006, 1011, 1012, 1013}
_NON_RETRYABLE_CLOSE_CODES = {4001, 4002, 4003, 4005, 4006}
_NON_RETRYABLE_MESSAGES = {"insufficient_credits", "api_key_revoked"}


@dataclass
class STTOptions:
    model: STTModels | str
    language: str
    language_code: LanguageCode
    sample_rate: int
    keywords: list[str]
    alpha: float
    text_formatting: bool
    bvc: bool
    base_url: str | None
    audio_chunk_duration_ms: int

    def ws_url(self) -> str:
        origin = _origin_for(self.model, self.base_url).rstrip("/")
        if origin.endswith(STREAM_PATH):
            base = origin
        else:
            base = f"{origin}{STREAM_PATH}"
        # Keep commas so language=hi,en_IN and keywords stay in Murf's query form.
        return f"{base}?{urlencode(_query_params(self), safe=',')}"


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: STTModels | str = "finch_global_1.0",
        language: STTLanguages | str | list[str] = "auto",
        sample_rate: int = 16000,
        keywords: list[str] | None = None,
        alpha: float = 0.5,
        text_formatting: bool = False,
        bvc: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        audio_chunk_duration_ms: int = 50,
    ) -> None:
        """Create a new instance of Murf STT.

        See https://murf.ai/api/docs/speech-to-text/streaming for the streaming API.

        Args:
            model: ``finch_global_1.0`` (English and European languages, US East) or
                ``finch_indic_1.0`` (English (India) and Indic languages, India).
                Defaults to ``finch_global_1.0``.
            language: Language of the audio. ``auto`` detects among the languages of
                the connected model. Pass several languages as a comma-separated
                string, for example ``"hi,en_IN"`` or ``"fr,en"``, or as a list
                such as ``["hi", "en_IN"]``. Murf then detects only from that set.
                BCP-47 tags such as ``en-US`` or ``hi-IN`` are mapped to Murf's codes.
                Any other code is sent as given, including languages this plugin does
                not list.
            sample_rate: Sample rate of audio sent to Murf, in Hz. Incoming frames are
                resampled to this rate. Must be one of 8000, 12000, 16000, 24000,
                32000, 44100, 48000. Defaults to 16000.
            keywords: Words or phrases to boost. Joined into Murf's comma-separated
                ``keywords`` query parameter.
            alpha: How strongly to favour ``keywords``, from 0.0 to 1.0. Defaults to 0.5.
            text_formatting: Format numbers and similar terms. Applies to Hindi and
                English (India) only. Defaults to False.
            bvc: Remove background voices before transcription. Defaults to False.
            api_key: Murf API key. Falls back to the ``MURF_API_KEY`` environment variable.
            base_url: Override the model host, for example ``https://in.api.murf.ai``.
                When omitted, Indic languages use ``wss://in.api.murf.ai`` and
                European languages use ``wss://us-east.api.murf.ai``. English and
                languages this plugin does not list keep ``model``.
            http_session: Existing aiohttp session. When omitted, the plugin uses the
                LiveKit HTTP context session.
            audio_chunk_duration_ms: Size of each binary audio frame, from 20 to 100 ms.
                Defaults to 50.

        Raises:
            ValueError: If no API key is available, or a sample rate, alpha, or chunk
                duration is invalid.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=False,
                aligned_transcript="word",
                offline_recognize=False,
            )
        )

        murf_api_key = api_key or os.environ.get("MURF_API_KEY")
        if not murf_api_key:
            raise ValueError("MURF_API_KEY must be set")

        self._api_key = murf_api_key
        self._opts = _build_options(
            model=model,
            language=language,
            sample_rate=sample_rate,
            keywords=keywords or [],
            alpha=alpha,
            text_formatting=text_formatting,
            bvc=bvc,
            base_url=base_url,
            audio_chunk_duration_ms=audio_chunk_duration_ms,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Murf"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[STTLanguages | str | list[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Murf STT is streaming only")

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str | list[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(
            stt=self,
            opts=self._sanitize_options(language=language),
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str | list[str]] = NOT_GIVEN,
        keywords: NotGivenOr[list[str]] = NOT_GIVEN,
        alpha: NotGivenOr[float] = NOT_GIVEN,
        text_formatting: NotGivenOr[bool] = NOT_GIVEN,
        bvc: NotGivenOr[bool] = NOT_GIVEN,
        base_url: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        """Update options that Murf fixes at connection time.

        Open streams reconnect. Model, language, keywords, and audio format cannot
        change on an existing Murf session. A language passed to ``stream`` stays
        on that stream when this update does not include ``language``.
        """
        opts = self._opts
        self._opts = _build_options(
            model=model if is_given(model) else opts.model,
            language=language if is_given(language) else opts.language,
            sample_rate=opts.sample_rate,
            keywords=keywords if is_given(keywords) else opts.keywords,
            alpha=alpha if is_given(alpha) else opts.alpha,
            text_formatting=text_formatting if is_given(text_formatting) else opts.text_formatting,
            bvc=bvc if is_given(bvc) else opts.bvc,
            base_url=base_url if is_given(base_url) else opts.base_url,
            audio_chunk_duration_ms=opts.audio_chunk_duration_ms,
        )
        for stream in self._streams:
            current = stream._opts
            stream.update_options(
                _build_options(
                    model=model if is_given(model) else current.model,
                    language=language if is_given(language) else current.language,
                    sample_rate=current.sample_rate,
                    keywords=keywords if is_given(keywords) else current.keywords,
                    alpha=alpha if is_given(alpha) else current.alpha,
                    text_formatting=(
                        text_formatting if is_given(text_formatting) else current.text_formatting
                    ),
                    bvc=bvc if is_given(bvc) else current.bvc,
                    base_url=base_url if is_given(base_url) else current.base_url,
                    audio_chunk_duration_ms=current.audio_chunk_duration_ms,
                )
            )

    async def aclose(self) -> None:
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)), return_exceptions=True)
        self._streams.clear()

    def _sanitize_options(self, *, language: NotGivenOr[STTLanguages | str | list[str]]) -> STTOptions:
        opts = replace(self._opts, keywords=list(self._opts.keywords))
        if is_given(language):
            murf_language, language_code = _resolve_language(opts.model, language)
            opts.language = murf_language
            opts.language_code = language_code
        routed = _route_model(str(opts.model), opts.language, opts.base_url)
        if routed != opts.model and is_given(language):
            opts.language, opts.language_code = _resolve_language(routed, language)
        opts.model = routed
        return opts


class SpeechStream(stt.SpeechStream):
    _KEEPALIVE_MSG: str = json.dumps({"type": "KeepAlive"})
    _CLOSE_MSG: str = json.dumps({"type": "CloseStream"})
    _FINALIZE_MSG: str = json.dumps({"type": "Finalize"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._last_interim = ""
        self._request_id = ""
        self._server_error: dict[str, Any] | None = None
        self._reconnect_event = asyncio.Event()
        self._audio_duration_collector = _PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

    def update_options(self, opts: STTOptions) -> None:
        self._opts = replace(opts, keywords=list(opts.keywords))
        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # With no audio flowing, this write is the only thing touching the socket,
            # so a drop surfaces here first.
            nonlocal closing_ws
            try:
                while True:
                    await ws.send_str(SpeechStream._KEEPALIVE_MSG)
                    await asyncio.sleep(5)
            except (aiohttp.ClientError, ConnectionError) as exc:
                if closing_ws or self._session.closed:
                    return
                raise _write_closed("keepalive", ws, exc) from exc

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            samples_per_chunk = self._opts.sample_rate * self._opts.audio_chunk_duration_ms // 1000
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_chunk,
            )

            try:
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []
                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(audio_bstream.push(data.data.tobytes()))
                    elif isinstance(data, self._FlushSentinel):
                        frames.extend(audio_bstream.flush())

                    for frame in frames:
                        self._audio_duration_collector.push(frame.duration)
                        await ws.send_bytes(frame.data.tobytes())

                    if isinstance(data, self._FlushSentinel):
                        self._audio_duration_collector.flush()
                        await ws.send_str(SpeechStream._FINALIZE_MSG)

                closing_ws = True
                self._audio_duration_collector.flush()
                await ws.send_str(SpeechStream._CLOSE_MSG)
            except (aiohttp.ClientError, ConnectionError) as exc:
                if closing_ws or self._session.closed:
                    return
                raise _write_closed("send", ws, exc) from exc

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            monitor_cls = getattr(utils.aio, "WebSocketMonitor", None)
            if monitor_cls is not None:
                liveness_timeout = getattr(self._conn_options, "liveness_timeout", 6.0)
                receive = monitor_cls(ws, liveness_timeout=liveness_timeout).receive
            else:
                receive = ws.receive

            while True:
                msg = await receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    logger.error(
                        "[Murf STT] server closed the websocket close_code=%s data=%r extra=%r server_error=%s request_id=%s",
                        ws.close_code,
                        msg.data,
                        msg.extra,
                        self._server_error,
                        self._request_id,
                    )
                    raise _error_from_close(ws, msg, self._server_error)

                if msg.type == aiohttp.WSMsgType.ERROR:
                    if closing_ws or self._session.closed:
                        return
                    logger.error(
                        "[Murf STT] websocket error close_code=%s exception=%r server_error=%s request_id=%s",
                        ws.close_code,
                        ws.exception(),
                        self._server_error,
                        self._request_id,
                    )
                    raise APIConnectionError(
                        f"Murf STT connection lost (close_code={ws.close_code})"
                    ) from ws.exception()

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("[Murf STT] unexpected message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except (APIStatusError, _FatalMurfSTTError):
                    raise
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
                    logger.exception("[Murf STT] failed to process message")

        ws: aiohttp.ClientWebSocketResponse | None = None
        while True:
            closing_ws = False
            try:
                ws = await _connect_ws(
                    self._session,
                    self._opts,
                    self._api_key,
                    self._conn_options.timeout,
                    autoping=getattr(utils.aio, "WebSocketMonitor", None) is None,
                )
                self._speaking = False
                self._last_interim = ""
                self._server_error = None
                send_worker = asyncio.create_task(send_task(ws))
                recv_worker = asyncio.create_task(recv_task(ws))
                keepalive_worker = asyncio.create_task(keepalive_task(ws))
                tasks = [send_worker, recv_worker, keepalive_worker]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if tasks_group in done:
                        await self._raise_connection_failure(ws, tasks_group, recv_worker)

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    try:
                        tasks_group.exception()
                    except (asyncio.CancelledError, asyncio.InvalidStateError):
                        pass
            finally:
                if ws is not None:
                    await ws.close()

    async def _raise_connection_failure(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        tasks_group: asyncio.Future[Any],
        recv_worker: asyncio.Task[None],
    ) -> None:
        group_exc = tasks_group.exception()
        if group_exc is None:
            return
        # A writer usually fails before recv reads the close frame. Wait briefly so
        # the server close code and Error payload are on the exception we raise.
        if not isinstance(group_exc, (APIStatusError, _FatalMurfSTTError)) and not recv_worker.done():
            try:
                await asyncio.wait_for(asyncio.shield(recv_worker), 0.5)
            except asyncio.TimeoutError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.debug("[Murf STT] received error: %s", exc)
        recv_exc = (
            recv_worker.exception()
            if recv_worker.done() and not recv_worker.cancelled()
            else None
        )
        tracked: BaseException = group_exc
        if isinstance(recv_exc, (APIStatusError, APIConnectionError, _FatalMurfSTTError)):
            tracked = recv_exc
        elif self._server_error is not None:
            tracked = _error_from_payload(self._server_error, ws.close_code)
        elif ws.close_code:
            tracked = _error_from_close(
                ws,
                aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None),
                self._server_error,
            )
        logger.error(
            "[Murf STT] connection lost close_code=%s closed=%s ws_exception=%r server_error=%s request_id=%s error=%s",
            ws.close_code,
            ws.closed,
            ws.exception(),
            self._server_error,
            self._request_id,
            tracked,
        )
        if tracked is group_exc:
            raise group_exc
        raise tracked from group_exc

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    def _process_stream_event(self, data: dict[str, Any]) -> None:
        msg_type, raw, transcript = _coerce_turn(data)
        if msg_type == "QuotaWarning":
            logger.warning("[Murf STT] Low balance: %s", data.get("remaining_funds"))
            return
        if msg_type == "Error":
            self._server_error = data
            logger.error(
                "[Murf STT] server error message=%s request_id=%s payload=%s",
                data.get("message"),
                self._request_id,
                data,
            )
            raise _error_from_payload(data, None)
        if msg_type not in ("partial", "final"):
            logger.debug("[Murf STT] unexpected message %s", data)
            return

        request_id = str(raw.get("request_id") or "")
        if request_id:
            self._request_id = request_id

        event_name = raw.get("event")
        if event_name == "StartOfTurn":
            self._last_interim = ""
        # type=final is the turn end. EndOfTurn is the same signal if it arrives
        # on another message, including one with an empty transcript.
        turn_ended = msg_type == "final" or event_name == "EndOfTurn"
        if (event_name == "StartOfTurn" or transcript) and not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

        if not turn_ended:
            # StartOfTurn often has no words yet. Keep a partial that carries
            # text on the TurnInfo even when the outer transcript is empty.
            if not transcript:
                return
            self._last_interim = transcript
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[
                        _speech_data(
                            self._opts.language_code,
                            transcript,
                            raw,
                            include_words=False,
                            time_offset=self.start_time_offset,
                        )
                    ],
                )
            )
            return

        # A pause can end the turn with an empty transcript. The last partial is
        # the utterance, and the turn still has to close or the agent stays speaking.
        if not transcript:
            transcript = self._last_interim
        self._last_interim = ""

        if transcript or self._speaking:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[
                        _speech_data(
                            self._opts.language_code,
                            transcript,
                            raw,
                            include_words=True,
                            time_offset=self.start_time_offset,
                        )
                    ],
                )
            )

        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))


def _build_options(
    *,
    model: STTModels | str,
    language: STTLanguages | str | list[str],
    sample_rate: int,
    keywords: list[str],
    alpha: float,
    text_formatting: bool,
    bvc: bool,
    base_url: str | None,
    audio_chunk_duration_ms: int,
) -> STTOptions:
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(
            f"sample_rate {sample_rate} is not supported. Allowed: {list(SUPPORTED_SAMPLE_RATES)}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    if not 20 <= audio_chunk_duration_ms <= 100:
        raise ValueError("audio_chunk_duration_ms must be between 20 and 100")
    if model not in _MODEL_ORIGINS and not base_url:
        raise ValueError(
            f"unknown Murf STT model {model!r}. Pass base_url, or use one of {list(_MODEL_ORIGINS)}"
        )

    murf_language, language_code = _resolve_language(model, language)
    routed = _route_model(str(model), murf_language, base_url)
    if routed != model:
        model = routed
        murf_language, language_code = _resolve_language(model, language)
    cleaned_keywords = [term.strip() for term in keywords if term and term.strip()]
    if any("," in term for term in cleaned_keywords):
        raise ValueError("keywords cannot contain commas; pass a list of phrases instead")

    return STTOptions(
        model=model,
        language=murf_language,
        language_code=language_code,
        sample_rate=sample_rate,
        keywords=cleaned_keywords,
        alpha=alpha,
        text_formatting=text_formatting,
        bvc=bvc,
        base_url=base_url,
        audio_chunk_duration_ms=audio_chunk_duration_ms,
    )


def _resolve_language(
    model: str, language: STTLanguages | str | list[str]
) -> tuple[str, LanguageCode]:
    raw = _language_input(language)
    if not raw or raw.lower() in {"auto", "multi"}:
        return "auto", LanguageCode("multi")

    if "," in raw:
        parts = [
            _murf_language_token(model, str(LanguageCode(part.strip())))
            for part in raw.split(",")
            if part.strip()
        ]
        if not parts:
            return "auto", LanguageCode("multi")
        return ",".join(parts), LanguageCode("multi")

    murf_language = _murf_language_token(model, str(LanguageCode(raw)))
    return murf_language, _language_code_for(murf_language)


def _language_input(language: STTLanguages | str | list[str]) -> str:
    if isinstance(language, (list, tuple)):
        return ",".join(str(part).strip() for part in language if str(part).strip())
    return str(language).strip()


def _language_code_for(murf_language: str) -> LanguageCode:
    if murf_language == "en_IN":
        return LanguageCode("en-IN")
    if murf_language in {"auto"} or "," in murf_language:
        return LanguageCode("multi")
    return LanguageCode(murf_language.replace("_", "-"))


def _murf_language_token(model: str, language: str) -> str:
    token = language.strip().replace("_", "-")
    pieces = token.split("-")
    base = pieces[0].lower()
    indic = str(model).startswith("finch_indic")

    if indic:
        if base == "en":
            return "en_IN"
        if base in _INDIC_LANGUAGES:
            return base
    else:
        if base in _GLOBAL_LANGUAGES:
            return base

    return language.strip()


def _origin_for(model: str, base_url: str | None) -> str:
    if base_url:
        origin = base_url.strip().rstrip("/")
        if origin.startswith("https://"):
            return "wss://" + origin[len("https://") :]
        if origin.startswith("http://"):
            return "ws://" + origin[len("http://") :]
        if origin.startswith(("ws://", "wss://")):
            return origin
        return f"wss://{origin}"

    origin = _MODEL_ORIGINS.get(model)
    if not origin:
        raise ValueError(f"no Murf STT host for model {model!r}")
    return origin


_ENDPOINT_TARGET = {
    "indic": ("wss://in.api.murf.ai", "finch_indic_1.0"),
    "global": ("wss://us-east.api.murf.ai", "finch_global_1.0"),
}


def _language_endpoint(token: str) -> str | None:
    """Return ``indic`` or ``global`` when this plugin knows the code.

    English and codes this plugin does not list return ``None``. Those keep the
    caller's model and host.
    """
    base = token.strip().replace("-", "_").split("_")[0].lower()
    if base == "en":
        return None
    if base in _GLOBAL_LANGUAGES:
        return "global"
    if base in _INDIC_LANGUAGES:
        return "indic"
    return None


def _route_model(model: str, language: str, base_url: str | None) -> str:
    """Pick the Murf model from the language when the caller did not set a host."""
    if base_url or not language or language == "auto" or model not in _MODEL_ORIGINS:
        return model

    families = {
        family
        for token in language.split(",")
        if (family := _language_endpoint(token)) is not None
    }
    if len(families) != 1:
        return model

    origin, routed = _ENDPOINT_TARGET[families.pop()]
    if _MODEL_ORIGINS[model] == origin:
        return model
    logger.info(
        "[Murf STT] language %s is served by %s at %s",
        language,
        routed,
        origin,
    )
    return routed


def _query_params(opts: STTOptions) -> dict[str, str]:
    params = {
        "model": str(opts.model),
        "language": opts.language,
    }
    if opts.keywords:
        params["keywords"] = ",".join(opts.keywords)
        params["alpha"] = str(opts.alpha)
    if opts.text_formatting:
        params["text_formatting"] = "true"
    if opts.bvc:
        params["bvc"] = "true"
    return params


async def _connect_ws(
    session: aiohttp.ClientSession,
    opts: STTOptions,
    api_key: str,
    timeout: float,
    *,
    autoping: bool = True,
) -> aiohttp.ClientWebSocketResponse:
    ws: aiohttp.ClientWebSocketResponse | None = None
    try:
        ws = await asyncio.wait_for(
            session.ws_connect(
                opts.ws_url(),
                headers={API_AUTH_HEADER: api_key},
                # The stream reader probes the socket itself. aiohttp's ping races that
                # probe and surfaces a dead peer as a late 1011 instead of a reconnect.
                autoping=autoping,
            ),
            timeout,
        )
        await ws.send_str(
            json.dumps(
                {
                    "type": "AudioConfig",
                    "codec": "pcm_s16le",
                    "sample_rate": opts.sample_rate,
                    "channels": 1,
                }
            )
        )
        return ws
    except aiohttp.WSServerHandshakeError as exc:
        raise _for_stream(
            APIStatusError(
                message=exc.message,
                status_code=exc.status,
                body=None,
                retryable=exc.status >= 500,
            )
        ) from exc
    except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as exc:
        raise APIConnectionError("failed to connect to Murf STT") from exc
    except aiohttp.ClientError as exc:
        if ws is not None and not ws.closed:
            await ws.close()
        raise APIConnectionError("failed to connect to Murf STT") from exc
    except Exception:
        if ws is not None and not ws.closed:
            await ws.close()
        raise


def _coerce_turn(data: dict[str, Any]) -> tuple[str | None, dict[str, Any], str]:
    """Normalize a Murf transcript message.

    The server usually wraps a TurnInfo in ``type=partial|final``. It can also
    send the TurnInfo itself, or put the only text on ``rawEvent.transcript``.
    """
    msg_type = data.get("type")
    if msg_type in ("QuotaWarning", "Error"):
        return msg_type if isinstance(msg_type, str) else None, {}, ""

    raw = data.get("rawEvent")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if not isinstance(raw, dict):
        raw = {}

    if msg_type == "TurnInfo" or (
        msg_type not in ("partial", "final") and data.get("event") in ("StartOfTurn", "Update", "EndOfTurn")
    ):
        raw = data
        msg_type = "final" if data.get("event") == "EndOfTurn" else "partial"

    transcript = data.get("transcript") if raw is not data else ""
    if not isinstance(transcript, str) or not transcript:
        nested = raw.get("transcript")
        transcript = nested if isinstance(nested, str) else ""
    if not transcript:
        words = raw.get("words")
        if isinstance(words, list):
            transcript = " ".join(
                word.get("word", "")
                for word in words
                if isinstance(word, dict) and isinstance(word.get("word"), str) and word.get("word")
            )
    return msg_type if isinstance(msg_type, str) else None, raw, transcript


def _speech_data(
    language: LanguageCode,
    transcript: str,
    raw: dict[str, Any],
    *,
    include_words: bool,
    time_offset: float,
) -> stt.SpeechData:
    words: list[TimedString] | None = None
    raw_words = raw.get("words") or []
    if include_words and raw_words:
        words = [
            TimedString(
                text=word.get("word", ""),
                start_time=float(word.get("start") or 0) + time_offset,
                end_time=float(word.get("end") or 0) + time_offset,
                start_time_offset=time_offset,
            )
            for word in raw_words
        ]

    start = raw.get("audio_window_start")
    end = raw.get("audio_window_end")
    if start is None and words:
        start = float(words[0].start_time) - time_offset
    if end is None and words:
        end = float(words[-1].end_time) - time_offset

    return stt.SpeechData(
        language=language,
        text=transcript,
        start_time=(float(start) if start is not None else 0.0) + time_offset,
        end_time=(float(end) if end is not None else 0.0) + time_offset,
        confidence=0.0,
        words=words,
    )


def _is_retryable(close_code: int | None, message: str | None) -> bool:
    if message in _NON_RETRYABLE_MESSAGES or (message or "").startswith("Invalid"):
        return False
    # Config mismatch. Reconnecting will get the same rejection.
    if message and "is not served at this endpoint" in message:
        return False
    if close_code in _NON_RETRYABLE_CLOSE_CODES:
        return False
    if close_code in _RETRYABLE_CLOSE_CODES:
        return True
    if close_code is None or close_code < 0:
        return True
    return close_code < 4000 or close_code >= 5000


class _FatalMurfSTTError(Exception):
    """Permanent Murf failure.

    Agents 1.4.5 retries every APIError and ignores ``retryable``, so a fatal
    failure is not an APIError. The wrapped status error still has
    ``retryable=False`` for releases that honor the flag.
    """

    def __init__(self, error: APIStatusError) -> None:
        super().__init__(str(error))
        self.error = error
        self.retryable = False


def _for_stream(error: APIStatusError) -> Exception:
    if error.retryable:
        return error
    return _FatalMurfSTTError(error)


def _error_from_payload(data: dict[str, Any], close_code: int | None) -> Exception:
    message = str(data.get("message") or "Murf STT error")
    return _for_stream(
        APIStatusError(
            message=message,
            status_code=close_code if close_code is not None else -1,
            request_id=str((data.get("rawEvent") or {}).get("request_id") or "") or None,
            body=data,
            retryable=_is_retryable(close_code, message),
        )
    )


def _write_closed(
    source: str,
    ws: aiohttp.ClientWebSocketResponse,
    exc: BaseException,
) -> APIConnectionError:
    logger.error(
        "[Murf STT] %s write failed close_code=%s closed=%s ws_exception=%r cause=%r",
        source,
        ws.close_code,
        ws.closed,
        ws.exception(),
        exc,
    )
    return APIConnectionError(
        f"Murf STT connection closed unexpectedly during {source} (close_code={ws.close_code})"
    )


def _error_from_close(
    ws: aiohttp.ClientWebSocketResponse,
    msg: aiohttp.WSMessage,
    server_error: dict[str, Any] | None,
) -> Exception:
    if server_error:
        return _error_from_payload(server_error, ws.close_code)
    return _for_stream(
        APIStatusError(
            message=f"Murf STT connection closed unexpectedly (close_code={ws.close_code})",
            status_code=ws.close_code or -1,
            body=f"{msg.data=} {msg.extra=}",
            retryable=_is_retryable(ws.close_code, None),
        )
    )


class _PeriodicCollector:
    def __init__(self, callback: Any, *, duration: float) -> None:
        self._duration = duration
        self._callback = callback
        self._last_flush = time.monotonic()
        self._total = 0.0

    def push(self, value: float) -> None:
        self._total += value
        if time.monotonic() - self._last_flush >= self._duration:
            self.flush()

    def flush(self) -> None:
        if self._total > 0:
            self._callback(self._total)
            self._total = 0.0
        self._last_flush = time.monotonic()
