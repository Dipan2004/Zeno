"""
ZENO Voice Input Manager - Push-to-Talk STT Coordinator

Phase 5: Voice Input  (FIXED - v2)

Bugs fixed:
- Monitor thread from previous session blocked new activations
- _is_listening flag left stale after stop_listening()
- faster-whisper tried to re-download model on every activation
- Race condition between monitor thread and start_listening()

Architecture (unchanged):
- Main process: VoiceInputManager (coordinator)
- STT process: whisper_stt_worker (separate process)
- IPC: multiprocessing.Queue + Event
- Model: downloaded once, then local_files_only=True on every subsequent load
"""

import logging
import multiprocessing as mp
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class VoiceInputManager:
    """
    Manages STT process lifecycle for push-to-talk.

    Key design rules (v2):
    - A fresh text_queue and stop_event are created on EVERY activation.
      This means old monitor threads can never accidentally drain a new queue.
    - stop_listening() is idempotent and safe to call from any thread.
    - _is_listening is protected by a threading.Lock so concurrent calls
      to start_listening() cannot double-spawn a process.
    - Whisper model is loaded with local_files_only=True after the first
      successful download so it never hits the network again.
    """

    IDLE_TIMEOUT = 10.0  # seconds of silence before auto-stop

    def __init__(
        self,
        callback: Callable[[Optional[str]], None],
        engine: str = "vosk",
        model_path: Optional[Path] = "C:\\Users\\KIIT0001\\Desktop\\ZENO\\zeno\\models",
    ):
        """
        Args:
            callback:   Called with transcribed text (str) on success,
                        or None on error/timeout.
            engine:     "whisper" (faster-whisper, default) or "vosk".
            model_path: Required only when engine="vosk".
        """
        self.engine     = engine.lower()
        self.model_path = model_path
        self.callback   = callback

        # Guards concurrent calls to start_listening()
        self._state_lock = threading.Lock()

        # IPC — recreated fresh on every activation
        self.text_queue: Optional[mp.Queue] = None
        self.stop_event: Optional[mp.Event] = None

        # Process / thread handles
        self.stt_process:    Optional[mp.Process] = None
        self.monitor_thread: Optional[threading.Thread] = None

        # Set once at shutdown(), never cleared
        self._shutdown = threading.Event()

        self._is_listening = False

        logger.info(f"VoiceInputManager initialized (engine: {engine})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_listening(self):
        """
        Spawn the STT process and start the monitor thread.
        Safe to call from hotkey callback (any thread).
        """
        with self._state_lock:
            if self._is_listening:
                logger.warning("Already listening — ignoring duplicate activation")
                return

            if self._shutdown.is_set():
                logger.warning("VoiceInputManager has been shut down")
                return

            # Create FRESH IPC primitives for this activation
            self.text_queue = mp.Queue()
            self.stop_event = mp.Event()

            # Pick worker function
            if self.engine == "whisper":
                worker_fn   = whisper_stt_worker
                worker_args = (self.text_queue, self.stop_event)
            else:
                if self.model_path is None:
                    logger.error("model_path is required for Vosk engine")
                    return
                worker_fn   = vosk_stt_worker
                worker_args = (self.model_path, self.text_queue, self.stop_event)

            self.stt_process = mp.Process(
                target=worker_fn,
                args=worker_args,
                daemon=True,
                name="ZENO-STT",
            )
            self.stt_process.start()

            # Mark as listening BEFORE starting monitor thread
            self._is_listening = True

            # Capture current queue/event for THIS activation's monitor thread
            # so a future activation cannot steal them
            current_queue = self.text_queue
            current_event = self.stop_event

        # Start monitor thread OUTSIDE the lock (it calls stop_listening)
        self.monitor_thread = threading.Thread(
            target=self._monitor_stt,
            args=(current_queue, current_event),
            daemon=True,
            name="ZENO-STT-Monitor",
        )
        self.monitor_thread.start()

        logger.info(f"STT process started ({self.engine}) - listening...")

    def stop_listening(self):
        """
        Stop the STT process.
        Idempotent — safe to call multiple times or from any thread.
        """
        with self._state_lock:
            if not self._is_listening:
                return  # already stopped

            self._is_listening = False  # mark stopped first

            proc = self.stt_process
            evt  = self.stop_event
            self.stt_process = None     # clear handle

        # Signal worker (outside lock)
        if evt:
            evt.set()

        if proc and proc.is_alive():
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
                logger.warning("STT process force-terminated")

        logger.info("STT process stopped")

    def is_listening(self) -> bool:
        return self._is_listening

    def shutdown(self):
        """Permanent shutdown — call once when ZENO exits."""
        logger.info("Shutting down VoiceInputManager")
        self._shutdown.set()
        self.stop_listening()

    # ------------------------------------------------------------------
    # Internal monitor
    # ------------------------------------------------------------------

    def _monitor_stt(self, text_queue: mp.Queue, stop_event: mp.Event):
        """
        Runs in a background daemon thread.

        text_queue and stop_event are passed as ARGUMENTS (not read from self)
        so this specific monitor thread always operates on the IPC objects
        it was created with, even after a new activation has replaced
        self.text_queue and self.stop_event.
        """
        last_activity = time.time()

        while not self._shutdown.is_set():
            # If stop_listening() was called externally, bail
            if not self._is_listening:
                break

            try:
                text = text_queue.get(timeout=0.5)

                if text is not None:
                    # SUCCESS: clean up process, then deliver result
                    logger.info(f"Transcribed: {text}")
                    self.stop_listening()
                    try:
                        self.callback(text)
                    except Exception as e:
                        logger.error(f"Callback error: {e}", exc_info=True)
                    break

                else:
                    # Worker sent None → error in worker
                    logger.error("STT worker signalled an error")
                    self.stop_listening()
                    try:
                        self.callback(None)
                    except Exception as e:
                        logger.error(f"Error-callback failed: {e}", exc_info=True)
                    break

            except Empty:
                # Check idle timeout
                if time.time() - last_activity > self.IDLE_TIMEOUT:
                    logger.info("STT idle timeout — stopping")
                    self.stop_listening()
                    break

            except Exception as e:
                logger.error(f"Monitor error: {e}", exc_info=True)
                self.stop_listening()
                break


# ======================================================================
# Whisper STT worker  (faster-whisper, CPU, offline after first download)
# ======================================================================

def whisper_stt_worker(text_queue: mp.Queue, stop_event: mp.Event):
    """
    Runs in a separate process.

    Records audio with silence detection, then batch-transcribes with
    faster-whisper tiny.en on CPU.

    KEY FIX: uses local_files_only=True so the model is NEVER re-downloaded
    after the first pull. Falls back to a normal download only on cache miss.
    """
    import struct
    import math
    import wave
    import tempfile
    import os
    import pyaudio

    worker_logger = logging.getLogger(f"{__name__}.whisper_worker")

    # Audio config
    RATE                = 16000
    CHANNELS            = 1
    CHUNK               = 1024
    FORMAT              = pyaudio.paInt16
    SILENCE_THRESHOLD   = 30    # RMS energy
    SILENCE_DURATION    = 1.5   # seconds of quiet to end utterance
    MAX_RECORD_SECONDS  = 30
    MIN_SPEECH_CHUNKS   = 3
    LOOKBACK_SECONDS    = 0.5

    stream   = None
    audio    = None
    tmp_path = None

    try:
        worker_logger.info("Opening microphone for Whisper STT...")
        audio = pyaudio.PyAudio()

        try:
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            worker_logger.error(f"Failed to open microphone: {e}")
            text_queue.put(None)
            return

        worker_logger.info("Whisper STT worker ready — listening for speech...")

        # Silence-detection recording loop
        lookback_size   = int(LOOKBACK_SECONDS * RATE / CHUNK)
        ring_buffer     = []
        speech_frames   = []
        silent_chunks   = 0
        speech_chunks   = 0
        chunks_silence  = int(SILENCE_DURATION * RATE / CHUNK)
        max_chunks      = int(MAX_RECORD_SECONDS * RATE / CHUNK)
        speech_started  = False
        total_chunks    = 0

        while not stop_event.is_set() and total_chunks < max_chunks:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                total_chunks += 1

                count  = len(data) // 2
                shorts = struct.unpack(f"{count}h", data)
                rms    = math.sqrt(sum(s * s for s in shorts) / count) if count else 0

                # Log RMS every ~2 s for debugging
                if total_chunks % ((RATE // CHUNK) * 2) == 0:
                    worker_logger.info(
                        f"RMS level: {rms:.0f} (threshold: {SILENCE_THRESHOLD})"
                    )

                if rms > SILENCE_THRESHOLD:
                    silent_chunks = 0
                    speech_chunks += 1
                    if not speech_started:
                        speech_started = True
                        worker_logger.info(f"Speech detected (RMS: {rms:.0f})")
                        speech_frames.extend(ring_buffer)
                        ring_buffer.clear()
                    speech_frames.append(data)
                else:
                    silent_chunks += 1
                    if speech_started:
                        speech_frames.append(data)
                    else:
                        ring_buffer.append(data)
                        if len(ring_buffer) > lookback_size:
                            ring_buffer.pop(0)

                if speech_started and silent_chunks >= chunks_silence:
                    worker_logger.info("End of speech detected (silence)")
                    break

            except Exception as e:
                worker_logger.error(f"Audio read error: {e}")
                break

        # Release mic immediately so next activation can grab it
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            stream = None
        if audio:
            try:
                audio.terminate()
            except Exception:
                pass
            audio = None

        if stop_event.is_set():
            worker_logger.info("Stop signal received — skipping transcription")
            return

        if not speech_started or speech_chunks < MIN_SPEECH_CHUNKS:
            worker_logger.info("No speech detected (or too short)")
            text_queue.put(None)
            return

        # Save speech frames to temp WAV
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)

        speech_duration = len(speech_frames) * CHUNK / RATE
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"".join(speech_frames))

        worker_logger.info(
            f"Recorded {speech_duration:.1f}s of audio — transcribing..."
        )

        # ── Transcribe with faster-whisper ────────────────────────────
        from faster_whisper import WhisperModel

        MODEL_NAME = "tiny.en"

        # Try local cache first — no network needed after first download
        try:
            worker_logger.info(f"Loading {MODEL_NAME} from local cache...")
            model = WhisperModel(
                MODEL_NAME,
                device="cpu",
                compute_type="int8",
                local_files_only=True,      # ← KEY FIX
            )
        except Exception:
            # Cache miss: download once, after this it will always be local
            worker_logger.info(
                f"Local cache miss — downloading {MODEL_NAME} (one-time only)..."
            )
            model = WhisperModel(
                MODEL_NAME,
                device="cpu",
                compute_type="int8",
                local_files_only=False,
            )

        segments, _ = model.transcribe(
            tmp_path,
            language="en",
            beam_size=1,
            vad_filter=True,                # strips leading/trailing silence
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()

        # De-duplicate consecutive repeated phrases (Whisper hallucination)
        if text:
            parts   = [s.strip() for s in text.replace(".", ".\n").split("\n") if s.strip()]
            deduped = []
            for part in parts:
                if not deduped or part.rstrip(".") != deduped[-1].rstrip("."):
                    deduped.append(part)
            text = " ".join(deduped)

        if text:
            worker_logger.info(f"Whisper recognized: {text}")
            text_queue.put(text)
        else:
            worker_logger.info("Whisper returned empty text")
            text_queue.put(None)

    except KeyboardInterrupt:
        worker_logger.info("Whisper STT worker interrupted")

    except Exception as e:
        worker_logger.error(f"Whisper STT worker error: {e}", exc_info=True)
        try:
            text_queue.put(None)
        except Exception:
            pass

    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if audio:
            try:
                audio.terminate()
            except Exception:
                pass
        if tmp_path:
            try:
                import os as _os
                if _os.path.exists(tmp_path):
                    _os.unlink(tmp_path)
            except Exception:
                pass
        worker_logger.info("Whisper STT worker terminated")


# ======================================================================
# Vosk STT worker  (streaming, fully offline — no download ever needed)
# ======================================================================

def vosk_stt_worker(model_path: Path, text_queue: mp.Queue, stop_event: mp.Event):
    """
    Vosk streaming STT — fully offline.
    Alternative to Whisper; requires a downloaded Vosk model folder.
    """
    import json
    import pyaudio

    worker_logger = logging.getLogger(f"{__name__}.vosk_worker")
    stream = None
    audio  = None

    try:
        from vosk import Model, KaldiRecognizer

        worker_logger.info("Initializing Vosk model...")
        model      = Model(str(model_path))
        recognizer = KaldiRecognizer(model, 16000)
        worker_logger.info("Vosk initialized")

        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4000,
            )
        except Exception as e:
            worker_logger.error(f"Failed to open microphone: {e}")
            text_queue.put(None)
            return

        worker_logger.info("Vosk STT worker ready — listening...")

        while not stop_event.is_set():
            try:
                data = stream.read(4000, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text   = result.get("text", "").strip()
                    if text:
                        worker_logger.info(f"Vosk recognized: {text}")
                        text_queue.put(text)
                        break
            except Exception as e:
                worker_logger.error(f"Audio error: {e}", exc_info=True)

    except Exception as e:
        worker_logger.error(f"Vosk worker error: {e}", exc_info=True)
        try:
            text_queue.put(None)
        except Exception:
            pass

    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if audio:
            try:
                audio.terminate()
            except Exception:
                pass
        worker_logger.info("Vosk STT worker terminated")