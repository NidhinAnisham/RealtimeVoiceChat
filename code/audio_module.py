import asyncio
import logging
import os
import struct
import threading
import time
from collections import namedtuple
from queue import Queue
from typing import Callable, Generator, Optional

import numpy as np
from huggingface_hub import hf_hub_download
# Assuming RealtimeTTS is installed and available
from RealtimeTTS import (TextToAudioStream)
from orpheus_engine import OrpheusEngine

logger = logging.getLogger(__name__)

# Default configuration constants
# START_ENGINE is removed as Orpheus is now the only engine.
Silence = namedtuple("Silence", ("comma", "sentence", "default"))
ENGINE_SILENCES = {
    # "coqui":   Silence(comma=0.3, sentence=0.6, default=0.3), # Removed
    # "kokoro":  Silence(comma=0.3, sentence=0.6, default=0.3), # Removed
    "orpheus": Silence(comma=0.3, sentence=0.6, default=0.3), # Kept for Orpheus
}
# Stream chunk sizes are general and can be kept if Orpheus uses them, or removed if not.
# For now, keeping them as they might be used by RealtimeTTS stream wrapper.
QUICK_ANSWER_STREAM_CHUNK_SIZE = 8
FINAL_ANSWER_STREAM_CHUNK_SIZE = 30

# Removed: create_directory and ensure_lasinya_models functions as Coqui is no longer used.

class AudioProcessor:
    """
    Manages Text-to-Speech (TTS) synthesis using the Orpheus engine via RealtimeTTS.

    This class initializes the Orpheus TTS engine, configures it for streaming output,
    measures initial latency (TTFT), and provides methods to synthesize audio
    from text strings or generators, placing the resulting audio chunks into a queue.
    It handles the synthesis lifecycle, including optional callbacks upon
    receiving the first audio chunk.
    """
    def __init__(
            self,
            # engine parameter removed, fixed to "orpheus"
            orpheus_model: str = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf", # Default model path
        ) -> None:
        """
        Initializes the AudioProcessor with the Orpheus TTS engine.

        Sets up the Orpheus engine, configures the RealtimeTTS stream, and performs
        an initial synthesis to measure Time To First Audio chunk (TTFA).

        Args:
            orpheus_model: The path or identifier for the Orpheus GGUF model file.
        """
        self.engine_name = "orpheus" # Fixed engine
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.audio_chunks = asyncio.Queue() # Queue for synthesized audio output
        self.orpheus_model = orpheus_model

        # Silence settings are now directly for Orpheus
        self.silence = ENGINE_SILENCES[self.engine_name]
        # current_stream_chunk_size might not be relevant for OrpheusEngine directly,
        # but TextToAudioStream might use it. Keeping for now.
        self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE

        # Initialize Orpheus Engine
        logger.info(f"👄⚙️ Initializing OrpheusEngine with model: {self.orpheus_model}")
        self.engine = OrpheusEngine(
            model=self.orpheus_model, # RealtimeTTS OrpheusEngine uses model_path
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=1200, # Renamed from max_tokens for clarity if RealtimeTTS uses this
        )
        # TODO: Verify OrpheusVoice usage. Assuming "tara" is a valid built-in or requires a file.
        # If "tara" is a placeholder that needs a file, this might need adjustment.
        # For GGUF models, voice often comes from the model itself or speaker embeddings.
        # The OrpheusEngine in RealtimeTTS might handle voice selection differently.
        # The original code had:
        # voice = OrpheusVoice("tara")
        # self.engine.set_voice(voice)
        # This part is commented out as GGUF models usually don't require external voice files in this manner.
        # If a specific speaker ID or embedding is needed, it should be configured via OrpheusEngine parameters.
        # For now, assuming default voice from the model or engine.
        logger.info("👄⚙️ OrpheusEngine initialized. Voice configuration depends on model/engine defaults.")

        # Initialize the RealtimeTTS stream
        self.stream = TextToAudioStream(
            self.engine,
            muted=True, # Do not play audio directly
            playout_chunk_size=4096, # Internal chunk size for processing
            on_audio_stream_stop=self.on_audio_stream_stop,
        )

        # Removed Coqui-specific stream_chunk_size setting logic

        # Prewarm the engine
        logger.info("👄🔥 Prewarming Orpheus engine...")
        self.stream.feed("Prewarm") # Using "Prewarm" as it's more descriptive than "prewarm"
        play_kwargs = dict(
            log_synthesized_text=False,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )
        self.stream.play(**play_kwargs)
        while self.stream.is_playing():
            time.sleep(0.01)
        self.finished_event.wait(timeout=5.0) # Added timeout for robustness
        if not self.finished_event.is_set():
            logger.warning("👄⚠️ Prewarm finished_event timed out. Forcing stream stop.")
            self.stream.stop() # Force stop if not finished
            self.finished_event.wait(timeout=2.0) # Wait again
        self.finished_event.clear()
        logger.info("👄🔥 Orpheus engine prewarmed.")

        # Measure Time To First Audio (TTFA)
        logger.info("👄⏱️ Measuring Orpheus TTFA...")
        start_time = time.time()
        ttfa = None
        def on_audio_chunk_ttfa(chunk: bytes):
            nonlocal ttfa
            if ttfa is None: # Only capture the first time
                ttfa = time.time() - start_time
                logger.debug(f"👄⏱️ TTFA measurement: First Orpheus audio chunk received. TTFA: {ttfa:.3f}s.")

        self.stream.feed("This is a test sentence to measure the time to first audio chunk.")
        play_kwargs_ttfa = dict(
            on_audio_chunk=on_audio_chunk_ttfa,
            log_synthesized_text=False,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )
        self.stream.play_async(**play_kwargs_ttfa)

        # Wait for TTFA or stream completion
        wait_start_ttfa = time.time()
        max_wait_ttfa = 10.0 # Max seconds to wait for TTFA
        while ttfa is None and (self.stream.is_playing() or not self.finished_event.is_set()):
            if time.time() - wait_start_ttfa > max_wait_ttfa:
                logger.warning(f"👄⚠️ TTFA measurement timed out after {max_wait_ttfa}s.")
                break
            time.sleep(0.01)
        
        if not self.stream.is_playing() and ttfa is None and not self.finished_event.is_set():
            logger.warning("👄⚠️ TTFA measurement: Stream stopped before TTFA could be measured and finished_event not set.")

        self.stream.stop() # Ensure stream stops
        if not self.finished_event.is_set():
            self.finished_event.wait(timeout=2.0) # Wait for stop confirmation
        self.finished_event.clear()

        if ttfa is not None:
            logger.info(f"👄⏱️ Orpheus TTFA measurement complete. TTFA: {ttfa:.3f}s.")
            self.tts_inference_time = ttfa * 1000  # Store as ms
        else:
            logger.warning("👄⚠️ Orpheus TTFA measurement failed (no audio chunk received or timed out). Setting to 0.")
            self.tts_inference_time = 0.0 # Set to a float

        # Callbacks to be set externally if needed
        self.on_first_audio_chunk_synthesize: Optional[Callable[[], None]] = None

    def on_audio_stream_stop(self) -> None:
        """
        Callback executed when the RealtimeTTS audio stream stops processing.

        Logs the event and sets the `finished_event` to signal completion or stop.
        """
        logger.info("👄🛑 Audio stream stopped.")
        self.finished_event.set()

    def synthesize(
            self,
            text: str,
            audio_chunks: Queue, 
            stop_event: threading.Event,
            generation_string: str = "",
        ) -> bool:
        """
        Synthesizes audio from a complete text string using Orpheus and puts chunks into a queue.

        Feeds the entire text string to the Orpheus TTS engine. As audio chunks are generated,
        they are potentially buffered initially for smoother streaming and then put
        into the provided queue. Synthesis can be interrupted via the stop_event.
        Skips initial silent chunks. Triggers the `on_first_audio_chunk_synthesize`
        callback when the first valid audio chunk is queued.

        Args:
            text: The text string to synthesize.
            audio_chunks: The queue to put the resulting audio chunks (bytes) into.
            stop_event: A threading.Event to signal interruption of the synthesis.
            generation_string: An optional identifier string for logging purposes.

        Returns:
            True if synthesis completed fully, False if interrupted by stop_event.
        """
        # Removed Coqui-specific stream chunk size setting
        # if self.engine_name == "coqui" ...

        self.stream.feed(text)
        self.finished_event.clear()

        # Buffering state variables
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = 24000, 2 # Assumed Sample Rate and Bytes Per Sample (16-bit)
        start = time.time()
        self._quick_prev_chunk_time: float = 0.0 # Track time of previous chunk

        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            # Check for interruption signal
            if stop_event.is_set():
                logger.info(f"👄🛑 {generation_string} Quick audio stream interrupted by stop_event. Text: {text[:50]}...")
                # We should not put more chunks, let the main loop handle stream stop
                return

            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR # Duration of the current chunk

            # --- Orpheus specific: Skip initial silence ---
            if on_audio_chunk.first_call and self.engine_name == "orpheus":
                if not hasattr(on_audio_chunk, "silent_chunks_count"):
                    # Initialize silence detection state
                    on_audio_chunk.silent_chunks_count = 0
                    on_audio_chunk.silent_chunks_time = 0.0
                    on_audio_chunk.silence_threshold = 200 # Amplitude threshold for silence

                try:
                    # Analyze chunk for silence
                    fmt = f"{samples}h" # Format for 16-bit signed integers
                    pcm_data = struct.unpack(fmt, chunk)
                    avg_amplitude = np.abs(np.array(pcm_data)).mean()

                    if avg_amplitude < on_audio_chunk.silence_threshold:
                        on_audio_chunk.silent_chunks_count += 1
                        on_audio_chunk.silent_chunks_time += play_duration
                        logger.debug(f"👄⏭️ {generation_string} Quick Skipping silent chunk {on_audio_chunk.silent_chunks_count} (avg_amp: {avg_amplitude:.2f})")
                        return # Skip this chunk
                    elif on_audio_chunk.silent_chunks_count > 0:
                        # First non-silent chunk after silence
                        logger.info(f"👄⏭️ {generation_string} Quick Skipped {on_audio_chunk.silent_chunks_count} silent chunks, saved {on_audio_chunk.silent_chunks_time*1000:.2f}ms")
                        # Proceed to process this non-silent chunk
                except Exception as e:
                    logger.warning(f"👄⚠️ {generation_string} Quick Error analyzing audio chunk for silence: {e}")
                    # Proceed assuming not silent on error

            # --- Timing and Logging ---
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._quick_prev_chunk_time = now
                ttfa_actual = now - start
                logger.info(f"👄🚀 {generation_string} Quick audio start. TTFA: {ttfa_actual:.2f}s. Text: {text[:50]}...")
            else:
                gap = now - self._quick_prev_chunk_time
                self._quick_prev_chunk_time = now
                if gap <= play_duration * 1.1: # Allow small tolerance
                    # logger.debug(f"👄✅ {generation_string} Quick chunk ok (gap={gap:.3f}s ≤ {play_duration:.3f}s). Text: {text[:50]}...")
                    good_streak += 1
                else:
                    logger.warning(f"👄❌ {generation_string} Quick chunk slow (gap={gap:.3f}s > {play_duration:.3f}s). Text: {text[:50]}...")
                    good_streak = 0 # Reset streak on slow chunk

            put_occurred_this_call = False # Track if put happened in this specific call

            # --- Buffering Logic ---
            buffer.append(chunk) # Always append the received chunk first
            buf_dur += play_duration # Update buffer duration

            if buffering:
                # Check conditions to flush buffer and stop buffering
                if good_streak >= 2 or buf_dur >= 0.5: # Flush if stable or buffer > 0.5s
                    logger.info(f"👄➡️ {generation_string} Quick Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s).")
                    for c in buffer:
                        try:
                            audio_chunks.put_nowait(c)
                            put_occurred_this_call = True
                        except asyncio.QueueFull:
                            logger.warning(f"👄⚠️ {generation_string} Quick audio queue full, dropping chunk.")
                    buffer.clear()
                    buf_dur = 0.0 # Reset buffer duration
                    buffering = False # Stop buffering mode
            else: # Not buffering, put chunk directly
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} Quick audio queue full, dropping chunk.")


            # --- First Chunk Callback ---
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f"👄🚀 {generation_string} Quick Firing on_first_audio_chunk_synthesize.")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(f"👄💥 {generation_string} Quick Error in on_first_audio_chunk_synthesize callback: {e}", exc_info=True)
                # Ensure callback fires only once per synthesize call
                on_audio_chunk.callback_fired = True

        # Initialize callback state for this run
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False

        play_kwargs = dict(
            log_synthesized_text=True, # Log the text being synthesized
            on_audio_chunk=on_audio_chunk,
            muted=True, # We handle audio via the queue
            fast_sentence_fragment=False, # Standard processing
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999, # Don't force early fragments
        )

        logger.info(f"👄▶️ {generation_string} Quick Starting synthesis. Text: {text[:50]}...")
        self.stream.play_async(**play_kwargs)

        # Wait loop for completion or interruption
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(f"👄🛑 {generation_string} Quick answer synthesis aborted by stop_event. Text: {text[:50]}...")
                # Drain remaining buffer if any? Decided against it to stop faster.
                buffer.clear()
                # Wait briefly for stop confirmation? The finished_event handles this.
                self.finished_event.wait(timeout=1.0) # Wait for stream stop confirmation
                return False # Indicate interruption
            time.sleep(0.01)

        # # If loop exited normally, check if buffer still has content (stream finished before flush)
        if buffering and buffer and not stop_event.is_set():
            logger.info(f"👄➡️ {generation_string} Quick Flushing remaining buffer after stream finished.")
            for c in buffer:
                 try:
                    audio_chunks.put_nowait(c)
                 except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} Quick audio queue full on final flush, dropping chunk.")
            buffer.clear()

        logger.info(f"👄✅ {generation_string} Quick answer synthesis complete. Text: {text[:50]}...")
        return True # Indicate successful completion

    def synthesize_generator(
            self,
            generator: Generator[str, None, None],
            audio_chunks: Queue, # Should match self.audio_chunks type
            stop_event: threading.Event,
            generation_string: str = "",
        ) -> bool:
        """
        Synthesizes audio from a generator yielding text chunks using Orpheus and puts audio into a queue.

        Feeds text chunks yielded by the generator to the Orpheus TTS engine.
        As audio chunks are generated, they are potentially buffered initially and then
        put into the provided queue. Synthesis can be interrupted via the stop_event.
        Skips initial silent chunks. Sets specific playback parameters for Orpheus.
        Triggers the `on_first_audio_chunk_synthesize` callback when the first
        valid audio chunk is queued.

        Args:
            generator: A generator yielding text chunks (strings) to synthesize.
            audio_chunks: The queue to put the resulting audio chunks (bytes) into.
            stop_event: A threading.Event to signal interruption of the synthesis.
            generation_string: An optional identifier string for logging purposes.

        Returns:
            True if synthesis completed fully, False if interrupted by stop_event.
        """
        # Removed Coqui-specific stream chunk size setting
        # if self.engine_name == "coqui" ...

        self.stream.feed(generator)
        self.finished_event.clear()

        # Buffering state variables
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = 24000, 2 # Assumed Sample Rate and Bytes Per Sample
        start = time.time()
        self._final_prev_chunk_time: float = 0.0 # Separate timer for generator synthesis

        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            if stop_event.is_set():
                logger.info(f"👄🛑 {generation_string} Final audio stream interrupted by stop_event.")
                return

            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR

            # --- Orpheus specific: Skip initial silence ---
            if on_audio_chunk.first_call and self.engine_name == "orpheus":
                if not hasattr(on_audio_chunk, "silent_chunks_count"):
                    on_audio_chunk.silent_chunks_count = 0
                    on_audio_chunk.silent_chunks_time = 0.0
                    # Lower threshold potentially for final answers? Or keep consistent? Using 100 as in original code.
                    on_audio_chunk.silence_threshold = 100

                try:
                    fmt = f"{samples}h"
                    pcm_data = struct.unpack(fmt, chunk)
                    avg_amplitude = np.abs(np.array(pcm_data)).mean()

                    if avg_amplitude < on_audio_chunk.silence_threshold:
                        on_audio_chunk.silent_chunks_count += 1
                        on_audio_chunk.silent_chunks_time += play_duration
                        logger.debug(f"👄⏭️ {generation_string} Final Skipping silent chunk {on_audio_chunk.silent_chunks_count} (avg_amp: {avg_amplitude:.2f})")
                        return # Skip
                    elif on_audio_chunk.silent_chunks_count > 0:
                        logger.info(f"👄⏭️ {generation_string} Final Skipped {on_audio_chunk.silent_chunks_count} silent chunks, saved {on_audio_chunk.silent_chunks_time*1000:.2f}ms")
                except Exception as e:
                    logger.warning(f"👄⚠️ {generation_string} Final Error analyzing audio chunk for silence: {e}")

            # --- Timing and Logging ---
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._final_prev_chunk_time = now
                ttfa_actual = now-start
                logger.info(f"👄🚀 {generation_string} Final audio start. TTFA: {ttfa_actual:.2f}s.")
            else:
                gap = now - self._final_prev_chunk_time
                self._final_prev_chunk_time = now
                if gap <= play_duration * 1.1:
                    # logger.debug(f"👄✅ {generation_string} Final chunk ok (gap={gap:.3f}s ≤ {play_duration:.3f}s).")
                    good_streak += 1
                else:
                    logger.warning(f"👄❌ {generation_string} Final chunk slow (gap={gap:.3f}s > {play_duration:.3f}s).")
                    good_streak = 0

            put_occurred_this_call = False

            # --- Buffering Logic ---
            buffer.append(chunk)
            buf_dur += play_duration
            if buffering:
                if good_streak >= 2 or buf_dur >= 0.5: # Same flush logic as synthesize
                    logger.info(f"👄➡️ {generation_string} Final Flushing buffer (streak={good_streak}, dur={buf_dur:.2f}s).")
                    for c in buffer:
                        try:
                           audio_chunks.put_nowait(c)
                           put_occurred_this_call = True
                        except asyncio.QueueFull:
                            logger.warning(f"👄⚠️ {generation_string} Final audio queue full, dropping chunk.")
                    buffer.clear()
                    buf_dur = 0.0
                    buffering = False
            else: # Not buffering
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} Final audio queue full, dropping chunk.")


            # --- First Chunk Callback --- (Using the same callback as synthesize)
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f"👄🚀 {generation_string} Final Firing on_first_audio_chunk_synthesize.")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(f"👄💥 {generation_string} Final Error in on_first_audio_chunk_synthesize callback: {e}", exc_info=True)
                on_audio_chunk.callback_fired = True

        # Initialize callback state
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False

        play_kwargs = dict(
            log_synthesized_text=True,
            on_audio_chunk=on_audio_chunk,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999, # Default to not forcing early fragments
        )

        # Orpheus specific parameters for generator streaming (engine_name is always "orpheus")
        # These encourage waiting for more text before synthesizing, potentially better for generators
        play_kwargs["minimum_sentence_length"] = 200
        play_kwargs["minimum_first_fragment_length"] = 200
        # Additional Orpheus specific parameters from original code, if applicable to RealtimeTTS OrpheusEngine:
        # play_kwargs["temperature"] = 0.8 # Already set in engine init
        # play_kwargs["top_p"] = 0.95 # Already set in engine init
        # play_kwargs["repetition_penalty"] = 1.1 # Already set in engine init

        logger.info(f"👄▶️ {generation_string} Final Starting Orpheus synthesis from generator.")
        self.stream.play_async(**play_kwargs)

        # Wait loop for completion or interruption
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(f"👄🛑 {generation_string} Final answer synthesis aborted by stop_event.")
                buffer.clear()
                self.finished_event.wait(timeout=1.0) # Wait for stream stop confirmation
                return False # Indicate interruption
            time.sleep(0.01)

        # Flush remaining buffer if stream finished before flush condition met
        if buffering and buffer and not stop_event.is_set():
            logger.info(f"👄➡️ {generation_string} Final Flushing remaining buffer after stream finished.")
            for c in buffer:
                try:
                   audio_chunks.put_nowait(c)
                except asyncio.QueueFull:
                   logger.warning(f"👄⚠️ {generation_string} Final audio queue full on final flush, dropping chunk.")
            buffer.clear()

        logger.info(f"👄✅ {generation_string} Final answer synthesis complete.")
        return True # Indicate successful completion