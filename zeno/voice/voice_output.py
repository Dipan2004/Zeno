"""
ZENO Voice Output Manager - Non-Blocking TTS

Phase 5: Voice Output

Responsibilities:
- Queue TTS requests
- Speak in background thread
- Don't block ZENO execution
- Handle errors gracefully

Architecture:
- Main thread: Queues text for speaking
- TTS thread: Consumes queue and speaks
- Non-blocking design
"""

import logging
import pyttsx3
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class VoiceOutputManager:
    """
    Non-blocking TTS manager.
    
    Design:
    - Lightweight wrapper around pyttsx3
    - Background worker thread
    - Queue-based for non-blocking operation
    - Graceful error handling
    """
    
    def __init__(self, rate: int = 175):
        """
        Initialize TTS engine.
        
        Args:
            rate: Speech rate (words per minute, default: 175)
        """
        self.tts_queue = Queue()
        self._shutdown = threading.Event()
        self._rate = rate
        
        # Start TTS worker thread
        self.tts_thread = threading.Thread(
            target=self._tts_worker,
            daemon=True,
            name="ZENO-TTS"
        )
        self.tts_thread.start()
        
        logger.info(f"VoiceOutputManager initialized (rate={rate})")
    
    def speak(self, text: str, priority: bool = False):
        """
        Queue text for speaking (non-blocking).
        
        Args:
            text: Text to speak
            priority: If True, clear queue and speak immediately
        """
        if not text or not text.strip():
            return
        
        if priority:
            # Clear queue for priority messages
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                except Empty:
                    break
            logger.debug("Cleared TTS queue for priority message")
        
        self.tts_queue.put(text)
        logger.debug(f"Queued TTS: {text[:50]}...")
    
    def _tts_worker(self):
        """
        TTS worker thread - consumes queue and speaks.
        
        Runs continuously until shutdown signal.
        """
        engine = None
        
        try:
            # Initialize engine in worker thread
            logger.info("Initializing TTS engine in worker thread...")
            engine = pyttsx3.init()
            engine.setProperty('rate', self._rate)
            logger.info("TTS engine initialized")
            
            while not self._shutdown.is_set():
                try:
                    # Wait for text to speak (timeout for shutdown check)
                    text = self.tts_queue.get(timeout=0.5)
                    
                    if text:
                        logger.info(f"Speaking: {text}")
                        engine.say(text)
                        engine.runAndWait()
                
                except Empty:
                    # Timeout - loop continues for shutdown check
                    continue
                    
                except Exception as e:
                    logger.error(f"TTS error: {e}", exc_info=True)
                    # Continue on error - don't crash TTS thread
                    # Reinitialize engine on error
                    try:
                        engine.stop()
                        engine = pyttsx3.init()
                        engine.setProperty('rate', self._rate)
                        logger.info("TTS engine reinitialized after error")
                    except:
                        logger.error("Failed to reinitialize TTS engine", exc_info=True)
        
        except Exception as e:
            logger.error(f"TTS worker initialization failed: {e}", exc_info=True)
        
        finally:
            if engine:
                try:
                    engine.stop()
                except:
                    pass
            logger.info("TTS worker shutting down")
    
    def is_speaking(self) -> bool:
        """
        Check if TTS queue has pending messages.
        
        Returns:
            True if queue is not empty
        """
        return not self.tts_queue.empty()
    
    def clear_queue(self):
        """Clear all pending TTS messages"""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except Empty:
                break
        logger.info("TTS queue cleared")
    
    def shutdown(self):
        """
        Clean shutdown of TTS manager.
        
        Signals worker thread to stop and waits for completion.
        """
        logger.info("Shutting down VoiceOutputManager")
        self._shutdown.set()
        
        # Wait for worker thread to finish
        if self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)
            
            if self.tts_thread.is_alive():
                logger.warning("TTS worker thread did not stop cleanly")
        
        logger.info("VoiceOutputManager shutdown complete")


# Convenience function for quick TTS without manager
def speak_once(text: str, rate: int = 175):
    """
    Speak text once (blocking).
    
    For testing or simple use cases.
    Does not use background thread.
    
    Args:
        text: Text to speak
        rate: Speech rate
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"speak_once failed: {e}", exc_info=True)