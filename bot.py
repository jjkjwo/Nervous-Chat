"""
================================================================================
TWITCH BOT WITH AI INTEGRATION + O7 WATCHER + WORD WATCHER + RAFFLE + VISION
================================================================================
A Twitch chatbot that integrates with a local Ollama LLM, can transcribe
live Twitch stream audio using Whisper, analyze images/screenshots with
a vision model, monitors WoW Hardcore streams for o7 spike detection,
and watches for keyword mentions in chat.

COMMANDS:
    - "OF"                : Responds with "OnlyFangs!"
    - "AI HOME"           : Return to home channel
    - "AI GOTO <channel>" : Switch to another channel
    - "AI BANNED"         : List channels the bot is banned from
    - "AI UNBAN <channel>": Remove a channel from the ban list
    - "AI BAN <user>"     : Ban a user from using public commands (e.g., AI SIGNUP)
    - "AI UNBANUSER <user>": Unban a user so they can use public commands again
    - "AI BANLIST"        : List all banned users
    - "AI LISTENTO <ch>"  : Transcribe audio from a channel
    - "AI LOOKAT <src> <prompt>" : Analyze image with AI vision
                            src can be: URL, @streamer, or twitch.tv/streamer
                            Example: AI lookat @cohhcarnage what game is this?
                            Example: AI lookat https://i.imgur.com/x.png count the cats
    - "AI CLIP <channel> [min] [quality] [title] [flags]"
                            Record stream video locally (default 1min)
                            Quality: 1080p, 720p, 480p, best (default: best)
                            Title: [my title] - adds to filename
                            Flags: --compress (-c), --text (-t), --chat
                            Examples:
                              AI clip shroud
                              AI clip shroud 3 720p
                              AI clip shroud 2 [epic death]
                              AI clip shroud 5 720p [boss kill] --compress --text
    - "AI AUDIO <channel> [min]" : Record audio only as .mp3 (default 1min)
                            Example: AI audio shroud 5
    - "AI GIF <channel> [sec]" : Record short GIF (default 10s, max 30s)
                            Example: AI gif shroud 15
    - "AI O7"             : Start monitoring WoW Hardcore streams for o7 spikes
    - "AI STOPO7"         : Stop the o7 watcher
    - "AI O7STATUS"       : Check current o7 activity and thresholds
    - "AI O7SCALE"        : View current threshold scale settings
    - "AI O7SCALE 0.7"    : Set new threshold scale (0.1-10)
    - "AI MUTEO7"         : Mute o7 death announcements (watcher keeps running)
    - "AI UNMUTEO7"       : Re-enable o7 death announcements
    - "AI WORDWATCH"      : Start watching for keywords (uses same HC streams as o7)
    - "AI WORDWATCH <words>": Start watching for specific keywords (comma-separated)
    - "AI STOPWORD"       : Stop the word watcher (o7 keeps running)
    - "AI WORDSTATUS"     : Check word watcher status and pending mentions
    - "AI WORDSET <words>": Change the keywords being watched (comma-separated)
    - "AI WORDCOOLDOWN user <sec>"   : Set user cooldown (default 60s)
    - "AI WORDCOOLDOWN channel <sec>": Set channel cooldown (default 60s)
    - "AI RAFFLE"         : Start a raffle (tracks all chatters)
    - "AI WINNER"         : Pick a random winner from raffle entries
    - "AI RAFFLE STATUS"  : Check how many entries in current raffle
    - "AI CANCEL RAFFLE"  : Cancel raffle without picking winner
    - "AI RECORDCHAT <ch>": Start recording a streamer's chat to a log file
    - "AI STOPCHAT"       : Stop recording chat
    - "AI CHATSTATUS"     : Check chat recording status (channel, messages, file size)
    - "AI SIGNUP <info>"  : Sign up with your class/spec/availability (anyone can use!)
    - "AI ROLL [range]"   : Roll random number (anyone can use!)
                            Default: 1-100
                            AI ROLL 200    → rolls 1-200
                            AI ROLL 20-56  → rolls 20-56
    - "AI <prompt>"       : Send a prompt to the AI

BAN HANDLING:
    - If the bot is banned from a channel, it auto-returns to home
    - Banned channels are tracked and the bot refuses to rejoin them
    - Use "AI UNBAN <channel>" to allow rejoining after being unbanned

USER BANS:
    - Use "AI BAN @username" to ban users from public commands (e.g., AI SIGNUP)
    - Banned users see a "not allowed" message when trying to use public commands
    - Use "AI UNBANUSER @username" to allow users to use commands again
    - Use "AI BANLIST" to see all banned users
    - Ban list persists in banned_users.json

WORD WATCHER:
    - Shares channel connections with o7 watcher (either can start them)
    - Detects when any keyword is mentioned in any monitored stream
    - Supports multiple keywords (comma-separated)
    - User cooldown: Same user can't trigger again for X seconds
    - Channel cooldown: Same channel won't re-announce for X seconds  
    - Accumulates mention count during cooldown periods
    - Fully independent of o7 - can enable/disable either without affecting the other
    - Channels stay open as long as at least one feature is enabled

CHAT RECORDING:
    - Records all chat messages from a streamer's channel to a log file
    - Rolling log with 1MB limit (oldest messages trimmed when limit reached)
    - Log files saved to chatlogs/ directory as <streamer>_chat.log
    - Session markers indicate when recording started/stopped
    - Only one channel can be recorded at a time

RECORDING FEATURES:
    - AI CLIP: Full video with optional quality, compression, text overlay, chat overlay
    - AI AUDIO: Audio-only .mp3 files (much smaller)
    - AI GIF: Short animated GIFs that auto-play everywhere
    - All recordings are live (start recording from NOW, not backdated like Twitch clips)
    - Files saved to clips/, audio/, gifs/ directories
    - Outputs announced as onlyfangs3.com links

RESOURCE MANAGEMENT:
    - Whisper (transcription) and Vision models share a lock
    - Only one heavy model runs at a time to stay under 4GB RAM
    - Commands will queue politely if another is running

================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from twitchio.ext import commands
import aiohttp
import asyncio
import logging
import subprocess
import tempfile
import os
import json
import time
import re
import random
import math
import base64
import gc

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not installed - o7 watcher will be unavailable")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not installed - o7 watcher will be unavailable")


# =============================================================================
# CONFIG
# =============================================================================

CLIENT_ID = 'PLACEHOLDER'
ACCESS_TOKEN = 'PLACEHOLDER'
REFRESH_TOKEN = 'PLACEHOLDER'
EXPIRES_AT = 1766871276.0
TOKEN_FILE = 'twitch_token.json'

HOME_CHANNEL = 'hey_youre_really_cute'
AUTHORIZED_USER = 'hey_youre_really_cute'

O7_CONFIG = {
    "min_threshold": 5,
    "threshold_scale": 0.5,
    "window": 30,
    "cooldown": 60,
    "user_cooldown": 30,
    "stale_timeout": 60,
    "refresh_interval": 300,
    "clip_verify_delay": 30,
    "announce_enabled": False,         # Whether to announce o7 deaths (must use AI O7 to enable)
}

WORD_CONFIG = {
    "keywords": ["ultrafather", "ultrafathertv"],  # Words to watch for (case-insensitive)
    "user_cooldown": 60,              # Seconds before same user can trigger again
    "channel_cooldown": 60,           # Seconds before announcing same channel again
    "enabled": False,                 # Whether word watching is active
}

O7_PATTERN = re.compile(r'\b[oO0]7\b')

SIGNUP_FILE = 'signups.json'
BANNED_CHANNELS_FILE = 'banned_channels.json'
BANNED_USERS_FILE = 'banned_users.json'

# Chat recording config
CHAT_RECORD_CONFIG = {
    "log_dir": "chatlogs",
    "max_file_size": 1024 * 1024,  # 1MB rolling limit
}

# Vision model config - moondream is ~1.6GB, great for 4GB RAM systems
VISION_MODEL = 'moondream'

# Recording config
RECORDING_CONFIG = {
    # Clip settings
    "clip_dir": "clips",
    "clip_base_url": "onlyfangs3.com/clips",
    "clip_default_min": 1,
    "clip_max_min": 10,
    
    # Audio settings
    "audio_dir": "audio",
    "audio_base_url": "onlyfangs3.com/audio",
    "audio_default_min": 1,
    "audio_max_min": 10,
    
    # GIF settings
    "gif_dir": "gifs",
    "gif_base_url": "onlyfangs3.com/gifs",
    "gif_default_sec": 10,
    "gif_max_sec": 30,
    "gif_fps": 15,
    "gif_width": 480,
    
    # Quality presets (streamlink quality string)
    "quality_presets": {
        "1080p": "best,1080p60,1080p,720p60,720p",
        "720p": "720p60,720p,480p,best",
        "480p": "480p,360p,worst",
        "best": "best",
    },
    "default_quality": "best",
    
    # Compression settings (for --compress)
    "compress_crf": 28,        # Higher = smaller file, lower quality (18-28 good range)
    "compress_preset": "fast", # ultrafast, fast, medium, slow
    
    # Text overlay settings
    "text_fontsize": 24,
    "text_fontcolor": "white",
    "text_bordercolor": "black",
    "text_position": "top-left",  # top-left, top-right, bottom-left, bottom-right
    
    # Chat overlay settings
    "chat_fontsize": 18,
    "chat_max_lines": 8,
    "chat_position": "right",  # right side overlay
    "chat_width_percent": 25,  # 25% of video width for chat
}


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    filename='bot.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


# =============================================================================
# TOKEN MANAGER
# =============================================================================

class TokenManager:
    def __init__(self, client_id, refresh_token, token_file, access_token=None, expires_at=0):
        self.client_id = client_id
        self.refresh_token = refresh_token
        self.token_file = token_file
        self.access_token = access_token
        self.expires_at = expires_at
    
    def load_token(self):
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    data = json.load(f)
                    if data.get('expires_at', 0) > time.time() + 1800:
                        self.access_token = data['access_token']
                        self.expires_at = data['expires_at']
                        if 'refresh_token' in data:
                            self.refresh_token = data['refresh_token']
                        logging.info("Loaded valid token from file")
                        return self.access_token
            except Exception as e:
                logging.error(f"Failed to load token file: {e}")
        return None
    
    def save_token(self):
        try:
            with open(self.token_file, 'w') as f:
                json.dump({
                    'access_token': self.access_token,
                    'expires_at': self.expires_at,
                    'refresh_token': self.refresh_token
                }, f)
            logging.info("Token saved to file")
        except Exception as e:
            logging.error(f"Failed to save token: {e}")
    
    def print_token_fix_instructions(self):
        print("\n" + "="*80)
        print("TOKEN EXPIRED OR INVALID - RUN THESE COMMANDS TO FIX:")
        print("="*80)
        print(f"""
STEP 1 - Get device code:
  curl -X POST "https://id.twitch.tv/oauth2/device" \\
    -d "client_id={self.client_id}&scopes=chat:read+chat:edit"

STEP 2 - Go to https://www.twitch.tv/activate and enter the user_code

STEP 3 - After authorizing, run this (replace DEVICE_CODE):
  curl -X POST "https://id.twitch.tv/oauth2/token" \\
    -d "client_id={self.client_id}&device_code=DEVICE_CODE&grant_type=urn:ietf:params:oauth:grant-type:device_code"

STEP 4 - Update token file:
  cat > {self.token_file} << 'EOF'
{{"access_token": "ACCESS_TOKEN", "expires_at": 9999999999.0, "refresh_token": "REFRESH_TOKEN"}}
EOF

STEP 5 - Restart bot
""")
        print("="*80 + "\n")
    
    async def refresh(self):
        logging.info("Refreshing OAuth token...")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://id.twitch.tv/oauth2/token',
                data={
                    'client_id': self.client_id,
                    'grant_type': 'refresh_token',
                    'refresh_token': self.refresh_token
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.access_token = data['access_token']
                    self.expires_at = time.time() + data.get('expires_in', 14400)
                    if 'refresh_token' in data:
                        self.refresh_token = data['refresh_token']
                    self.save_token()
                    logging.info("Token refreshed successfully")
                    return self.access_token
                else:
                    error = await resp.text()
                    self.print_token_fix_instructions()
                    raise Exception(f"Token refresh failed: {error}")
    
    async def get_token(self):
        if self.access_token and self.expires_at > time.time():
            return self.access_token
        self.load_token()
        if self.access_token and self.expires_at > time.time():
            return self.access_token
        await self.refresh()
        return self.access_token


# =============================================================================
# AUDIO TRANSCRIBER
# =============================================================================

class AudioTranscriber:
    def __init__(self, whisper_model="base"):
        self.whisper_model = whisper_model
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            logging.info(f"Loading Whisper {self.whisper_model} model...")
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
            logging.info("Whisper model loaded successfully")
    
    def unload_model(self):
        """Unload model to free memory for other heavy tasks."""
        if self.model is not None:
            logging.info("Unloading Whisper model to free memory...")
            del self.model
            self.model = None
            gc.collect()
            logging.info("Whisper model unloaded")
    
    async def capture_audio(self, channel, duration=20):
        channel = channel.lower().strip()
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_path = temp_file.name
        temp_file.close()
        
        # OPTIMIZED: Added nice, threads limit, and nobuffer for lower CPU usage
        streamlink_cmd = [
            "nice", "-n", "10",
            "streamlink", f"https://twitch.tv/{channel}", 
            "audio_only", "--stdout", "--twitch-low-latency"
        ]
        ffmpeg_cmd = [
            "nice", "-n", "10",
            "ffmpeg",
            "-threads", "1",              # Limit CPU threads
            "-fflags", "nobuffer",        # Don't buffer
            "-i", "pipe:0", 
            "-t", str(duration), 
            "-vn", 
            "-acodec", "pcm_s16le", 
            "-ar", "16000", 
            "-ac", "1", 
            "-f", "wav", 
            "-loglevel", "error",
            "-y", audio_path
        ]
        
        def run_capture():
            streamlink_proc = None
            ffmpeg_proc = None
            try:
                logging.info(f"Capturing {duration}s audio from {channel}...")
                streamlink_proc = subprocess.Popen(streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=streamlink_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                streamlink_proc.stdout.close()
                
                try:
                    ffmpeg_proc.communicate(timeout=duration + 30)
                except subprocess.TimeoutExpired:
                    ffmpeg_proc.kill()
                    ffmpeg_proc.wait()
                    return False, "Audio capture timed out"
                
                try:
                    streamlink_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    streamlink_proc.terminate()
                    try:
                        streamlink_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        streamlink_proc.kill()
                        streamlink_proc.wait()
                
                sl_stderr = streamlink_proc.stderr.read().decode('utf-8', errors='ignore')
                streamlink_proc.stderr.close()
                
                if "No playable streams found" in sl_stderr or "Could not find a stream" in sl_stderr:
                    return False, f"{channel} doesn't appear to be live"
                if "error" in sl_stderr.lower() and "404" in sl_stderr:
                    return False, f"Channel '{channel}' not found"
                if not os.path.exists(audio_path):
                    return False, "Failed to create audio file"
                
                file_size = os.path.getsize(audio_path)
                if file_size < 1000:
                    return False, "Audio file too small - stream may be silent"
                
                logging.info(f"Audio captured successfully: {file_size} bytes")
                return True, ""
            except FileNotFoundError as e:
                if "streamlink" in str(e):
                    return False, "streamlink not installed"
                elif "ffmpeg" in str(e):
                    return False, "ffmpeg not installed"
                return False, f"Missing dependency: {e}"
            except Exception as e:
                logging.error(f"Capture error: {type(e).__name__}: {e}")
                return False, f"Capture failed: {type(e).__name__}"
            finally:
                for proc in [streamlink_proc, ffmpeg_proc]:
                    if proc is not None:
                        try:
                            if proc.poll() is None:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=2)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait()
                        except Exception:
                            pass
        
        try:
            loop = asyncio.get_running_loop()
            success, error_msg = await loop.run_in_executor(None, run_capture)
            if success:
                return audio_path, None
            else:
                self._cleanup(audio_path)
                return None, error_msg
        except Exception as e:
            logging.error(f"Audio capture error: {e}")
            self._cleanup(audio_path)
            return None, f"Audio capture error: {type(e).__name__}"
    
    def _cleanup(self, path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
    
    def _is_hallucination(self, text):
        text_lower = text.lower().strip()
        hallucinations = ["thank you for watching", "thanks for watching", "like and subscribe", "see you in the next video", "please subscribe", "don't forget to subscribe", "hit the bell", "thanks for listening", "see you next time", "bye bye"]
        for phrase in hallucinations:
            if phrase in text_lower:
                return True
        words = text_lower.split()
        if len(words) > 4:
            half = len(words) // 2
            if words[:half] == words[half:half*2]:
                return True
        return False
    
    async def transcribe(self, audio_path):
        def run_transcription():
            self._load_model()
            try:
                segments, info = self.model.transcribe(audio_path, beam_size=1, language="en", condition_on_previous_text=False, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000, speech_pad_ms=400))
                filtered_texts = []
                for seg in segments:
                    if seg.no_speech_prob > 0.9 or seg.avg_logprob < -1.5:
                        continue
                    filtered_texts.append(seg.text.strip())
                text = " ".join(filtered_texts)
                if text and self._is_hallucination(text):
                    return ""
                return text
            except Exception as e:
                logging.error(f"Transcription error: {e}")
                return ""
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, run_transcription)
    
    async def listen_and_transcribe(self, channel, duration=20):
        audio_path, capture_error = await self.capture_audio(channel, duration)
        if capture_error:
            return None, capture_error
        try:
            transcript = await self.transcribe(audio_path)
            if not transcript or not transcript.strip():
                return None, f"No speech detected from {channel}"
            logging.info(f"Transcribed {len(transcript.split())} words from {channel}")
            return transcript, None
        finally:
            self._cleanup(audio_path)


# =============================================================================
# VISION ANALYZER
# =============================================================================

class VisionAnalyzer:
    """Analyze images using Ollama vision models (moondream, llava, etc)."""
    
    def __init__(self, model=VISION_MODEL):
        self.model = model
        self._model_loaded = False
    
    def unload_model(self):
        """Signal to Ollama to unload the model (frees VRAM/RAM)."""
        if self._model_loaded:
            logging.info(f"Requesting Ollama to unload {self.model}...")
            try:
                # Send a request with keep_alive=0 to unload
                import requests as req
                req.post(
                    'http://localhost:11434/api/generate',
                    json={'model': self.model, 'keep_alive': 0},
                    timeout=10
                )
                self._model_loaded = False
                gc.collect()
                logging.info(f"Vision model {self.model} unload requested")
            except Exception as e:
                logging.warning(f"Could not unload vision model: {e}")
    
    async def capture_screenshot(self, channel, timeout=15):
        """
        Capture a single frame from a Twitch stream.
        
        OPTIMIZED for low-memory systems (4GB RAM):
        - Limits ffmpeg to 1 thread to prevent CPU maxing
        - Uses pipe from streamlink to avoid two-step URL fetch
        - Minimizes buffer/analysis time for faster capture
        - Uses nice to lower process priority
        """
        channel = channel.lower().strip()
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        image_path = temp_file.name
        temp_file.close()
        
        def run_capture():
            streamlink_proc = None
            ffmpeg_proc = None
            try:
                logging.info(f"Capturing screenshot from {channel}...")
                
                # Use streamlink to pipe video directly to ffmpeg (single step, less overhead)
                # Use 720p/480p to reduce bandwidth and decoding work
                streamlink_cmd = [
                    "nice", "-n", "10",  # Lower priority to avoid starving other processes
                    "streamlink", 
                    f"https://twitch.tv/{channel}", 
                    "720p,720p60,480p,best,worst",  # Prefer 720p - less data to decode
                    "--stdout",
                    "--twitch-low-latency",
                    "--hls-live-edge", "2",  # Minimal buffering
                ]
                
                # ffmpeg optimized for minimal CPU usage:
                # -threads 1: Use only 1 CPU core (critical for 4GB systems)
                # -fflags nobuffer: Don't buffer input
                # -flags low_delay: Low latency mode
                # -analyzeduration/probesize: Minimize analysis time
                # -frames:v 1: Stop after first frame
                ffmpeg_cmd = [
                    "nice", "-n", "10",
                    "ffmpeg",
                    "-threads", "1",              # CRITICAL: Limit to 1 CPU thread
                    "-fflags", "nobuffer",        # Don't buffer input
                    "-flags", "low_delay",        # Low latency decoding
                    "-analyzeduration", "1000000", # Analyze only 1 second (1M microseconds)
                    "-probesize", "500000",       # Probe only 500KB
                    "-i", "pipe:0",               # Read from stdin (streamlink pipe)
                    "-frames:v", "1",             # Capture exactly 1 frame
                    "-q:v", "3",                  # Quality 3 (good balance)
                    "-y",                         # Overwrite output
                    "-loglevel", "error",         # Reduce log spam
                    image_path
                ]
                
                # Start streamlink, pipe to ffmpeg
                streamlink_proc = subprocess.Popen(
                    streamlink_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd, 
                    stdin=streamlink_proc.stdout, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.PIPE
                )
                
                # Allow streamlink to receive SIGPIPE if ffmpeg exits early
                streamlink_proc.stdout.close()
                
                # Wait for ffmpeg with timeout
                try:
                    _, ffmpeg_stderr = ffmpeg_proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    ffmpeg_proc.kill()
                    ffmpeg_proc.wait()
                    return False, "Screenshot capture timed out"
                
                # Clean up streamlink
                try:
                    streamlink_proc.terminate()
                    streamlink_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    streamlink_proc.kill()
                    streamlink_proc.wait()
                except Exception:
                    pass
                
                # Check stderr for errors
                sl_stderr = streamlink_proc.stderr.read().decode('utf-8', errors='ignore') if streamlink_proc.stderr else ""
                streamlink_proc.stderr.close() if streamlink_proc.stderr else None
                
                # Check for various streamlink error patterns
                sl_lower = sl_stderr.lower()
                if "no playable streams found" in sl_lower or "could not find" in sl_lower:
                    return False, f"{channel} doesn't appear to be live"
                if "404" in sl_stderr or "unable to open url" in sl_lower:
                    return False, f"Channel '{channel}' not found"
                if "unable to find" in sl_lower or "no plugin can handle" in sl_lower:
                    return False, f"Channel '{channel}' not found"
                
                # Verify output exists
                if not os.path.exists(image_path):
                    # Catch-all: if no image and streamlink had any error output, log it
                    if sl_stderr.strip():
                        logging.warning(f"Streamlink stderr for {channel}: {sl_stderr[:200]}")
                    return False, f"Couldn't capture from {channel} - may not exist or be offline"
                
                file_size = os.path.getsize(image_path)
                if file_size < 1000:
                    return False, f"{channel} may be offline or stream unavailable"
                
                logging.info(f"Screenshot captured: {file_size} bytes (low-CPU mode)")
                return True, ""
                
            except FileNotFoundError as e:
                if "streamlink" in str(e):
                    return False, "streamlink not installed"
                elif "ffmpeg" in str(e):
                    return False, "ffmpeg not installed"
                return False, f"Missing dependency: {e}"
            except Exception as e:
                logging.error(f"Screenshot capture error: {type(e).__name__}: {e}")
                return False, f"Capture failed: {type(e).__name__}"
            finally:
                # Ensure cleanup of any hanging processes
                for proc in [streamlink_proc, ffmpeg_proc]:
                    if proc is not None:
                        try:
                            if proc.poll() is None:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=2)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait()
                        except Exception:
                            pass
        
        try:
            loop = asyncio.get_running_loop()
            success, error_msg = await loop.run_in_executor(None, run_capture)
            if success:
                return image_path, None
            else:
                self._cleanup(image_path)
                return None, error_msg
        except Exception as e:
            logging.error(f"Screenshot error: {e}")
            self._cleanup(image_path)
            return None, f"Screenshot error: {type(e).__name__}"
    
    async def download_image(self, url, timeout=15):
        """Download an image from a URL."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        image_path = temp_file.name
        temp_file.close()
        
        try:
            logging.info(f"Downloading image from {url[:50]}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status != 200:
                        return None, f"Failed to download image (HTTP {resp.status})"
                    
                    content_type = resp.headers.get('content-type', '')
                    if not any(t in content_type for t in ['image', 'octet-stream']):
                        return None, f"URL doesn't appear to be an image"
                    
                    data = await resp.read()
                    
                    if len(data) < 1000:
                        return None, "Downloaded file too small"
                    if len(data) > 20 * 1024 * 1024:  # 20MB limit
                        return None, "Image too large (max 20MB)"
                    
                    with open(image_path, 'wb') as f:
                        f.write(data)
                    
                    logging.info(f"Image downloaded: {len(data)} bytes")
                    return image_path, None
                    
        except asyncio.TimeoutError:
            self._cleanup(image_path)
            return None, "Image download timed out"
        except Exception as e:
            self._cleanup(image_path)
            logging.error(f"Image download error: {e}")
            return None, f"Download failed: {type(e).__name__}"
    
    def _cleanup(self, path):
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except:
            pass
    
    def _encode_image(self, image_path):
        """Encode image to base64 for Ollama API."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    async def analyze(self, image_path, prompt, max_retries=2):
        """Send image + prompt to Ollama vision model."""
        logging.info(f"Analyzing image with prompt: {prompt[:50]}...")
        
        try:
            image_b64 = self._encode_image(image_path)
        except Exception as e:
            return f"Failed to read image: {e}"
        
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=120)  # Vision can be slow
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    payload = {
                        'model': self.model,
                        'prompt': prompt,
                        'images': [image_b64],
                        'stream': False,
                        'keep_alive': '10m',  # Keep loaded for 10 min
                        'options': {
                            'temperature': 0.7,
                            'num_predict': 200
                        }
                    }
                    
                    async with session.post(
                        'http://localhost:11434/api/generate',
                        json=payload
                    ) as resp:
                        if resp.status == 200:
                            self._model_loaded = True
                            data = await resp.json()
                            response = data.get('response', '').strip()
                            if response:
                                return response
                            else:
                                last_error = "Vision model returned empty response"
                        else:
                            error_text = await resp.text()
                            if "model" in error_text.lower() and "not found" in error_text.lower():
                                last_error = f"Model '{self.model}' not found. Run: ollama pull {self.model}"
                            else:
                                last_error = f"Vision error (status {resp.status})"
                                
            except aiohttp.ClientConnectorError:
                last_error = "Can't connect to Ollama - is it running?"
            except asyncio.TimeoutError:
                last_error = "Vision analysis timed out (model may be loading)"
            except Exception as e:
                last_error = f"Error: {type(e).__name__}"
            
            if attempt < max_retries:
                await asyncio.sleep(2)
        
        return last_error or "Vision unavailable"
    
    async def analyze_from_source(self, source, prompt):
        """
        Analyze image from various sources.
        
        source can be:
        - URL (http:// or https://)
        - Twitch channel (@name, twitch.tv/name, or just name)
        """
        image_path = None
        cleanup_needed = True
        
        try:
            # Determine source type and get image
            if source.startswith('http://') or source.startswith('https://'):
                # Direct image URL
                if 'twitch.tv/' in source:
                    # It's a Twitch URL, extract channel
                    channel = source.split('twitch.tv/')[-1].split('/')[0].split('?')[0]
                    image_path, error = await self.capture_screenshot(channel)
                else:
                    # Regular image URL
                    image_path, error = await self.download_image(source)
            else:
                # Assume it's a Twitch channel name
                channel = source.lstrip('@').lower()
                image_path, error = await self.capture_screenshot(channel)
            
            if error:
                return None, error
            
            # Analyze the image
            result = await self.analyze(image_path, prompt)
            return result, None
            
        finally:
            if cleanup_needed and image_path:
                self._cleanup(image_path)


# =============================================================================
# CLIP RECORDER (Video, Audio, GIF recording with overlays)
# =============================================================================

class ClipRecorder:
    """Records video clips, audio, and GIFs from Twitch streams with optional overlays."""
    
    def __init__(self):
        self.chat_buffer = {}  # {channel: [(timestamp, user, message), ...]}
        self.chat_capture_active = {}  # {channel: start_time}
        self._chat_lock = asyncio.Lock()
    
    def _sanitize_title(self, title):
        """Sanitize title for use in filename."""
        if not title:
            return ""
        # Remove/replace invalid filename characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
        sanitized = sanitized.strip().replace(' ', '-').lower()
        sanitized = re.sub(r'-+', '-', sanitized)  # Collapse multiple dashes
        return sanitized[:50]  # Limit length
    
    def _parse_clip_args(self, args):
        """
        Parse clip command arguments.
        Returns: (channel, minutes, quality, title, flags)
        
        Examples:
            "shroud" -> (shroud, 1, best, None, {})
            "shroud 3" -> (shroud, 3, best, None, {})
            "shroud 3 720p" -> (shroud, 3, 720p, None, {})
            "shroud [epic death]" -> (shroud, 1, best, "epic death", {})
            "shroud 3 720p [boss kill] --compress --text" -> (shroud, 3, 720p, "boss kill", {compress, text})
        """
        if not args:
            return None, None, None, None, {}
        
        # Extract title from brackets first
        title = None
        title_match = re.search(r'\[([^\]]+)\]', args)
        if title_match:
            title = title_match.group(1)
            args = args[:title_match.start()] + args[title_match.end():]
        
        # Extract flags
        flags = set()
        flag_patterns = [
            (r'--compress\b|-c\b', 'compress'),
            (r'--text\b|-t\b', 'text'),
            (r'--chat\b', 'chat'),
        ]
        for pattern, flag_name in flag_patterns:
            if re.search(pattern, args):
                flags.add(flag_name)
                args = re.sub(pattern, '', args)
        
        # Parse remaining: channel [minutes] [quality]
        parts = args.strip().split()
        if not parts:
            return None, None, None, None, {}
        
        channel = parts[0].lower().lstrip('@').replace('twitch.tv/', '').replace('/', '')
        
        minutes = RECORDING_CONFIG["clip_default_min"]
        quality = RECORDING_CONFIG["default_quality"]
        
        for part in parts[1:]:
            # Check if it's a number (minutes)
            try:
                mins = int(part)
                if 1 <= mins <= RECORDING_CONFIG["clip_max_min"]:
                    minutes = mins
                continue
            except ValueError:
                pass
            
            # Check if it's a quality preset
            if part.lower() in RECORDING_CONFIG["quality_presets"]:
                quality = part.lower()
        
        return channel, minutes, quality, title, flags
    
    def _parse_audio_args(self, args):
        """Parse audio command arguments. Returns: (channel, minutes)"""
        if not args:
            return None, None
        
        parts = args.strip().split()
        if not parts:
            return None, None
        
        channel = parts[0].lower().lstrip('@').replace('twitch.tv/', '').replace('/', '')
        minutes = RECORDING_CONFIG["audio_default_min"]
        
        if len(parts) >= 2:
            try:
                mins = int(parts[1])
                if 1 <= mins <= RECORDING_CONFIG["audio_max_min"]:
                    minutes = mins
            except ValueError:
                pass
        
        return channel, minutes
    
    def _parse_gif_args(self, args):
        """Parse GIF command arguments. Returns: (channel, seconds)"""
        if not args:
            return None, None
        
        parts = args.strip().split()
        if not parts:
            return None, None
        
        channel = parts[0].lower().lstrip('@').replace('twitch.tv/', '').replace('/', '')
        seconds = RECORDING_CONFIG["gif_default_sec"]
        
        if len(parts) >= 2:
            try:
                secs = int(parts[1])
                if 1 <= secs <= RECORDING_CONFIG["gif_max_sec"]:
                    seconds = secs
            except ValueError:
                pass
        
        return channel, seconds
    
    async def capture_chat_websocket(self, channel, duration):
        """
        Connect to Twitch IRC and capture chat messages for specified duration.
        Returns list of (relative_time, user, message) tuples.
        """
        if not WEBSOCKETS_AVAILABLE:
            logging.warning("websockets not available for chat capture")
            return []
        
        messages = []
        start_time = time.time()
        channel_lower = channel.lower()
        
        uri = "wss://irc-ws.chat.twitch.tv:443"
        nick = f"justinfan{random.randint(10000, 99999)}"
        
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(f"NICK {nick}")
                await ws.send(f"JOIN #{channel_lower}")
                logging.info(f"Chat capture: Connected to #{channel_lower}")
                
                while (time.time() - start_time) < duration:
                    remaining = duration - (time.time() - start_time)
                    if remaining <= 0:
                        break
                    
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(ws.recv(), timeout=min(5.0, remaining))
                    except asyncio.TimeoutError:
                        continue
                    
                    for line in message.split('\r\n'):
                        if line.startswith('PING'):
                            await ws.send('PONG :tmi.twitch.tv')
                        elif 'PRIVMSG' in line:
                            try:
                                user = line.split('!')[0][1:]
                                msg = line.split('PRIVMSG', 1)[1].split(':', 1)[1]
                                relative_time = time.time() - start_time
                                messages.append((relative_time, user, msg))
                                logging.debug(f"Chat capture: {user}: {msg[:50]}")
                            except Exception:
                                pass
                
                logging.info(f"Chat capture: Finished, got {len(messages)} messages")
                
        except asyncio.CancelledError:
            logging.info(f"Chat capture cancelled for #{channel_lower}")
        except Exception as e:
            logging.error(f"Chat capture error for #{channel_lower}: {type(e).__name__}: {e}")
        
        return messages
    
    async def start_chat_capture(self, channel):
        """Start capturing chat for a channel."""
        async with self._chat_lock:
            channel_lower = channel.lower()
            self.chat_buffer[channel_lower] = []
            self.chat_capture_active[channel_lower] = time.time()
            logging.info(f"Started chat capture for #{channel}")
    
    async def add_chat_message(self, channel, user, message):
        """Add a chat message to the buffer."""
        async with self._chat_lock:
            channel_lower = channel.lower()
            if channel_lower in self.chat_capture_active:
                start_time = self.chat_capture_active[channel_lower]
                relative_time = time.time() - start_time
                self.chat_buffer[channel_lower].append((relative_time, user, message))
    
    async def stop_chat_capture(self, channel):
        """Stop capturing chat and return captured messages."""
        async with self._chat_lock:
            channel_lower = channel.lower()
            messages = self.chat_buffer.pop(channel_lower, [])
            self.chat_capture_active.pop(channel_lower, None)
            logging.info(f"Stopped chat capture for #{channel}, got {len(messages)} messages")
            return messages
    
    def _generate_chat_ass(self, messages, duration, video_width, video_height):
        """Generate ASS subtitle file for chat overlay."""
        chat_width = int(video_width * RECORDING_CONFIG["chat_width_percent"] / 100)
        x_pos = video_width - chat_width - 10
        y_start = 50
        line_height = RECORDING_CONFIG["chat_fontsize"] + 4
        max_lines = RECORDING_CONFIG["chat_max_lines"]
        
        # ASS header
        ass_content = f"""[Script Info]
Title: Chat Overlay
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Chat,Arial,{RECORDING_CONFIG["chat_fontsize"]},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,7,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        # Group messages into time windows for display
        display_duration = 5.0  # Each message visible for 5 seconds
        
        for msg_time, user, message in messages:
            if msg_time > duration:
                break
            
            start_time = msg_time
            end_time = min(msg_time + display_duration, duration)
            
            # Format times as H:MM:SS.cc
            def format_time(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = t % 60
                return f"{h}:{m:02d}:{s:05.2f}"
            
            # Escape special characters and truncate
            safe_user = user[:15].replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
            safe_msg = message[:80].replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
            text = f"{{\\c&H00FF00&}}{safe_user}{{\\c&HFFFFFF&}}: {safe_msg}"
            
            ass_content += f"Dialogue: 0,{format_time(start_time)},{format_time(end_time)},Chat,,0,0,0,,{text}\n"
        
        return ass_content
    
    def _build_text_overlay_filter(self, channel, quality):
        """Build ffmpeg filter for text overlay."""
        cfg = RECORDING_CONFIG
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        text = f"{channel} | {quality} | {timestamp}"
        
        # Escape special characters for ffmpeg
        text = text.replace("'", "\\'").replace(":", "\\:")
        
        # Position mapping
        positions = {
            "top-left": "x=10:y=10",
            "top-right": "x=w-tw-10:y=10",
            "bottom-left": "x=10:y=h-th-10",
            "bottom-right": "x=w-tw-10:y=h-th-10",
        }
        pos = positions.get(cfg["text_position"], "x=10:y=10")
        
        return (
            f"drawtext=text='{text}':"
            f"fontsize={cfg['text_fontsize']}:"
            f"fontcolor={cfg['text_fontcolor']}:"
            f"borderw=2:bordercolor={cfg['text_bordercolor']}:"
            f"{pos}"
        )
    
    async def record_clip(self, channel, minutes, quality, title=None, flags=None):
        """
        Record a video clip from a Twitch stream (CPU-optimized).
        
        Returns: (success, result_or_error, filename)
        """
        flags = flags or set()
        cfg = RECORDING_CONFIG
        
        # Create output directory
        os.makedirs(cfg["clip_dir"], exist_ok=True)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        title_part = f"_{self._sanitize_title(title)}" if title else ""
        filename = f"clip_{channel}_{timestamp}{title_part}.mp4"
        output_path = os.path.join(cfg["clip_dir"], filename)
        
        duration = minutes * 60
        quality_str = cfg["quality_presets"].get(quality, quality)
        
        def run_recording():
            streamlink_proc = None
            ffmpeg_proc = None
            temp_output = output_path + ".tmp.mp4"
            
            try:
                logging.info(f"Recording {minutes}min clip from {channel} ({quality})...")
                
                # Get stream (with nice for lower priority)
                streamlink_cmd = [
                    "nice", "-n", "10",
                    "streamlink", f"https://twitch.tv/{channel}",
                    quality_str, "--stdout", "--twitch-low-latency"
                ]
                
                # Determine if we need to re-encode
                # If chat overlay is requested, we'll do ALL overlays in post-processing
                # to avoid encoding twice
                needs_reencode = ('compress' in flags or 'text' in flags) and 'chat' not in flags
                
                # Build ffmpeg command (with CPU limits for 4GB systems)
                ffmpeg_cmd = [
                    "nice", "-n", "10",
                    "ffmpeg", 
                    "-threads", "2" if needs_reencode else "1",  # Limit threads
                    "-i", "pipe:0", 
                    "-t", str(duration)
                ]
                
                if needs_reencode:
                    # Build filter chain
                    filters = []
                    
                    if 'text' in flags:
                        filters.append(self._build_text_overlay_filter(channel, quality))
                    
                    if filters:
                        ffmpeg_cmd.extend(["-vf", ",".join(filters)])
                    
                    if 'compress' in flags:
                        ffmpeg_cmd.extend([
                            "-c:v", "libx264",
                            "-crf", str(cfg["compress_crf"]),
                            "-preset", cfg["compress_preset"],
                            "-c:a", "aac", "-b:a", "128k"
                        ])
                    else:
                        ffmpeg_cmd.extend(["-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "copy"])
                else:
                    ffmpeg_cmd.extend(["-c", "copy"])
                
                ffmpeg_cmd.extend(["-loglevel", "error", "-y", temp_output])
                
                streamlink_proc = subprocess.Popen(
                    streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd, stdin=streamlink_proc.stdout,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
                streamlink_proc.stdout.close()
                
                try:
                    # Extra time for re-encoding
                    timeout = duration + (120 if needs_reencode else 60)
                    ffmpeg_proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    ffmpeg_proc.kill()
                    ffmpeg_proc.wait()
                    return False, "Recording timed out", None
                
                try:
                    streamlink_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    streamlink_proc.terminate()
                    try:
                        streamlink_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        streamlink_proc.kill()
                        streamlink_proc.wait()
                
                sl_stderr = streamlink_proc.stderr.read().decode('utf-8', errors='ignore')
                streamlink_proc.stderr.close()
                
                if "No playable streams" in sl_stderr or "Could not find" in sl_stderr:
                    return False, f"{channel} not live", None
                if "error" in sl_stderr.lower() and "404" in sl_stderr:
                    return False, f"Channel '{channel}' not found", None
                
                if not os.path.exists(temp_output):
                    return False, "Failed to create clip", None
                
                file_size = os.path.getsize(temp_output)
                if file_size < 10000:
                    return False, "Clip too small - stream may have ended", None
                
                # Move temp to final (will add chat overlay after if needed)
                os.rename(temp_output, output_path)
                
                size_mb = file_size / (1024 * 1024)
                return True, f"{size_mb:.1f}MB", filename
                
            except FileNotFoundError as e:
                if "streamlink" in str(e):
                    return False, "streamlink not installed", None
                elif "ffmpeg" in str(e):
                    return False, "ffmpeg not installed", None
                return False, f"Missing dependency: {e}", None
            except Exception as e:
                logging.error(f"Clip recording error: {type(e).__name__}: {e}")
                return False, str(e), None
            finally:
                for proc in [streamlink_proc, ffmpeg_proc]:
                    if proc is not None:
                        try:
                            if proc.poll() is None:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=2)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait()
                        except Exception:
                            pass
                # Cleanup temp file
                if os.path.exists(temp_output):
                    try:
                        os.remove(temp_output)
                    except:
                        pass
        
        # Run recording and chat capture concurrently if chat overlay requested
        loop = asyncio.get_running_loop()
        
        if 'chat' in flags:
            # Run both video recording and chat capture at the same time
            logging.info(f"Starting concurrent video recording and chat capture for #{channel}")
            
            # Create tasks for both
            recording_task = loop.run_in_executor(None, run_recording)
            chat_task = self.capture_chat_websocket(channel, duration)
            
            # Wait for both to complete
            (success, result, filename), chat_messages = await asyncio.gather(
                recording_task, chat_task
            )
            
            logging.info(f"Recording done: {success}, Chat captured: {len(chat_messages)} messages")
            
            # Apply overlays if recording succeeded (chat + text + compress all in one pass)
            if success:
                success, result = await self._apply_overlays(
                    output_path, chat_messages, duration, channel, quality, flags
                )
        else:
            # No chat - just record video
            success, result, filename = await loop.run_in_executor(None, run_recording)
        
        # Cleanup on failure
        if not success and filename and os.path.exists(os.path.join(cfg["clip_dir"], filename)):
            try:
                os.remove(os.path.join(cfg["clip_dir"], filename))
            except:
                pass
        
        return success, result, filename
    
    async def _apply_overlays(self, video_path, chat_messages, duration, channel, quality, flags):
        """Apply all overlays (chat, text, compress) in a single encoding pass."""
        cfg = RECORDING_CONFIG
        
        logging.info(f"Applying overlays to {video_path}: chat={len(chat_messages) if chat_messages else 0} msgs, text={'text' in flags}, compress={'compress' in flags}")
        
        def run_overlay():
            ass_path = None
            temp_output = None
            try:
                # Get video dimensions
                probe_cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0",
                    video_path
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    logging.error(f"ffprobe failed: {result.stderr}")
                    file_size = os.path.getsize(video_path) / (1024 * 1024)
                    return True, f"{file_size:.1f}MB (overlays skipped)"
                
                dims = result.stdout.strip().split(',')
                if len(dims) != 2:
                    logging.error(f"Could not parse dimensions: {result.stdout}")
                    file_size = os.path.getsize(video_path) / (1024 * 1024)
                    return True, f"{file_size:.1f}MB (overlays skipped)"
                
                width, height = int(dims[0]), int(dims[1])
                logging.info(f"Video dimensions: {width}x{height}")
                
                # Build filter chain
                filters = []
                
                # Add text overlay if requested
                if 'text' in flags:
                    text_filter = self._build_text_overlay_filter(channel, quality)
                    filters.append(text_filter)
                
                # Add chat overlay if we have messages
                if chat_messages:
                    ass_content = self._generate_chat_ass(chat_messages, duration, width, height)
                    ass_path = video_path + ".chat.ass"
                    with open(ass_path, 'w', encoding='utf-8') as f:
                        f.write(ass_content)
                    logging.info(f"ASS file written with {len(chat_messages)} messages")
                    
                    # Escape for ffmpeg filter
                    ass_path_for_filter = ass_path.replace('\\', '/').replace(':', '\\:').replace("'", "'\\''")
                    filters.append(f"ass={ass_path_for_filter}")
                
                # Build ffmpeg command
                temp_output = video_path + ".overlay.mp4"
                ffmpeg_cmd = ["ffmpeg", "-i", video_path]
                
                if filters:
                    ffmpeg_cmd.extend(["-vf", ",".join(filters)])
                
                # Encoding settings
                if 'compress' in flags:
                    ffmpeg_cmd.extend([
                        "-c:v", "libx264",
                        "-crf", str(cfg["compress_crf"]),
                        "-preset", "fast",
                        "-c:a", "aac", "-b:a", "128k"
                    ])
                else:
                    # Use ultrafast for speed when just adding overlays
                    ffmpeg_cmd.extend([
                        "-c:v", "libx264",
                        "-crf", "23",
                        "-preset", "ultrafast",
                        "-c:a", "copy"
                    ])
                
                ffmpeg_cmd.extend(["-y", temp_output])
                
                logging.info(f"Running ffmpeg with {len(filters)} filter(s)...")
                proc_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
                
                if proc_result.returncode != 0:
                    logging.error(f"ffmpeg overlay failed (code {proc_result.returncode}): {proc_result.stderr[:500]}")
                    file_size = os.path.getsize(video_path) / (1024 * 1024)
                    return True, f"{file_size:.1f}MB (overlays failed)"
                
                # Replace original with overlay version
                if os.path.exists(temp_output) and os.path.getsize(temp_output) > 10000:
                    os.remove(video_path)
                    os.rename(temp_output, video_path)
                    new_size = os.path.getsize(video_path) / (1024 * 1024)
                    logging.info(f"Overlays applied successfully, new size: {new_size:.1f}MB")
                    return True, f"{new_size:.1f}MB"
                
                logging.warning("Overlay output file missing or too small")
                file_size = os.path.getsize(video_path) / (1024 * 1024)
                return True, f"{file_size:.1f}MB (overlays failed)"
                
            except subprocess.TimeoutExpired:
                logging.error("ffmpeg overlay timed out")
                file_size = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0
                return True, f"{file_size:.1f}MB (overlays timed out)"
            except Exception as e:
                logging.error(f"Overlay error: {type(e).__name__}: {e}")
                file_size = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0
                return True, f"{file_size:.1f}MB (overlays failed)"
            finally:
                # Cleanup temp files
                if ass_path and os.path.exists(ass_path):
                    try:
                        os.remove(ass_path)
                    except:
                        pass
                if temp_output and os.path.exists(temp_output):
                    try:
                        os.remove(temp_output)
                    except:
                        pass
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, run_overlay)
    
    async def record_audio(self, channel, minutes):
        """Record audio only from a Twitch stream (CPU-optimized)."""
        cfg = RECORDING_CONFIG
        
        # Create output directory
        os.makedirs(cfg["audio_dir"], exist_ok=True)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{channel}_{timestamp}.mp3"
        output_path = os.path.join(cfg["audio_dir"], filename)
        
        duration = minutes * 60
        
        def run_recording():
            streamlink_proc = None
            ffmpeg_proc = None
            
            try:
                logging.info(f"Recording {minutes}min audio from {channel}...")
                
                # CPU-optimized: nice and low-latency
                streamlink_cmd = [
                    "nice", "-n", "10",
                    "streamlink", f"https://twitch.tv/{channel}",
                    "audio_only", "--stdout", "--twitch-low-latency"
                ]
                
                ffmpeg_cmd = [
                    "nice", "-n", "10",
                    "ffmpeg",
                    "-threads", "1",       # Limit CPU threads
                    "-fflags", "nobuffer", # Don't buffer input
                    "-i", "pipe:0",
                    "-t", str(duration),
                    "-vn",  # No video
                    "-acodec", "libmp3lame",
                    "-ab", "192k",
                    "-loglevel", "error",
                    "-y", output_path
                ]
                
                streamlink_proc = subprocess.Popen(
                    streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd, stdin=streamlink_proc.stdout,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
                streamlink_proc.stdout.close()
                
                try:
                    ffmpeg_proc.communicate(timeout=duration + 60)
                except subprocess.TimeoutExpired:
                    ffmpeg_proc.kill()
                    ffmpeg_proc.wait()
                    return False, "Recording timed out", None
                
                try:
                    streamlink_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    streamlink_proc.terminate()
                    try:
                        streamlink_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        streamlink_proc.kill()
                        streamlink_proc.wait()
                
                sl_stderr = streamlink_proc.stderr.read().decode('utf-8', errors='ignore')
                streamlink_proc.stderr.close()
                
                if "No playable streams" in sl_stderr or "Could not find" in sl_stderr:
                    return False, f"{channel} not live", None
                
                if not os.path.exists(output_path):
                    return False, "Failed to create audio", None
                
                file_size = os.path.getsize(output_path)
                if file_size < 5000:
                    return False, "Audio too small - stream may be silent", None
                
                size_mb = file_size / (1024 * 1024)
                return True, f"{size_mb:.1f}MB", filename
                
            except FileNotFoundError as e:
                return False, "Missing streamlink or ffmpeg", None
            except Exception as e:
                logging.error(f"Audio recording error: {type(e).__name__}: {e}")
                return False, str(e), None
            finally:
                for proc in [streamlink_proc, ffmpeg_proc]:
                    if proc is not None:
                        try:
                            if proc.poll() is None:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=2)
                                except subprocess.TimeoutExpired:
                                    proc.kill()
                                    proc.wait()
                        except Exception:
                            pass
        
        loop = asyncio.get_running_loop()
        success, result, filename = await loop.run_in_executor(None, run_recording)
        
        # Cleanup on failure
        if not success and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        
        return success, result, filename
    
    async def record_gif(self, channel, seconds):
        """Record a GIF from a Twitch stream (CPU-optimized)."""
        cfg = RECORDING_CONFIG
        
        # Create output directory
        os.makedirs(cfg["gif_dir"], exist_ok=True)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"gif_{channel}_{timestamp}.gif"
        output_path = os.path.join(cfg["gif_dir"], filename)
        
        def run_recording():
            streamlink_proc = None
            ffmpeg_proc = None
            palette_path = output_path + ".palette.png"
            temp_video = output_path + ".tmp.mp4"
            
            try:
                logging.info(f"Recording {seconds}s GIF from {channel}...")
                
                # First, capture raw video (stream copy = minimal CPU)
                streamlink_cmd = [
                    "nice", "-n", "10",
                    "streamlink", f"https://twitch.tv/{channel}",
                    "720p,best,480p,worst", "--stdout", "--twitch-low-latency"
                ]
                
                ffmpeg_cmd = [
                    "nice", "-n", "10",
                    "ffmpeg", 
                    "-threads", "1",
                    "-i", "pipe:0",
                    "-t", str(seconds),
                    "-c", "copy",
                    "-loglevel", "error",
                    "-y", temp_video
                ]
                
                streamlink_proc = subprocess.Popen(
                    streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd, stdin=streamlink_proc.stdout,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )
                streamlink_proc.stdout.close()
                
                try:
                    ffmpeg_proc.communicate(timeout=seconds + 30)
                except subprocess.TimeoutExpired:
                    ffmpeg_proc.kill()
                    ffmpeg_proc.wait()
                    return False, "Recording timed out", None
                
                try:
                    streamlink_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    streamlink_proc.terminate()
                    streamlink_proc.wait(timeout=5)
                
                sl_stderr = streamlink_proc.stderr.read().decode('utf-8', errors='ignore')
                streamlink_proc.stderr.close()
                
                if "No playable streams" in sl_stderr:
                    return False, f"{channel} not live", None
                
                if not os.path.exists(temp_video) or os.path.getsize(temp_video) < 10000:
                    return False, "Failed to capture video", None
                
                # Generate palette for high-quality GIF (CPU limited)
                fps = cfg["gif_fps"]
                width = cfg["gif_width"]
                
                palette_cmd = [
                    "nice", "-n", "15",  # Even lower priority for encoding
                    "ffmpeg", 
                    "-threads", "2",    # Allow 2 threads for palette gen
                    "-i", temp_video,
                    "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff",
                    "-loglevel", "error",
                    "-y", palette_path
                ]
                subprocess.run(palette_cmd, capture_output=True, timeout=60)
                
                # Create GIF with palette (CPU limited)
                gif_cmd = [
                    "nice", "-n", "15",
                    "ffmpeg", 
                    "-threads", "2",    # Allow 2 threads for GIF creation
                    "-i", temp_video, "-i", palette_path,
                    "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-loglevel", "error",
                    "-y", output_path
                ]
                subprocess.run(gif_cmd, capture_output=True, timeout=120)
                
                if not os.path.exists(output_path):
                    return False, "Failed to create GIF", None
                
                file_size = os.path.getsize(output_path)
                if file_size < 5000:
                    return False, "GIF too small", None
                
                size_mb = file_size / (1024 * 1024)
                return True, f"{size_mb:.1f}MB", filename
                
            except FileNotFoundError:
                return False, "Missing streamlink or ffmpeg", None
            except Exception as e:
                logging.error(f"GIF recording error: {type(e).__name__}: {e}")
                return False, str(e), None
            finally:
                for proc in [streamlink_proc, ffmpeg_proc]:
                    if proc is not None:
                        try:
                            if proc.poll() is None:
                                proc.terminate()
                                proc.wait(timeout=2)
                        except:
                            pass
                # Cleanup temp files
                for temp in [palette_path, temp_video]:
                    try:
                        if os.path.exists(temp):
                            os.remove(temp)
                    except:
                        pass
        
        loop = asyncio.get_running_loop()
        success, result, filename = await loop.run_in_executor(None, run_recording)
        
        # Cleanup on failure
        if not success and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        
        return success, result, filename


# =============================================================================
# O7 TRACKER
# =============================================================================

class O7Tracker:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.recent_o7s = {}
        self.user_last_o7 = {}
        self.streamer_names = {}
        self.streamer_titles = {}
        self.streamer_viewers = {}
        self.active_alert = False
        self.active_alert_streamers = set()
        self.active_alert_count = 0
        self.active_alert_peak = 0
        self.active_alert_last_o7 = 0
        self.last_alert_time = 0
        self.last_announcement_time = 0
        self.total_o7s_session = 0
    
    def get_threshold_for_channel(self, login):
        viewers = self.streamer_viewers.get(login.lower(), 0)
        if viewers <= 0:
            return O7_CONFIG["min_threshold"]
        dynamic_threshold = int(math.sqrt(viewers) * O7_CONFIG["threshold_scale"])
        return max(O7_CONFIG["min_threshold"], dynamic_threshold)
    
    def set_streamer_viewers(self, login, viewers):
        self.streamer_viewers[login.lower()] = viewers
    
    def is_user_on_cooldown(self, streamer, user):
        key = (streamer.lower(), user.lower())
        now = time.time()
        last = self.user_last_o7.get(key, 0)
        return (now - last) < O7_CONFIG["user_cooldown"]
    
    async def add_o7(self, streamer, display_name, user):
        if self.is_user_on_cooldown(streamer, user):
            return None
        now = time.time()
        key = (streamer.lower(), user.lower())
        async with self.lock:
            self.user_last_o7[key] = now
            s = streamer.lower()
            self.streamer_names[s] = display_name
            if s not in self.recent_o7s:
                self.recent_o7s[s] = []
            self.recent_o7s[s].append(now)
            self.recent_o7s[s] = [t for t in self.recent_o7s[s] if now - t < O7_CONFIG["window"]]
            self.total_o7s_session += 1
            return len(self.recent_o7s[s])
    
    def set_streamer_title(self, login, title):
        self.streamer_titles[login.lower()] = title
    
    async def check_for_spike(self):
        now = time.time()
        async with self.lock:
            if self.active_alert and (now - self.active_alert_last_o7) > O7_CONFIG["stale_timeout"]:
                self._clear_active_alert()
            
            active_streamers = []
            total_count = 0
            
            for streamer, timestamps in list(self.recent_o7s.items()):
                recent = [t for t in timestamps if now - t < O7_CONFIG["window"]]
                self.recent_o7s[streamer] = recent
                threshold = self.get_threshold_for_channel(streamer)
                if len(recent) >= threshold:
                    display_name = self.streamer_names.get(streamer, streamer)
                    viewers = self.streamer_viewers.get(streamer, 0)
                    active_streamers.append((display_name, len(recent), streamer, threshold, viewers))
                    total_count += len(recent)
            
            if not active_streamers:
                return [], 0, False, False
            
            active_streamers.sort(key=lambda x: x[1], reverse=True)
            current_streamer_set = set(s[2] for s in active_streamers)
            
            if self.active_alert:
                self.active_alert_last_o7 = now
                self.active_alert_streamers = current_streamer_set
                if total_count > self.active_alert_peak:
                    self.active_alert_peak = total_count
                if total_count != self.active_alert_count:
                    self.active_alert_count = total_count
                    return active_streamers, total_count, False, True
                return active_streamers, total_count, False, True
            
            if (now - self.last_alert_time) <= O7_CONFIG["cooldown"]:
                return [], 0, False, False
            
            self.last_alert_time = now
            self.active_alert = True
            self.active_alert_streamers = current_streamer_set
            self.active_alert_count = total_count
            self.active_alert_peak = total_count
            self.active_alert_last_o7 = now
            return active_streamers, total_count, True, False
    
    def _clear_active_alert(self):
        self.active_alert = False
        self.active_alert_streamers = set()
        self.active_alert_count = 0
        self.active_alert_peak = 0
        self.active_alert_last_o7 = 0
    
    def can_announce(self):
        now = time.time()
        return (now - self.last_announcement_time) >= O7_CONFIG["cooldown"]
    
    def mark_announced(self):
        self.last_announcement_time = time.time()
    
    async def get_status(self):
        async with self.lock:
            now = time.time()
            activity = []
            for streamer, timestamps in self.recent_o7s.items():
                recent = [t for t in timestamps if now - t < O7_CONFIG["window"]]
                if recent:
                    viewers = self.streamer_viewers.get(streamer, 0)
                    threshold = self.get_threshold_for_channel(streamer)
                    activity.append({"login": streamer, "name": self.streamer_names.get(streamer, streamer), "count": len(recent), "viewers": viewers, "threshold": threshold})
            activity.sort(key=lambda x: x["count"], reverse=True)
            return {"active_channels": len(activity), "activity": activity[:5], "total_o7s": self.total_o7s_session, "alert_active": self.active_alert, "scale": O7_CONFIG["threshold_scale"], "min_threshold": O7_CONFIG["min_threshold"]}


# =============================================================================
# WORD TRACKER (for keyword detection feature)
# =============================================================================

class WordTracker:
    """Tracks keyword mentions across streams with user and channel cooldowns."""
    
    def __init__(self):
        self.lock = asyncio.Lock()
        self.user_last_trigger = {}     # {(channel, user): timestamp}
        self.channel_last_announce = {} # {channel: timestamp}
        self.channel_pending_count = {} # {channel: count since last announce}
        self.channel_pending_keywords = {} # {channel: set of keywords matched}
        self.total_mentions_session = 0
        self.keywords = [k.lower() for k in WORD_CONFIG["keywords"]]
        self.keyword_pattern = None
        self._update_pattern()
    
    def _update_pattern(self):
        """Update the regex pattern for current keywords."""
        if not self.keywords:
            self.keyword_pattern = None
            return
        # Build pattern that matches any of the keywords
        escaped = [re.escape(k) for k in self.keywords]
        pattern = r'\b(' + '|'.join(escaped) + r')\b'
        self.keyword_pattern = re.compile(pattern, re.IGNORECASE)
    
    def set_keywords(self, keywords):
        """Change the keywords being watched. Accepts list or comma-separated string."""
        if isinstance(keywords, str):
            # Parse comma-separated string
            keywords = [k.strip().lower() for k in keywords.split(',') if k.strip()]
        else:
            keywords = [k.lower() for k in keywords]
        
        self.keywords = keywords
        WORD_CONFIG["keywords"] = keywords
        self._update_pattern()
    
    def add_keyword(self, keyword):
        """Add a keyword to the watch list."""
        keyword = keyword.lower().strip()
        if keyword and keyword not in self.keywords:
            self.keywords.append(keyword)
            WORD_CONFIG["keywords"] = self.keywords
            self._update_pattern()
            return True
        return False
    
    def remove_keyword(self, keyword):
        """Remove a keyword from the watch list."""
        keyword = keyword.lower().strip()
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            WORD_CONFIG["keywords"] = self.keywords
            self._update_pattern()
            return True
        return False
    
    async def check_message(self, channel, user, message):
        """
        Check if message contains any keyword. Returns announcement info if should announce.
        Returns: (should_announce, mention_count, channel_name, matched_keywords) or (False, 0, None, None)
        """
        if not WORD_CONFIG["enabled"]:
            return False, 0, None, None
        
        if not self.keyword_pattern:
            return False, 0, None, None
        
        matches = self.keyword_pattern.findall(message)
        if not matches:
            return False, 0, None, None
        
        # Get unique matched keywords (lowercase)
        matched_keywords = set(m.lower() for m in matches)
        
        now = time.time()
        channel_lower = channel.lower()
        user_lower = user.lower()
        user_key = (channel_lower, user_lower)
        
        async with self.lock:
            self.total_mentions_session += 1
            
            # Check user cooldown
            last_user_time = self.user_last_trigger.get(user_key, 0)
            if (now - last_user_time) < WORD_CONFIG["user_cooldown"]:
                logging.debug(f"Word mention from {user} in #{channel} ignored (user cooldown)")
                return False, 0, None, None
            
            # User is not on cooldown, record this trigger
            self.user_last_trigger[user_key] = now
            
            # Increment pending count for this channel
            self.channel_pending_count[channel_lower] = self.channel_pending_count.get(channel_lower, 0) + 1
            
            # Track which keywords were matched during cooldown
            if channel_lower not in self.channel_pending_keywords:
                self.channel_pending_keywords[channel_lower] = set()
            self.channel_pending_keywords[channel_lower].update(matched_keywords)
            
            # Check channel cooldown
            last_announce_time = self.channel_last_announce.get(channel_lower, 0)
            if (now - last_announce_time) < WORD_CONFIG["channel_cooldown"]:
                logging.debug(f"Word mention in #{channel} queued (channel cooldown), pending: {self.channel_pending_count[channel_lower]}")
                return False, 0, None, None
            
            # Channel is ready for announcement!
            count = self.channel_pending_count[channel_lower]
            pending_keywords = self.channel_pending_keywords[channel_lower]
            self.channel_pending_count[channel_lower] = 0
            self.channel_pending_keywords[channel_lower] = set()
            self.channel_last_announce[channel_lower] = now
            
            return True, count, channel, pending_keywords
    
    async def get_status(self):
        """Get current word watcher status."""
        async with self.lock:
            pending_channels = {ch: count for ch, count in self.channel_pending_count.items() if count > 0}
            return {
                "enabled": WORD_CONFIG["enabled"],
                "keywords": self.keywords,
                "total_mentions": self.total_mentions_session,
                "pending_channels": pending_channels,
                "user_cooldown": WORD_CONFIG["user_cooldown"],
                "channel_cooldown": WORD_CONFIG["channel_cooldown"],
            }
    
    def reset_stats(self):
        """Reset session statistics."""
        self.total_mentions_session = 0
        self.channel_pending_count.clear()


# =============================================================================
# CHAT RECORDER (for logging chat to file)
# =============================================================================

class ChatRecorder:
    """Records a streamer's chat to a rolling log file."""
    
    def __init__(self):
        self.running = False
        self.target_channel = None
        self.connection_task = None
        self.message_count = 0
        self.start_time = None
        self._lock = asyncio.Lock()
    
    async def start(self, channel):
        """Start recording chat for a channel."""
        if not WEBSOCKETS_AVAILABLE:
            return False, "Missing dependency: pip install websockets"
        
        async with self._lock:
            if self.running:
                return False, f"Already recording #{self.target_channel}"
            
            channel = channel.lower().strip()
            self.target_channel = channel
            self.running = True
            self.message_count = 0
            self.start_time = time.time()
            
            # Create log directory
            os.makedirs(CHAT_RECORD_CONFIG["log_dir"], exist_ok=True)
            
            # Start connection task
            self.connection_task = asyncio.create_task(self._chat_reader())
            logging.info(f"ChatRecorder: Started recording #{channel}")
            return True, None
    
    async def stop(self):
        """Stop recording chat."""
        async with self._lock:
            if not self.running:
                return False, "Not recording any chat"
            
            self.running = False
            channel = self.target_channel
            count = self.message_count
            duration = time.time() - self.start_time if self.start_time else 0
            
            if self.connection_task:
                self.connection_task.cancel()
                try:
                    await self.connection_task
                except asyncio.CancelledError:
                    pass
                self.connection_task = None
            
            self.target_channel = None
            logging.info(f"ChatRecorder: Stopped recording #{channel}")
            return True, (channel, count, duration)
    
    def _get_log_path(self, channel):
        """Get the log file path for a channel."""
        return os.path.join(CHAT_RECORD_CONFIG["log_dir"], f"{channel}_chat.log")
    
    def _enforce_rolling_limit(self, filepath):
        """
        Enforce the rolling log size limit by trimming from the beginning.
        Keeps the file under max_file_size by removing oldest lines.
        """
        max_size = CHAT_RECORD_CONFIG["max_file_size"]
        
        try:
            if not os.path.exists(filepath):
                return
            
            file_size = os.path.getsize(filepath)
            if file_size <= max_size:
                return
            
            # Read file, remove oldest lines until under limit
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Remove lines from the start until we're under the limit
            total_size = file_size
            while lines and total_size > max_size * 0.8:  # Trim to 80% to avoid constant trimming
                removed_line = lines.pop(0)
                total_size -= len(removed_line.encode('utf-8'))
            
            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            logging.debug(f"ChatRecorder: Trimmed log to {len(lines)} lines ({total_size} bytes)")
            
        except Exception as e:
            logging.error(f"ChatRecorder: Error enforcing rolling limit: {e}")
    
    def _write_message(self, channel, user, message):
        """Write a chat message to the log file."""
        filepath = self._get_log_path(channel)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Append message
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {user}: {message}\n")
            
            self.message_count += 1
            
            # Periodically check size (every 100 messages)
            if self.message_count % 100 == 0:
                self._enforce_rolling_limit(filepath)
                
        except Exception as e:
            logging.error(f"ChatRecorder: Error writing to log: {e}")
    
    async def _chat_reader(self):
        """Connect to Twitch chat via websocket and record messages."""
        uri = "wss://irc-ws.chat.twitch.tv:443"
        nick = f"justinfan{random.randint(10000, 99999)}"
        channel = self.target_channel
        
        while self.running:
            try:
                async with websockets.connect(uri) as ws:
                    await ws.send(f"NICK {nick}")
                    await ws.send(f"JOIN #{channel}")
                    logging.info(f"ChatRecorder: Connected to #{channel}")
                    
                    # Write session start marker
                    filepath = self._get_log_path(channel)
                    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(filepath, 'a', encoding='utf-8') as f:
                        f.write(f"\n=== Recording started at {start_time} ===\n")
                    
                    while self.running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            # Send keepalive
                            await ws.send("PING :keepalive")
                            continue
                        
                        for line in message.split('\r\n'):
                            if line.startswith('PING'):
                                await ws.send('PONG :tmi.twitch.tv')
                            elif 'PRIVMSG' in line:
                                try:
                                    # Parse: :user!user@user.tmi.twitch.tv PRIVMSG #channel :message
                                    user = line.split('!')[0][1:]
                                    msg = line.split('PRIVMSG', 1)[1].split(':', 1)[1]
                                    self._write_message(channel, user, msg)
                                except Exception:
                                    pass
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logging.warning(f"ChatRecorder: Connection error for #{channel}: {type(e).__name__}, reconnecting...")
                    await asyncio.sleep(5)
        
        # Write session end marker
        try:
            filepath = self._get_log_path(channel)
            end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"=== Recording stopped at {end_time} ({self.message_count} messages) ===\n\n")
        except Exception:
            pass
    
    async def get_status(self):
        """Get current recording status."""
        async with self._lock:
            if not self.running:
                return {
                    "recording": False,
                    "channel": None,
                    "messages": 0,
                    "duration": 0,
                    "file_size": 0
                }
            
            duration = time.time() - self.start_time if self.start_time else 0
            filepath = self._get_log_path(self.target_channel)
            file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
            
            return {
                "recording": True,
                "channel": self.target_channel,
                "messages": self.message_count,
                "duration": duration,
                "file_size": file_size
            }


# =============================================================================
# O7 WATCHER
# =============================================================================

class O7Watcher:
    def __init__(self, bot, tracker, word_tracker=None):
        self.bot = bot
        self.tracker = tracker
        self.word_tracker = word_tracker
        self.running = False
        self.connections = {}
        self.refresh_task = None
        self.monitor_task = None
        self._channels_lock = asyncio.Lock()
        self.pending_spikes = {}
        self.broadcaster_ids = {}
        self.pending_word_announces = asyncio.Queue()
    
    async def start(self):
        if not WEBSOCKETS_AVAILABLE or not REQUESTS_AVAILABLE:
            return False, "Missing dependencies: pip install websockets requests"
        if self.running:
            return False, "Already running!"
        self.running = True
        logging.info("O7 Watcher starting...")
        self.refresh_task = asyncio.create_task(self._refresh_loop())
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        return True, None
    
    async def stop(self):
        if not self.running:
            return
        logging.info("O7 Watcher stopping...")
        self.running = False
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        async with self._channels_lock:
            tasks = list(self.connections.values())
            for task in tasks:
                task.cancel()
            self.connections.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logging.info("O7 Watcher stopped")
    
    def _get_hardcore_streams(self):
        streams = []
        headers = {"Client-ID": "kimne78kx3ncx6brgo4mv6wki5h1ko", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        payload = [{"operationName": "DirectoryPage_Game", "variables": {"slug": "world-of-warcraft", "options": {"sort": "VIEWER_COUNT", "freeformTags": None, "tags": [], "recommendationsContext": {"platform": "web"}, "requestID": "JIRA-VXP-2397"}, "sortTypeIsRecency": False, "limit": 100}, "extensions": {"persistedQuery": {"version": 1, "sha256Hash": "c7c9d5aad09155c4161d2382092dc44610367f3536aac39019ec2582ae5065f9"}}}]
        
        try:
            resp = requests.post("https://gql.twitch.tv/gql", headers=headers, json=payload, timeout=10)
            data = resp.json()
            edges = data[0].get("data", {}).get("game", {}).get("streams", {}).get("edges", [])
            for edge in edges:
                node = edge.get("node", {})
                title = (node.get("title") or "").lower()
                tags = [t.get("localizedName", "").lower() for t in node.get("freeformTags", [])]
                is_hardcore = ("hardcore" in title or any("hardcore" in t for t in tags) or " hc " in f" {title} " or title.startswith("hc ") or title.endswith(" hc"))
                if is_hardcore:
                    broadcaster = node.get("broadcaster", {})
                    streams.append({"login": broadcaster.get("login", ""), "name": broadcaster.get("displayName", ""), "title": node.get("title", ""), "viewers": node.get("viewersCount", 0)})
        except Exception as e:
            logging.error(f"Failed to fetch streams: {e}")
        
        if not streams:
            try:
                payload2 = {"query": "query { game(slug: \"world-of-warcraft\") { streams(first: 100) { edges { node { title viewersCount broadcaster { login displayName } freeformTags { name } } } } } }"}
                resp = requests.post("https://gql.twitch.tv/gql", headers=headers, json=payload2, timeout=10)
                data = resp.json()
                edges = data.get("data", {}).get("game", {}).get("streams", {}).get("edges", [])
                for edge in edges:
                    node = edge.get("node", {})
                    title = (node.get("title") or "").lower()
                    tags = [t.get("name", "").lower() for t in node.get("freeformTags", [])]
                    is_hardcore = ("hardcore" in title or any("hardcore" in t for t in tags) or " hc " in f" {title} ")
                    if is_hardcore:
                        broadcaster = node.get("broadcaster", {})
                        streams.append({"login": broadcaster.get("login", ""), "name": broadcaster.get("displayName", ""), "title": node.get("title", ""), "viewers": node.get("viewersCount", 0)})
            except Exception as e:
                logging.error(f"Fallback API also failed: {e}")
        return streams
    
    async def _refresh_loop(self):
        while self.running:
            try:
                logging.info("O7 Watcher: Refreshing stream list...")
                loop = asyncio.get_running_loop()
                streams = await loop.run_in_executor(None, self._get_hardcore_streams)
                if streams:
                    logging.info(f"O7 Watcher: Found {len(streams)} hardcore streams")
                    live_logins = set(s["login"].lower() for s in streams)
                    for s in sorted(streams, key=lambda x: x["viewers"], reverse=True):
                        await self._join_channel(s["login"], s["name"], s.get("title", ""), s.get("viewers", 0))
                    async with self._channels_lock:
                        stale = [ch for ch in self.connections if ch not in live_logins]
                    for ch in stale:
                        await self._leave_channel(ch)
                    async with self._channels_lock:
                        logging.info(f"O7 Watcher: Monitoring {len(self.connections)} channels")
                else:
                    logging.warning("O7 Watcher: No hardcore streams found")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"O7 Watcher refresh error: {e}")
            try:
                await asyncio.sleep(O7_CONFIG["refresh_interval"])
            except asyncio.CancelledError:
                break
    
    async def _monitor_loop(self):
        while self.running:
            try:
                await asyncio.sleep(1)
                if not self.running:
                    break
                now = time.time()
                
                # Handle word watcher announcements
                while not self.pending_word_announces.empty():
                    try:
                        display_name, count, user, matched_keywords, source_channel = self.pending_word_announces.get_nowait()
                        current_channel = await self._get_bot_channel()
                        if current_channel:
                            # Skip announcing if the word hit is from the channel we're already in
                            if source_channel.lower() == current_channel.lower():
                                logging.debug(f"Word Watcher: Skipping announcement for #{source_channel} (already in this channel)")
                                continue
                            
                            # Format matched keywords for display
                            if matched_keywords:
                                keywords_str = ', '.join(sorted(matched_keywords))
                            else:
                                keywords_str = ', '.join(WORD_CONFIG["keywords"])
                            
                            if count == 1:
                                msg = f"📢 '{keywords_str}' mentioned in {display_name}'s chat!"
                            else:
                                msg = f"📢 '{keywords_str}' mentioned {count}x in {display_name}'s chat!"
                            await self.bot._send_message_safe(current_channel, msg)
                            logging.info(f"Word Watcher announced: {msg}")
                    except asyncio.QueueEmpty:
                        break
                
                # Handle o7 spike detection
                streamers, total_count, is_new, is_update = await self.tracker.check_for_spike()
                if is_new and streamers and self.tracker.can_announce():
                    top_login = streamers[0][2]
                    if top_login not in self.pending_spikes:
                        self.pending_spikes[top_login] = (now, streamers, total_count)
                        logging.info(f"O7 spike detected for {top_login}, waiting {O7_CONFIG['clip_verify_delay']}s for clip verification...")
                
                verify_delay = O7_CONFIG["clip_verify_delay"]
                spikes_to_remove = []
                for login, (spike_time, spike_streamers, spike_count) in list(self.pending_spikes.items()):
                    if now - spike_time >= verify_delay:
                        has_clips = await self._check_recent_clips(login, spike_time)
                        if has_clips:
                            top = spike_streamers[0]
                            names = ", ".join(s[0] for s in spike_streamers[:3])
                            if len(spike_streamers) > 3:
                                names += f" +{len(spike_streamers) - 3} more"
                            msg = f"☠️ POTENTIAL DEATH! {spike_count} o7s + clip created: {names}"
                            # Only announce if o7 announcements are enabled
                            if O7_CONFIG["announce_enabled"]:
                                current_channel = await self._get_bot_channel()
                                if current_channel:
                                    await self.bot._send_message_safe(current_channel, msg)
                                    logging.info(f"O7 Watcher announced (clip verified): {msg}")
                            else:
                                logging.info(f"O7 spike detected but announcements disabled: {msg}")
                            self.tracker.mark_announced()
                        else:
                            logging.info(f"O7 spike for {login} not verified (no clips), discarding")
                        spikes_to_remove.append(login)
                for login in spikes_to_remove:
                    del self.pending_spikes[login]
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"O7 monitor error: {e}")
    
    async def _get_bot_channel(self):
        return self.bot.current_channel
    
    async def _get_broadcaster_id(self, login):
        login_lower = login.lower()
        if login_lower in self.broadcaster_ids:
            return self.broadcaster_ids[login_lower]
        try:
            token = await self.bot.token_manager.get_token()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.twitch.tv/helix/users?login={login_lower}", headers={"Authorization": f"Bearer {token}", "Client-Id": self.bot.token_manager.client_id}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("data"):
                            broadcaster_id = data["data"][0]["id"]
                            self.broadcaster_ids[login_lower] = broadcaster_id
                            return broadcaster_id
        except Exception as e:
            logging.error(f"Failed to get broadcaster ID for {login}: {e}")
        return None
    
    async def _check_recent_clips(self, login, since_time):
        broadcaster_id = await self._get_broadcaster_id(login)
        if not broadcaster_id:
            logging.warning(f"Could not get broadcaster ID for {login}, skipping clip check")
            return False
        try:
            token = await self.bot.token_manager.get_token()
            from datetime import datetime, timezone
            started_at = datetime.fromtimestamp(since_time - 10, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.twitch.tv/helix/clips?broadcaster_id={broadcaster_id}&started_at={started_at}&first=5", headers={"Authorization": f"Bearer {token}", "Client-Id": self.bot.token_manager.client_id}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        clips = data.get("data", [])
                        if clips:
                            logging.info(f"Found {len(clips)} recent clip(s) for {login}")
                            return True
                        else:
                            logging.debug(f"No recent clips for {login}")
                            return False
                    else:
                        logging.warning(f"Clips API returned {resp.status} for {login}")
                        return False
        except Exception as e:
            logging.error(f"Failed to check clips for {login}: {e}")
            return False
    
    async def _join_channel(self, login, name, title="", viewers=0):
        login_lower = login.lower()
        self.tracker.set_streamer_viewers(login, viewers)
        async with self._channels_lock:
            if login_lower in self.connections:
                return
            self.tracker.set_streamer_title(login, title)
            task = asyncio.create_task(self._channel_reader(login, name))
            self.connections[login_lower] = task
    
    async def _leave_channel(self, login):
        login_lower = login.lower()
        async with self._channels_lock:
            if login_lower in self.connections:
                self.connections[login_lower].cancel()
                del self.connections[login_lower]
                logging.debug(f"O7 Watcher: Left #{login}")
    
    async def _channel_reader(self, login, display_name):
        uri = "wss://irc-ws.chat.twitch.tv:443"
        nick = f"justinfan{random.randint(10000, 99999)}"
        try:
            async with websockets.connect(uri) as ws:
                await ws.send(f"NICK {nick}")
                await ws.send(f"JOIN #{login}")
                logging.debug(f"O7 Watcher: Joined #{login}")
                while self.running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        await ws.send("PING :keepalive")
                        continue
                    for line in message.split('\r\n'):
                        if line.startswith('PING'):
                            await ws.send('PONG :tmi.twitch.tv')
                        elif 'PRIVMSG' in line:
                            try:
                                user = line.split('!')[0][1:]
                                msg = line.split('PRIVMSG', 1)[1].split(':', 1)[1]
                                if O7_PATTERN.search(msg):
                                    result = await self.tracker.add_o7(login, display_name, user)
                                    if result is not None:
                                        logging.debug(f"[o7] #{login}: {user} (count: {result})")
                                # Check for keyword (word watcher)
                                if self.word_tracker and WORD_CONFIG["enabled"]:
                                    should_announce, count, channel, matched_keywords = await self.word_tracker.check_message(login, user, msg)
                                    if should_announce:
                                        await self.pending_word_announces.put((display_name, count, user, matched_keywords, login))
                            except Exception:
                                pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.debug(f"O7 Watcher: Connection closed for #{login}: {type(e).__name__}")
        finally:
            async with self._channels_lock:
                self.connections.pop(login.lower(), None)


# =============================================================================
# BOT CLASS
# =============================================================================

class Bot(commands.Bot):
    def __init__(self, token, token_manager):
        super().__init__(token=f'oauth:{token}', prefix='!', initial_channels=[HOME_CHANNEL])
        self.token_manager = token_manager
        self.request_queue = asyncio.Queue()
        self.queue_worker_task = None
        self.home_channel = HOME_CHANNEL
        self.current_channel = self.home_channel
        self._worker_running = False
        self.transcriber = AudioTranscriber(whisper_model="base")
        self.vision = VisionAnalyzer(model=VISION_MODEL)
        self.clip_recorder = ClipRecorder()
        self._token_refresh_task = None
        self.o7_tracker = O7Tracker()
        self.word_tracker = WordTracker()
        self.o7_watcher = O7Watcher(self, self.o7_tracker, self.word_tracker)
        self.chat_recorder = ChatRecorder()
        self.raffle_active = False
        self.raffle_entries = set()
        self.raffle_channel = None
        self.signups = self._load_signups()
        self.banned_channels = self._load_banned_channels()
        self.banned_users = self._load_banned_users()
        
        # Shared lock for heavy models (Whisper + Vision)
        # Only one can run at a time to stay under 4GB RAM
        self.heavy_model_lock = asyncio.Lock()
    
    async def event_ready(self):
        logging.info(f'Bot is ready! Logged in as {self.nick}')
        self._start_queue_worker()
        self._start_token_refresh()
    
    async def event_raw_data(self, data):
        """Handle raw IRC data to detect ban notices."""
        # Look for NOTICE messages that indicate we're banned
        # Format: @msg-id=msg_banned :tmi.twitch.tv NOTICE #channel :You are permanently banned...
        if 'NOTICE' in data and ('msg_banned' in data or 'msg_timedout' in data):
            try:
                # Extract channel name from the NOTICE
                if '#' in data:
                    parts = data.split('#')
                    if len(parts) >= 2:
                        channel_part = parts[1].split()[0].split(':')[0].strip()
                        channel_name = channel_part.lower()
                        
                        if channel_name and channel_name != self.home_channel:
                            logging.warning(f"Bot is banned/timed out from #{channel_name}! Auto-returning home.")
                            
                            # Track the banned channel
                            self.banned_channels.add(channel_name)
                            self._save_banned_channels()
                            
                            # Schedule return to home (can't await directly in raw_data)
                            asyncio.create_task(self._handle_ban_return(channel_name))
            except Exception as e:
                logging.error(f"Error processing ban notice: {e}")
    
    async def _handle_ban_return(self, banned_channel):
        """Handle returning home after being banned from a channel."""
        try:
            logging.info(f"Leaving banned channel #{banned_channel} and returning home")
            await asyncio.sleep(1)  # Small delay to let things settle
            
            try:
                await self.part_channels([banned_channel])
            except Exception:
                pass  # May fail if already disconnected
            
            # Rejoin home channel if not already there
            if self.current_channel != self.home_channel:
                await self.join_channels([self.home_channel])
                self.current_channel = self.home_channel
                await asyncio.sleep(1)
                await self._send_message_safe(self.home_channel, f"🚫 Got banned from #{banned_channel}, I'm back home!")
        except Exception as e:
            logging.error(f"Error returning home after ban: {e}")
    
    def _start_queue_worker(self):
        if self.queue_worker_task and not self.queue_worker_task.done():
            return
        self._worker_running = True
        self.queue_worker_task = asyncio.create_task(self.process_queue())
        asyncio.create_task(self._monitor_worker())
        logging.info("Queue worker started")
    
    def _start_token_refresh(self):
        if self._token_refresh_task and not self._token_refresh_task.done():
            return
        self._token_refresh_task = asyncio.create_task(self._auto_refresh_token())
        logging.info("Token refresh task started")
    
    async def _auto_refresh_token(self):
        while self._worker_running:
            await asyncio.sleep(3 * 60 * 60)
            try:
                await self.token_manager.refresh()
                logging.info("Token auto-refresh complete")
            except Exception as e:
                logging.error(f"Token auto-refresh failed: {e}")
    
    async def _monitor_worker(self):
        while self._worker_running:
            await asyncio.sleep(5)
            if self.queue_worker_task and self.queue_worker_task.done():
                exc = self.queue_worker_task.exception() if not self.queue_worker_task.cancelled() else None
                if exc:
                    logging.error(f"Queue worker died: {exc}")
                logging.info("Restarting queue worker...")
                self.queue_worker_task = asyncio.create_task(self.process_queue())
    
    async def close(self):
        self._worker_running = False
        await self.o7_watcher.stop()
        if self.queue_worker_task:
            self.queue_worker_task.cancel()
            try:
                await self.queue_worker_task
            except asyncio.CancelledError:
                pass
        if self._token_refresh_task:
            self._token_refresh_task.cancel()
            try:
                await self._token_refresh_task
            except asyncio.CancelledError:
                pass
        await super().close()
    
    async def _send_message_safe(self, channel_name, message, max_retries=3):
        for attempt in range(1, max_retries + 1):
            try:
                channel = self.get_channel(channel_name)
                if channel is None:
                    logging.warning(f"Channel {channel_name} not found, rejoining...")
                    await self.join_channels([channel_name])
                    await asyncio.sleep(1)
                    channel = self.get_channel(channel_name)
                    if channel is None:
                        logging.error(f"Could not rejoin {channel_name}")
                        continue
                await channel.send(message)
                await asyncio.sleep(0.1)
                logging.info(f"Message sent to {channel_name}")
                return True
            except Exception as e:
                logging.error(f"Send failed attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
        return False
    
    async def process_queue(self):
        logging.info("Queue worker starting...")
        while True:
            try:
                channel_name, prompt = await self.request_queue.get()
                logging.info(f"Processing: {prompt[:50]}...")
                try:
                    response = await self.query_ollama(prompt)
                    if len(response) > 450:
                        truncated = response[:447]
                        for char in ['. ', '! ', '? ', ', ']:
                            last_break = truncated.rfind(char)
                            if last_break > 300:
                                truncated = truncated[:last_break + 1]
                                break
                        response = truncated + "..."
                    await self._send_message_safe(channel_name, response)
                except Exception as e:
                    logging.error(f"Failed to process: {e}")
                    await self._send_message_safe(channel_name, "Something went wrong, try again!")
                self.request_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Queue worker error: {e}")
                await asyncio.sleep(1)
    
    async def query_ollama(self, prompt, max_retries=2):
        logging.info(f"Got prompt: {prompt}")
        messages = [{"role": "system", "content": "You are a Twitch chatbot. Give very short answers, 1-2 sentences max. Be direct."}, {"role": "user", "content": prompt}]
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post('http://localhost:11434/api/chat', json={'model': 'qwen2.5:1.5b-instruct-q4_K_M', 'messages': messages, 'stream': False, 'keep_alive': '30m', 'options': {'temperature': 0.7, 'top_p': 0.8, 'num_predict': 150}}) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            response = data.get('message', {}).get('content', '').strip()
                            if response:
                                return response
                            else:
                                last_error = "AI returned empty response"
                        else:
                            last_error = f"AI error (status {resp.status})"
            except aiohttp.ClientConnectorError:
                last_error = "Can't connect to AI"
            except asyncio.TimeoutError:
                last_error = "AI timed out"
            except Exception as e:
                last_error = f"Error: {type(e).__name__}"
            if attempt < max_retries:
                await asyncio.sleep(1)
        return last_error or "AI unavailable, please try again"
    
    async def handle_listento(self, channel_name, target_channel):
        """Handle audio transcription with resource lock."""
        async with self.heavy_model_lock:
            logging.info(f"LISTENTO: capturing from {target_channel} (lock acquired)")
            transcript, error = await self.transcriber.listen_and_transcribe(target_channel, duration=20)
            
            if error:
                await self._send_message_safe(channel_name, error)
                return
            
            summary_prompt = f"Summarize what this streamer said in 1-2 short sentences: {transcript}"
            response = await self.query_ollama(summary_prompt)
            final_response = f"🎧 {target_channel}: {response}"
            
            if len(final_response) > 450:
                truncated = final_response[:447]
                for char in ['. ', '! ', '? ', ', ']:
                    last_break = truncated.rfind(char)
                    if last_break > 300:
                        truncated = truncated[:last_break + 1]
                        break
                final_response = truncated + "..."
            
            await self._send_message_safe(channel_name, final_response)
            
            # Optionally unload whisper to free memory
            # self.transcriber.unload_model()
    
    async def handle_lookat(self, channel_name, source, prompt):
        """Handle vision analysis with resource lock."""
        async with self.heavy_model_lock:
            logging.info(f"LOOKAT: analyzing {source} with prompt '{prompt[:30]}...' (lock acquired)")
            
            result, error = await self.vision.analyze_from_source(source, prompt)
            
            if error:
                await self._send_message_safe(channel_name, f"👁️ {error}")
                return
            
            # Format response
            source_name = source.lstrip('@').split('/')[-1].split('?')[0][:20]
            final_response = f"👁️ {source_name}: {result}"
            
            if len(final_response) > 450:
                truncated = final_response[:447]
                for char in ['. ', '! ', '? ', ', ']:
                    last_break = truncated.rfind(char)
                    if last_break > 300:
                        truncated = truncated[:last_break + 1]
                        break
                final_response = truncated + "..."
            
            await self._send_message_safe(channel_name, final_response)
            
            # Optionally unload vision to free memory
            # self.vision.unload_model()

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================
    
    async def cmd_of(self, channel_name):
        await self._send_message_safe(channel_name, 'OnlyFangs!')
    
    async def cmd_home(self, channel_name):
        if channel_name.lower() == self.home_channel:
            await self._send_message_safe(channel_name, "Already home! 🏠")
            return
        try:
            await self._send_message_safe(channel_name, f"Heading home to {self.home_channel}! 👋")
            await self.part_channels([channel_name])
            await self.join_channels([self.home_channel])
            self.current_channel = self.home_channel
        except Exception as e:
            logging.error(f"Failed to return home: {e}")
    
    async def cmd_goto(self, channel_name, new_channel):
        if not new_channel:
            await self._send_message_safe(channel_name, "Need a channel name! AI goto channelname")
            return
        if new_channel == 'home':
            await self.cmd_home(channel_name)
            return
        
        # Check if already in this channel
        if new_channel.lower() == channel_name.lower():
            await self._send_message_safe(channel_name, f"Already in #{new_channel}! 📍")
            return
        
        # Check if channel is banned
        if new_channel.lower() in self.banned_channels:
            await self._send_message_safe(channel_name, f"🚫 Can't go to #{new_channel} - I'm banned there!")
            return
        
        try:
            await self._send_message_safe(channel_name, f"Heading over to {new_channel}! 👋")
            await self.part_channels([channel_name])
            await self.join_channels([new_channel])
            self.current_channel = new_channel
        except Exception as e:
            logging.error(f"Failed to switch channels: {e}")
            await self._send_message_safe(channel_name, f"Couldn't switch: {e}")
    
    async def cmd_banned(self, channel_name):
        if not self.banned_channels:
            await self._send_message_safe(channel_name, "🚫 No banned channels recorded!")
            return
        channels = ', '.join(sorted(self.banned_channels))
        await self._send_message_safe(channel_name, f"🚫 Banned from: {channels}")
    
    async def cmd_unban(self, channel_name, target_channel):
        if not target_channel:
            await self._send_message_safe(channel_name, "Need a channel! AI unban channelname")
            return
        target_lower = target_channel.lower()
        if target_lower not in self.banned_channels:
            await self._send_message_safe(channel_name, f"#{target_channel} isn't in the ban list!")
            return
        self.banned_channels.discard(target_lower)
        self._save_banned_channels()
        await self._send_message_safe(channel_name, f"✅ Removed #{target_channel} from ban list. You can try going there again!")
    
    async def cmd_ban_user(self, channel_name, target_user):
        """Ban a user from using public bot commands like AI SIGNUP."""
        if not target_user:
            await self._send_message_safe(channel_name, "Need a username! AI BAN @username")
            return
        target_lower = target_user.lower().lstrip('@')
        if target_lower in self.banned_users:
            await self._send_message_safe(channel_name, f"⛔ @{target_lower} is already banned!")
            return
        self.banned_users.add(target_lower)
        self._save_banned_users()
        await self._send_message_safe(channel_name, f"⛔ Banned @{target_lower} from using bot commands.")
        logging.info(f"User banned: {target_lower}")
    
    async def cmd_unban_user(self, channel_name, target_user):
        """Unban a user so they can use public bot commands again."""
        if not target_user:
            await self._send_message_safe(channel_name, "Need a username! AI UNBANUSER @username")
            return
        target_lower = target_user.lower().lstrip('@')
        if target_lower not in self.banned_users:
            await self._send_message_safe(channel_name, f"@{target_lower} isn't in the user ban list!")
            return
        self.banned_users.discard(target_lower)
        self._save_banned_users()
        await self._send_message_safe(channel_name, f"✅ Unbanned @{target_lower}. They can use bot commands again!")
        logging.info(f"User unbanned: {target_lower}")
    
    async def cmd_banlist(self, channel_name):
        """List all banned users."""
        if not self.banned_users:
            await self._send_message_safe(channel_name, "⛔ No users are banned!")
            return
        users = ', '.join(sorted(self.banned_users))
        if len(users) > 400:
            # Truncate if too long
            users = users[:400] + f"... (+{len(self.banned_users)} total)"
        await self._send_message_safe(channel_name, f"⛔ Banned users: {users}")
    
    async def cmd_listento(self, channel_name, target_channel):
        if not target_channel:
            await self._send_message_safe(channel_name, "Need a channel! AI listento channelname")
            return
        
        # Check if lock is already held
        if self.heavy_model_lock.locked():
            await self._send_message_safe(channel_name, "🎧 Another heavy task is running, please wait...")
            
        await self._send_message_safe(channel_name, f"🎧 Listening to {target_channel} for 20s...")
        await self.handle_listento(channel_name, target_channel)
    
    async def cmd_lookat(self, channel_name, args):
        """Parse and handle AI LOOKAT command."""
        if not args:
            await self._send_message_safe(channel_name, "👁️ Usage: AI lookat <url/@streamer> <question>")
            return
        
        # Parse source and prompt from args
        parts = args.split(None, 1)  # Split on first whitespace
        
        if len(parts) < 2:
            await self._send_message_safe(channel_name, "👁️ Need both a source and a question! Example: AI lookat @cohhcarnage what game is this?")
            return
        
        source = parts[0]
        prompt = parts[1]
        
        # Check if lock is already held
        if self.heavy_model_lock.locked():
            await self._send_message_safe(channel_name, "👁️ Another heavy task is running, please wait...")
        
        # Determine source type for user feedback
        if source.startswith('http'):
            if 'twitch.tv/' in source:
                await self._send_message_safe(channel_name, f"👁️ Capturing screenshot from stream...")
            else:
                await self._send_message_safe(channel_name, f"👁️ Downloading image...")
        else:
            await self._send_message_safe(channel_name, f"👁️ Capturing screenshot from {source.lstrip('@')}...")
        
        await self.handle_lookat(channel_name, source, prompt)
    
    async def cmd_clip(self, channel_name, args):
        """
        Record video clip with optional quality, title, and effects.
        
        Usage: AI CLIP <channel> [min] [quality] [title] [flags]
        Examples:
            AI CLIP shroud
            AI CLIP shroud 3 720p
            AI CLIP shroud 2 [epic death]
            AI CLIP shroud 3 720p [boss kill] --compress --text --chat
        """
        cfg = RECORDING_CONFIG
        channel, minutes, quality, title, flags = self.clip_recorder._parse_clip_args(args)
        
        if not channel:
            await self._send_message_safe(
                channel_name,
                f"Usage: AI CLIP <channel> [min] [720p/480p] [title] [--compress] [--text] [--chat]"
            )
            return
        
        # Build status message
        extras = []
        if quality != "best":
            extras.append(quality)
        if title:
            extras.append(f"[{title}]")
        if 'compress' in flags:
            extras.append("compressed")
        if 'text' in flags:
            extras.append("text overlay")
        if 'chat' in flags:
            extras.append("chat overlay")
        
        extra_str = f" ({', '.join(extras)})" if extras else ""
        await self._send_message_safe(channel_name, f"🎬 Recording {minutes}min from {channel}{extra_str}...")
        
        success, result, filename = await self.clip_recorder.record_clip(
            channel, minutes, quality, title, flags
        )
        
        if success:
            clip_url = f"{cfg['clip_base_url']}/{filename}"
            quality_note = f" ({quality})" if quality != "best" else ""
            flags_note = ""
            if 'compress' in flags:
                flags_note += " (compressed)"
            if 'text' in flags:
                flags_note += " (text)"
            if 'chat' in flags:
                flags_note += " (chat)"
            await self._send_message_safe(channel_name, f"✅ {clip_url} ({result}){quality_note}{flags_note}")
        else:
            await self._send_message_safe(channel_name, f"❌ Clip failed: {result}")
    
    async def cmd_audio(self, channel_name, args):
        """
        Record audio only from a stream.
        
        Usage: AI AUDIO <channel> [minutes]
        Examples:
            AI AUDIO shroud
            AI AUDIO shroud 5
        """
        cfg = RECORDING_CONFIG
        channel, minutes = self.clip_recorder._parse_audio_args(args)
        
        if not channel:
            await self._send_message_safe(
                channel_name,
                f"Usage: AI AUDIO <channel> [minutes 1-{cfg['audio_max_min']}]"
            )
            return
        
        await self._send_message_safe(channel_name, f"🎵 Recording {minutes}min audio from {channel}...")
        
        success, result, filename = await self.clip_recorder.record_audio(channel, minutes)
        
        if success:
            audio_url = f"{cfg['audio_base_url']}/{filename}"
            await self._send_message_safe(channel_name, f"✅ {audio_url} ({result})")
        else:
            await self._send_message_safe(channel_name, f"❌ Audio failed: {result}")
    
    async def cmd_gif(self, channel_name, args):
        """
        Record a GIF from a stream.
        
        Usage: AI GIF <channel> [seconds]
        Examples:
            AI GIF shroud
            AI GIF shroud 15
        """
        cfg = RECORDING_CONFIG
        channel, seconds = self.clip_recorder._parse_gif_args(args)
        
        if not channel:
            await self._send_message_safe(
                channel_name,
                f"Usage: AI GIF <channel> [seconds 1-{cfg['gif_max_sec']}]"
            )
            return
        
        await self._send_message_safe(channel_name, f"🎞️ Recording {seconds}s GIF from {channel}...")
        
        success, result, filename = await self.clip_recorder.record_gif(channel, seconds)
        
        if success:
            gif_url = f"{cfg['gif_base_url']}/{filename}"
            await self._send_message_safe(channel_name, f"✅ {gif_url} ({result})")
        else:
            await self._send_message_safe(channel_name, f"❌ GIF failed: {result}")
    
    async def cmd_o7_start(self, channel_name):
        if self.o7_watcher.running:
            # Watcher already running (possibly started by word watcher)
            if O7_CONFIG["announce_enabled"]:
                # Already fully on
                status = await self.o7_tracker.get_status()
                async with self.o7_watcher._channels_lock:
                    channel_count = len(self.o7_watcher.connections)
                await self._send_message_safe(channel_name, f"☠️ o7 already active! Monitoring {channel_count} streams, {status['total_o7s']} o7s detected")
            else:
                # Channels running but o7 announcements were off - enable them
                O7_CONFIG["announce_enabled"] = True
                async with self.o7_watcher._channels_lock:
                    count = len(self.o7_watcher.connections)
                await self._send_message_safe(channel_name, f"☠️ o7 announcements enabled! Monitoring {count} streams. Say 'AI stopo7' to stop.")
            return
        
        await self._send_message_safe(channel_name, "☠️ Starting o7 watcher for WoW Hardcore streams...")
        O7_CONFIG["announce_enabled"] = True
        success, error = await self.o7_watcher.start()
        if success:
            await asyncio.sleep(3)
            async with self.o7_watcher._channels_lock:
                count = len(self.o7_watcher.connections)
            await self._send_message_safe(channel_name, f"☠️ o7 watcher active! Monitoring {count} streams. Say 'AI stopo7' to stop.")
        else:
            await self._send_message_safe(channel_name, f"Failed to start: {error}")
    
    async def cmd_o7_stop(self, channel_name):
        if not self.o7_watcher.running:
            await self._send_message_safe(channel_name, "o7 watcher isn't running!")
            return
        
        status = await self.o7_tracker.get_status()
        O7_CONFIG["announce_enabled"] = False
        
        # Only stop the watcher entirely if word watching is also disabled
        if not WORD_CONFIG["enabled"]:
            await self.o7_watcher.stop()
            await self._send_message_safe(channel_name, f"☠️ o7 watcher stopped. Total o7s detected: {status['total_o7s']}")
        else:
            await self._send_message_safe(channel_name, f"☠️ o7 announcements disabled (word watcher still using channels). Total o7s: {status['total_o7s']}")
    
    async def cmd_o7_status(self, channel_name):
        if not self.o7_watcher.running:
            await self._send_message_safe(channel_name, "o7 watcher isn't running. Say 'AI o7' to start!")
            return
        status = await self.o7_tracker.get_status()
        async with self.o7_watcher._channels_lock:
            channel_count = len(self.o7_watcher.connections)
        
        # Show status of both features
        o7_status = "ON" if O7_CONFIG["announce_enabled"] else "OFF"
        word_status = "ON" if WORD_CONFIG["enabled"] else "OFF"
        
        if status['activity']:
            top = status['activity'][0]
            await self._send_message_safe(channel_name, f"☠️ o7:{o7_status} word:{word_status} | {channel_count} streams | {status['total_o7s']} o7s | Hot: {top['name']} ({top['count']}/{top['threshold']})")
        else:
            await self._send_message_safe(channel_name, f"☠️ o7:{o7_status} word:{word_status} | {channel_count} streams | {status['total_o7s']} o7s | No activity")
    
    async def cmd_o7_scale(self, channel_name, content):
        parts = content.split()
        if len(parts) == 2:
            scale = O7_CONFIG["threshold_scale"]
            min_t = O7_CONFIG["min_threshold"]
            ex_100 = max(min_t, int(math.sqrt(100) * scale))
            ex_1k = max(min_t, int(math.sqrt(1000) * scale))
            ex_10k = max(min_t, int(math.sqrt(10000) * scale))
            ex_40k = max(min_t, int(math.sqrt(40000) * scale))
            await self._send_message_safe(channel_name, f"☠️ Scale: {scale}x | Min: {min_t} | Thresholds: 100v→{ex_100}, 1k→{ex_1k}, 10k→{ex_10k}, 40k→{ex_40k}")
        elif len(parts) == 3:
            try:
                new_scale = float(parts[2])
                if new_scale <= 0 or new_scale > 10:
                    await self._send_message_safe(channel_name, "Scale must be between 0.1 and 10")
                    return
                O7_CONFIG["threshold_scale"] = new_scale
                min_t = O7_CONFIG["min_threshold"]
                ex_40k = max(min_t, int(math.sqrt(40000) * new_scale))
                await self._send_message_safe(channel_name, f"☠️ Scale set to {new_scale}x | 40k viewers now needs {ex_40k} o7s to trigger")
            except ValueError:
                await self._send_message_safe(channel_name, "Invalid number. Usage: AI o7scale 0.5")
    
    async def cmd_o7_mute(self, channel_name):
        """Mute o7 death announcements (watcher keeps running)."""
        if not O7_CONFIG["announce_enabled"]:
            await self._send_message_safe(channel_name, "☠️ o7 announcements already muted!")
            return
        O7_CONFIG["announce_enabled"] = False
        await self._send_message_safe(channel_name, "☠️ o7 death announcements muted. Watcher still running. Say 'AI UNMUTEO7' to re-enable.")
        logging.info("O7 announcements muted")
    
    async def cmd_o7_unmute(self, channel_name):
        """Unmute o7 death announcements."""
        if O7_CONFIG["announce_enabled"]:
            await self._send_message_safe(channel_name, "☠️ o7 announcements already enabled!")
            return
        O7_CONFIG["announce_enabled"] = True
        await self._send_message_safe(channel_name, "☠️ o7 death announcements enabled!")
        logging.info("O7 announcements unmuted")
    
    # =========================================================================
    # WORD WATCHER COMMANDS
    # =========================================================================
    
    async def cmd_word_start(self, channel_name, keywords=None):
        """Start the word watcher (piggybacks on o7 watcher)."""
        if keywords:
            self.word_tracker.set_keywords(keywords)
        
        keywords_display = ', '.join(WORD_CONFIG['keywords'])
        
        # Check if already enabled
        if WORD_CONFIG["enabled"]:
            async with self.o7_watcher._channels_lock:
                count = len(self.o7_watcher.connections)
            await self._send_message_safe(channel_name, f"📢 Word watcher already active for: {keywords_display} | Monitoring {count} streams.")
            return
        
        # Word watcher requires channel connections (shared with o7 watcher)
        if not self.o7_watcher.running:
            await self._send_message_safe(channel_name, f"📢 Starting channel monitor + word watcher for: {keywords_display}...")
            # Don't enable o7 announcements - user must explicitly use AI O7 for that
            O7_CONFIG["announce_enabled"] = False
            success, error = await self.o7_watcher.start()
            if not success:
                await self._send_message_safe(channel_name, f"Failed to start: {error}")
                return
            await asyncio.sleep(3)
        
        WORD_CONFIG["enabled"] = True
        async with self.o7_watcher._channels_lock:
            count = len(self.o7_watcher.connections)
        await self._send_message_safe(channel_name, f"📢 Word watcher active for: {keywords_display} | Monitoring {count} streams. Say 'AI STOPWORD' to stop.")
        logging.info(f"Word watcher started for keywords: {keywords_display}")
    
    async def cmd_word_stop(self, channel_name):
        """Stop the word watcher."""
        if not WORD_CONFIG["enabled"]:
            await self._send_message_safe(channel_name, "📢 Word watcher isn't running!")
            return
        
        status = await self.word_tracker.get_status()
        WORD_CONFIG["enabled"] = False
        
        # Only stop the watcher entirely if o7 announcements are also disabled
        if not O7_CONFIG["announce_enabled"]:
            await self.o7_watcher.stop()
            await self._send_message_safe(channel_name, f"📢 Word watcher stopped, channels closed. Total mentions: {status['total_mentions']}")
        else:
            await self._send_message_safe(channel_name, f"📢 Word watcher stopped (o7 still using channels). Total mentions: {status['total_mentions']}")
        logging.info("Word watcher stopped")
    
    async def cmd_word_status(self, channel_name):
        """Show word watcher status."""
        status = await self.word_tracker.get_status()
        
        if not status["enabled"]:
            await self._send_message_safe(channel_name, f"📢 Word watcher is OFF. Say 'AI WORDWATCH' or 'AI WORDWATCH <keywords>' to start!")
            return
        
        async with self.o7_watcher._channels_lock:
            channel_count = len(self.o7_watcher.connections)
        
        pending_info = ""
        if status["pending_channels"]:
            pending_list = [f"{ch}({cnt})" for ch, cnt in list(status["pending_channels"].items())[:3]]
            pending_info = f" | Pending: {', '.join(pending_list)}"
        
        keywords_display = ', '.join(status['keywords']) if status['keywords'] else "(none)"
        
        await self._send_message_safe(
            channel_name, 
            f"📢 Watching: {keywords_display} | {channel_count} streams | "
            f"{status['total_mentions']} mentions | "
            f"Cooldowns: user={status['user_cooldown']}s, ch={status['channel_cooldown']}s{pending_info}"
        )
    
    async def cmd_word_set(self, channel_name, new_keywords):
        """Change the keywords being watched. Accepts comma-separated list."""
        if not new_keywords:
            keywords_display = ', '.join(WORD_CONFIG['keywords'])
            await self._send_message_safe(channel_name, f"📢 Current keywords: {keywords_display}. Usage: AI WORDSET <word1, word2, ...>")
            return
        
        old_keywords = ', '.join(WORD_CONFIG["keywords"])
        self.word_tracker.set_keywords(new_keywords)
        self.word_tracker.reset_stats()
        new_display = ', '.join(WORD_CONFIG["keywords"])
        await self._send_message_safe(channel_name, f"📢 Keywords changed from [{old_keywords}] to [{new_display}]")
        logging.info(f"Word watcher keywords changed: [{old_keywords}] -> [{new_display}]")
    
    async def cmd_word_cooldown(self, channel_name, cooldown_type, value):
        """Set user or channel cooldown."""
        try:
            seconds = int(value)
            if seconds < 0 or seconds > 3600:
                await self._send_message_safe(channel_name, "Cooldown must be 0-3600 seconds")
                return
            
            if cooldown_type == "user":
                WORD_CONFIG["user_cooldown"] = seconds
                await self._send_message_safe(channel_name, f"📢 User cooldown set to {seconds}s")
            elif cooldown_type == "channel":
                WORD_CONFIG["channel_cooldown"] = seconds
                await self._send_message_safe(channel_name, f"📢 Channel cooldown set to {seconds}s")
            else:
                await self._send_message_safe(channel_name, "Usage: AI WORDCOOLDOWN user/channel <seconds>")
        except ValueError:
            await self._send_message_safe(channel_name, "Invalid number. Usage: AI WORDCOOLDOWN user/channel <seconds>")
    
    async def cmd_raffle_start(self, channel_name):
        if self.raffle_active:
            await self._send_message_safe(channel_name, f"🎟️ Raffle already running! {len(self.raffle_entries)} entries so far. Say 'AI winner' to pick!")
            return
        self.raffle_active = True
        self.raffle_entries = set()
        self.raffle_channel = channel_name
        await self._send_message_safe(channel_name, "🎟️ RAFFLE STARTED! Everyone who chats is automatically entered. Say 'AI winner' to pick a winner!")
        logging.info(f"Raffle started in #{channel_name}")
    
    async def cmd_raffle_winner(self, channel_name):
        if not self.raffle_active:
            await self._send_message_safe(channel_name, "No raffle running! Say 'AI raffle' to start one.")
            return
        if len(self.raffle_entries) == 0:
            await self._send_message_safe(channel_name, "🎟️ No entries yet! Need people to chat first.")
            return
        winner = random.choice(list(self.raffle_entries))
        entry_count = len(self.raffle_entries)
        self.raffle_active = False
        self.raffle_channel = None
        await self._send_message_safe(channel_name, f"🎉 WINNER: {winner}! 🎉 (from {entry_count} entries)")
        logging.info(f"Raffle winner: {winner} (from {entry_count} entries)")
        self.raffle_entries = set()
    
    async def cmd_raffle_status(self, channel_name):
        if not self.raffle_active:
            await self._send_message_safe(channel_name, "No raffle running. Say 'AI raffle' to start!")
            return
        await self._send_message_safe(channel_name, f"🎟️ Raffle active with {len(self.raffle_entries)} entries!")
    
    async def cmd_raffle_cancel(self, channel_name):
        if not self.raffle_active:
            await self._send_message_safe(channel_name, "No raffle to cancel!")
            return
        entry_count = len(self.raffle_entries)
        self.raffle_active = False
        self.raffle_entries = set()
        self.raffle_channel = None
        await self._send_message_safe(channel_name, f"🎟️ Raffle cancelled. ({entry_count} entries discarded)")
        logging.info(f"Raffle cancelled with {entry_count} entries")
    
    # =========================================================================
    # CHAT RECORDING COMMANDS
    # =========================================================================
    
    async def cmd_recordchat(self, channel_name, target_channel):
        """Start recording chat from a channel."""
        if not target_channel:
            await self._send_message_safe(channel_name, "📝 Usage: AI RECORDCHAT <streamer>")
            return
        
        target_channel = target_channel.lower().strip().lstrip('@')
        
        success, error = await self.chat_recorder.start(target_channel)
        if success:
            await self._send_message_safe(channel_name, f"📝 Recording chat from #{target_channel}. Say 'AI STOPCHAT' to stop.")
        else:
            await self._send_message_safe(channel_name, f"📝 {error}")
    
    async def cmd_stopchat(self, channel_name):
        """Stop recording chat."""
        success, result = await self.chat_recorder.stop()
        if success:
            recorded_channel, count, duration = result
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            await self._send_message_safe(channel_name, f"📝 Stopped recording #{recorded_channel}. {count} messages in {minutes}m{seconds}s.")
        else:
            await self._send_message_safe(channel_name, f"📝 {result}")
    
    async def cmd_chatstatus(self, channel_name):
        """Check chat recording status."""
        status = await self.chat_recorder.get_status()
        if status["recording"]:
            duration = status["duration"]
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            size_kb = status["file_size"] / 1024
            max_kb = CHAT_RECORD_CONFIG["max_file_size"] / 1024
            await self._send_message_safe(
                channel_name, 
                f"📝 Recording #{status['channel']}: {status['messages']} msgs, {minutes}m{seconds}s, {size_kb:.0f}KB/{max_kb:.0f}KB"
            )
        else:
            await self._send_message_safe(channel_name, "📝 Not recording any chat. Use 'AI RECORDCHAT <streamer>' to start.")
    
    # =========================================================================
    # SIGNUP COMMANDS
    # =========================================================================
    
    def _load_signups(self):
        if os.path.exists(SIGNUP_FILE):
            try:
                with open(SIGNUP_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load signups: {e}")
        return {}
    
    def _save_signups(self):
        try:
            with open(SIGNUP_FILE, 'w') as f:
                json.dump(self.signups, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save signups: {e}")
    
    def _load_banned_channels(self):
        if os.path.exists(BANNED_CHANNELS_FILE):
            try:
                with open(BANNED_CHANNELS_FILE, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logging.error(f"Failed to load banned channels: {e}")
        return set()
    
    def _save_banned_channels(self):
        try:
            with open(BANNED_CHANNELS_FILE, 'w') as f:
                json.dump(list(self.banned_channels), f)
        except Exception as e:
            logging.error(f"Failed to save banned channels: {e}")
    
    def _load_banned_users(self):
        if os.path.exists(BANNED_USERS_FILE):
            try:
                with open(BANNED_USERS_FILE, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logging.error(f"Failed to load banned users: {e}")
        return set()
    
    def _save_banned_users(self):
        try:
            with open(BANNED_USERS_FILE, 'w') as f:
                json.dump(list(self.banned_users), f)
        except Exception as e:
            logging.error(f"Failed to save banned users: {e}")
    
    async def cmd_signup(self, author, display_name, channel_name, signup_text):
        # Check if user is banned from using bot commands
        if author in self.banned_users:
            await self._send_message_safe(channel_name, f"@{display_name} ⛔ You are not allowed to use this command.")
            return
        
        if not signup_text:
            await self._send_message_safe(channel_name, f"@{display_name} Usage: AI SIGNUP <class, spec, availability, etc>")
            return
        
        is_new = author not in self.signups
        self.signups[author] = {
            "display_name": display_name,
            "message": signup_text,
            "timestamp": time.time(),
            "channel": channel_name
        }
        self._save_signups()
        
        count = len(self.signups)
        url = f"onlyfangs3.com/user/{author}"
        if is_new:
            await self._send_message_safe(channel_name, f"@{display_name} ✅ Signed up! {url}")
        else:
            await self._send_message_safe(channel_name, f"@{display_name} ✅ Updated! {url}")
        logging.info(f"Signup: {display_name} - {signup_text[:50]}")
    
    async def cmd_roll(self, author, display_name, channel_name, args):
        """Roll a random number. Default 1-100, or custom range."""
        # Check if user is banned from using bot commands
        if author in self.banned_users:
            await self._send_message_safe(channel_name, f"@{display_name} ⛔ You are not allowed to use this command.")
            return
        
        min_val = 1
        max_val = 100
        
        if args:
            args = args.strip()
            # Check for range format: "20-56"
            if '-' in args:
                parts = args.split('-', 1)
                try:
                    min_val = int(parts[0].strip())
                    max_val = int(parts[1].strip())
                except ValueError:
                    await self._send_message_safe(channel_name, f"@{display_name} Invalid range! Use: AI ROLL 20-56")
                    return
            else:
                # Single number = max value
                try:
                    max_val = int(args)
                except ValueError:
                    await self._send_message_safe(channel_name, f"@{display_name} Invalid number! Use: AI ROLL 200 or AI ROLL 20-56")
                    return
        
        # Validate range
        if min_val > max_val:
            min_val, max_val = max_val, min_val  # Swap if backwards
        
        if min_val == max_val:
            await self._send_message_safe(channel_name, f"@{display_name} 🎲 rolled {min_val} (wow, what were the odds?)")
            return
        
        result = random.randint(min_val, max_val)
        await self._send_message_safe(channel_name, f"@{display_name} 🎲 rolled {result} ({min_val}-{max_val})")
        logging.info(f"Roll: {display_name} rolled {result} ({min_val}-{max_val})")
    
    async def cmd_ai_prompt(self, channel_name, prompt):
        if not prompt:
            await self._send_message_safe(channel_name, "Need a prompt after AI!")
            return
        queue_size = self.request_queue.qsize()
        if queue_size == 0:
            await self._send_message_safe(channel_name, "🤔 thinking...")
        else:
            await self._send_message_safe(channel_name, f"📋 Queued! ({queue_size} ahead)")
        await self.request_queue.put((channel_name, prompt))

    # =========================================================================
    # MESSAGE ROUTER
    # =========================================================================
    
    async def event_message(self, message):
        if message.echo:
            return
        if message.author is None:
            return
        
        content = message.content.strip()
        author = message.author.name.lower()
        channel_name = message.channel.name
        
        logging.debug(f"Message from {author} in #{channel_name}: {content[:50]}")
        
        # Track users for raffle
        if self.raffle_active and channel_name == self.raffle_channel:
            self.raffle_entries.add(message.author.name)
            logging.debug(f"Raffle: Added {message.author.name} (total: {len(self.raffle_entries)})")
        
        upper = content.upper()
        
        # Public commands (anyone can use)
        if upper.startswith('AI SIGNUP '):
            signup_text = content[10:].strip()
            await self.cmd_signup(author, message.author.name, channel_name, signup_text)
            return
        
        if upper.startswith('AI ROLL'):
            args = content[7:].strip() if len(content) > 7 else ""
            await self.cmd_roll(author, message.author.name, channel_name, args)
            return
        
        # Only process remaining commands from authorized user
        if author != AUTHORIZED_USER:
            return
        
        # Basic commands
        if upper == 'OF':
            await self.cmd_of(channel_name)
        
        # Navigation commands
        elif upper == 'AI HOME':
            await self.cmd_home(channel_name)
        elif upper.startswith('AI GOTO '):
            new_channel = content[8:].strip().lower().lstrip('@')
            # Handle twitch.tv URLs (twitch.tv/user, https://twitch.tv/user, etc.)
            if 'twitch.tv/' in new_channel:
                new_channel = new_channel.split('twitch.tv/')[-1].split('/')[0].split('?')[0]
            await self.cmd_goto(channel_name, new_channel)
        elif upper == 'AI BANNED':
            await self.cmd_banned(channel_name)
        elif upper.startswith('AI UNBAN '):
            target_channel = content[9:].strip().lower().lstrip('@')
            # Handle twitch.tv URLs
            if 'twitch.tv/' in target_channel:
                target_channel = target_channel.split('twitch.tv/')[-1].split('/')[0].split('?')[0]
            await self.cmd_unban(channel_name, target_channel)
        
        # User ban commands
        elif upper.startswith('AI BAN '):
            target_user = content[7:].strip()
            await self.cmd_ban_user(channel_name, target_user)
        elif upper.startswith('AI UNBANUSER '):
            target_user = content[13:].strip()
            await self.cmd_unban_user(channel_name, target_user)
        elif upper == 'AI BANLIST':
            await self.cmd_banlist(channel_name)
        
        # Listen command
        elif upper.startswith('AI LISTENTO '):
            target_channel = content[12:].strip().lower().lstrip('@')
            # Handle twitch.tv URLs
            if 'twitch.tv/' in target_channel:
                target_channel = target_channel.split('twitch.tv/')[-1].split('/')[0].split('?')[0]
            await self.cmd_listento(channel_name, target_channel)
        
        # Vision command (new!)
        elif upper.startswith('AI LOOKAT '):
            args = content[10:].strip()
            await self.cmd_lookat(channel_name, args)
        
        # Recording commands (clip, audio, gif)
        elif upper.startswith('AI CLIP'):
            args = content[7:].strip() if len(content) > 7 else ""
            await self.cmd_clip(channel_name, args)
        elif upper.startswith('AI AUDIO'):
            args = content[8:].strip() if len(content) > 8 else ""
            await self.cmd_audio(channel_name, args)
        elif upper.startswith('AI GIF'):
            args = content[6:].strip() if len(content) > 6 else ""
            await self.cmd_gif(channel_name, args)
        
        # O7 commands
        elif upper == 'AI O7':
            await self.cmd_o7_start(channel_name)
        elif upper == 'AI STOPO7':
            await self.cmd_o7_stop(channel_name)
        elif upper == 'AI O7STATUS':
            await self.cmd_o7_status(channel_name)
        elif upper.startswith('AI O7SCALE'):
            await self.cmd_o7_scale(channel_name, content)
        elif upper == 'AI MUTEO7':
            await self.cmd_o7_mute(channel_name)
        elif upper == 'AI UNMUTEO7':
            await self.cmd_o7_unmute(channel_name)
        
        # Word Watcher commands
        elif upper == 'AI WORDWATCH':
            await self.cmd_word_start(channel_name)
        elif upper.startswith('AI WORDWATCH '):
            keyword = content[13:].strip()
            await self.cmd_word_start(channel_name, keyword)
        elif upper == 'AI STOPWORD':
            await self.cmd_word_stop(channel_name)
        elif upper == 'AI WORDSTATUS':
            await self.cmd_word_status(channel_name)
        elif upper.startswith('AI WORDSET '):
            new_keyword = content[11:].strip()
            await self.cmd_word_set(channel_name, new_keyword)
        elif upper.startswith('AI WORDCOOLDOWN '):
            parts = content[16:].strip().split()
            if len(parts) >= 2:
                await self.cmd_word_cooldown(channel_name, parts[0].lower(), parts[1])
            else:
                await self._send_message_safe(channel_name, "Usage: AI WORDCOOLDOWN user/channel <seconds>")
        
        # Raffle commands
        elif upper == 'AI RAFFLE':
            await self.cmd_raffle_start(channel_name)
        elif upper == 'AI WINNER':
            await self.cmd_raffle_winner(channel_name)
        elif upper == 'AI RAFFLE STATUS':
            await self.cmd_raffle_status(channel_name)
        elif upper == 'AI CANCEL RAFFLE':
            await self.cmd_raffle_cancel(channel_name)
        
        # Chat recording commands
        elif upper.startswith('AI RECORDCHAT '):
            target = content[14:].strip().lower().lstrip('@')
            # Handle twitch.tv URLs
            if 'twitch.tv/' in target:
                target = target.split('twitch.tv/')[-1].split('/')[0].split('?')[0]
            await self.cmd_recordchat(channel_name, target)
        elif upper == 'AI STOPCHAT':
            await self.cmd_stopchat(channel_name)
        elif upper == 'AI CHATSTATUS':
            await self.cmd_chatstatus(channel_name)
        
        # AI prompt (catch-all, must be last)
        elif upper.startswith('AI '):
            prompt = content[3:].strip()
            await self.cmd_ai_prompt(channel_name, prompt)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    token_manager = TokenManager(CLIENT_ID, REFRESH_TOKEN, TOKEN_FILE, ACCESS_TOKEN, EXPIRES_AT)
    try:
        token = await token_manager.get_token()
        logging.info("Got valid access token")
    except Exception as e:
        logging.error(f"Failed to get token: {e}")
        raise
    bot = Bot(token, token_manager)
    await bot.start()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Bot crashed: {e}")
        import traceback
        logging.error(traceback.format_exc())
