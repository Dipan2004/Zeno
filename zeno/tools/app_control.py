"""
ZENO App Control - Rule-Based Application & URL Launching + OS Control

This is NOT intelligence - it is pure OS glue with explicit mappings.

Phase 5 Updates:
- Volume control (set, adjust, mute, unmute)
- Brightness control (laptops only, graceful desktop failure)
- Last volume caching for smart unmute
- Brightness availability caching (one-time warning)

Safety Rules:
- Explicit app registry only
- Try commands in order
- Fail cleanly if not found
- Cache results for speed
"""

import logging
import platform
import subprocess
import webbrowser
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class AppNotFoundError(Exception):
    """Raised when requested application is not found"""
    pass


class AppControlError(Exception):
    """Base exception for app control errors"""
    pass


class BrightnessNotAvailableError(AppControlError):
    """Raised when brightness control is not available (desktop monitors)"""
    pass


# ============================================================================
# GLOBAL STATE FOR OS CONTROL
# ============================================================================

# Volume state for smart unmute
_last_nonzero_volume = 50  # Default fallback

# Brightness availability cache
_brightness_available = None  # None = unknown, True/False = cached


# ============================================================================
# APP REGISTRY - Windows Applications
# ============================================================================

APP_REGISTRY = {
    # Development Tools
    "vscode": {
        "display": "Visual Studio Code",
        "aliases": ["vscode", "vs code", "visual studio code", "code"],
        "commands": ["code"],
        "paths": [
            r"C:\Program Files\Microsoft VS Code\Code.exe",
            r"C:\Users\{user}\AppData\Local\Programs\Microsoft VS Code\Code.exe"
        ],
        "shell_required": True  # 'code' is a batch file on Windows
    },
    
    # Browsers
    "edge": {
        "display": "Microsoft Edge",
        "aliases": ["edge", "microsoft edge"],
        "commands": ["msedge"],
        "paths": [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
    },
    "chrome": {
        "display": "Google Chrome",
        "aliases": ["chrome", "google chrome"],
        "commands": ["chrome"],
        "paths": [r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    },
    "firefox": {
        "display": "Firefox",
        "aliases": ["firefox", "mozilla firefox"],
        "commands": ["firefox"],
        "paths": [r"C:\Program Files\Mozilla Firefox\firefox.exe"]
    },
    
    # Windows System Apps
    "explorer": {
        "display": "File Explorer",
        "aliases": ["explorer", "file explorer", "files", "file manager"],
        "commands": ["explorer"],
        "shell_required": False
    },
    "notepad": {
        "display": "Notepad",
        "aliases": ["notepad"],
        "commands": ["notepad"],
        "shell_required": False
    },
    "calc": {
        "display": "Calculator",
        "aliases": ["calculator", "calc"],
        "commands": ["calc"],
        "shell_required": False
    },
    "paint": {
        "display": "Paint",
        "aliases": ["paint", "mspaint"],
        "commands": ["mspaint"],
        "shell_required": False
    },
    "cmd": {
        "display": "Command Prompt",
        "aliases": ["cmd", "command prompt"],
        "commands": ["cmd"],
        "shell_required": False
    },
    "powershell": {
        "display": "PowerShell",
        "aliases": ["powershell", "pwsh"],
        "commands": ["powershell"],
        "shell_required": False
    },
    "terminal": {
        "display": "Windows Terminal",
        "aliases": ["terminal", "windows terminal", "wt"],
        "commands": ["wt"],
        "shell_required": False
    },
    "taskmgr": {
        "display": "Task Manager",
        "aliases": ["task manager", "taskmgr"],
        "commands": ["taskmgr"],
        "shell_required": False
    },
    "control": {
        "display": "Control Panel",
        "aliases": ["control panel", "control"],
        "commands": ["control"],
        "shell_required": False
    },
    "settings": {
        "display": "Windows Settings",
        "aliases": ["settings", "windows settings"],
        "commands": ["start ms-settings:"],
        "shell_required": True
    },
    "camera": {
        "display": "Camera",
        "aliases": ["camera", "webcam"],
        "commands": ["explorer.exe camera:"],
        "shell_required": True
    },
    "photos": {
        "display": "Photos",
        "aliases": ["photos", "pictures"],
        "commands": ["start ms-photos:"],
        "shell_required": True
    },
    "snipping": {
        "display": "Snipping Tool",
        "aliases": ["snipping tool", "snip", "screenshot"],
        "commands": ["snippingtool"],
        "shell_required": False
    },
    "smart notebook": {
        "display": "Smart Notebook",
        "aliases": ["smart notebook", "smartboard"],
        "commands": ["smart notebook"],
        "shell_required": False
    },
    
    # Office Apps (if installed)
    "word": {
        "display": "Microsoft Word",
        "aliases": ["word", "microsoft word"],
        "commands": ["winword"],
        "paths": [r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"]
    },
    "excel": {
        "display": "Microsoft Excel",
        "aliases": ["excel", "microsoft excel"],
        "commands": ["excel"],
        "paths": [r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"]
    },
    "powerpoint": {
        "display": "Microsoft PowerPoint",
        "aliases": ["powerpoint", "microsoft powerpoint", "ppt"],
        "commands": ["powerpnt"],
        "paths": [r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"]
    },
    
    # Dev Tools
    "git": {
        "display": "Git Bash",
        "aliases": ["git", "git bash"],
        "commands": ["git"],
        "paths": [r"C:\Program Files\Git\git-bash.exe"]
    },
    "python": {
        "display": "Python",
        "aliases": ["python", "python3"],
        "commands": ["python", "python3"],
        "shell_required": False
    },
    "node": {
        "display": "Node.js",
        "aliases": ["node", "nodejs"],
        "commands": ["node"],
        "shell_required": False
    },
    
    # Notepad++
    "notepad++": {
        "display": "Notepad++",
        "aliases": ["notepad++", "notepad plus plus", "npp"],
        "commands": ["notepad++"],
        "paths": [r"C:\Program Files\Notepad++\notepad++.exe"]
    },

    "whatsapp": {
        "display": "WhatsApp",
        "aliases": ["whatsapp", "whatsapp desktop"],
        "commands": ["whatsapp"],
        "paths": [r"C:\Program Files\WhatsApp\WhatsApp.exe"]
    },
    "discord": {
        "display": "Discord",
        "aliases": ["discord", "discord desktop"],
        "commands": ["discord"],
        "paths": [r"C:\Program Files\Discord\Discord.exe"]
    },
}


# Common websites (rule-based URL shortcuts)
COMMON_SITES = {
    "github": "https://github.com",
    "google": "https://google.com",
    "youtube": "https://youtube.com",
    "stackoverflow": "https://stackoverflow.com",
    "twitter": "https://twitter.com",
    "reddit": "https://reddit.com",
    "gmail": "https://gmail.com",
    "whatsapp": "https://web.whatsapp.com/",
    "discord": "https://discord.com/"
}


# Cache for resolved app paths (lazy loaded)
_app_cache: Dict[str, Optional[str]] = {}


def get_platform() -> str:
    """Get current platform name"""
    return platform.system()


# ============================================================================
# VOLUME CONTROL (Phase 5)
# ============================================================================

def set_volume(level: int) -> bool:
    """
    Set system volume.
    
    Args:
        level: Volume level 0-100
        
    Returns:
        True if successful
        
    Raises:
        ValueError: If level out of range
        AppControlError: If volume control fails
    """
    global _last_nonzero_volume
    
    if not 0 <= level <= 100:
        raise ValueError("Volume must be 0-100")
    
    # Store non-zero volumes for smart unmute
    if level > 0:
        _last_nonzero_volume = level
    
    try:
        if get_platform() == "Windows":
            # Use pycaw (comtypes wrapper)
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Set master volume (0.0 to 1.0)
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)
            
            logger.info(f"Set volume to {level}%")
            return True
        else:
            # macOS/Linux fallback
            raise NotImplementedError("Volume control only supported on Windows in Phase 5")
            
    except Exception as e:
        logger.error(f"Failed to set volume: {e}", exc_info=True)
        raise AppControlError(f"Failed to set volume: {e}") from e


def get_volume() -> int:
    """
    Get current system volume.
    
    Returns:
        Current volume level 0-100
        
    Raises:
        AppControlError: If getting volume fails
    """
    try:
        if get_platform() == "Windows":
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Get master volume (0.0 to 1.0)
            current = volume.GetMasterVolumeLevelScalar()
            level = int(current * 100)
            
            logger.debug(f"Current volume: {level}%")
            return level
        else:
            raise NotImplementedError("Volume control only supported on Windows in Phase 5")
            
    except Exception as e:
        logger.error(f"Failed to get volume: {e}", exc_info=True)
        raise AppControlError(f"Failed to get volume: {e}") from e


def adjust_volume(delta: int) -> bool:
    """
    Adjust volume by delta.
    
    Args:
        delta: Amount to adjust (+/- 10 typical)
        
    Returns:
        True if successful
        
    Raises:
        AppControlError: If adjustment fails
    """
    current = get_volume()
    new_level = max(0, min(100, current + delta))
    return set_volume(new_level)


def mute_volume() -> bool:
    """
    Mute system volume (set to 0).
    
    Returns:
        True if successful
    """
    return set_volume(0)


def unmute_volume() -> bool:
    """
    Unmute system volume (restore last non-zero level).
    
    Returns:
        True if successful
    """
    global _last_nonzero_volume
    return set_volume(_last_nonzero_volume)


# ============================================================================
# BRIGHTNESS CONTROL (Phase 5)
# ============================================================================

def set_brightness(level: int) -> bool:
    """
    Set screen brightness (laptops only).
    
    Args:
        level: Brightness level 0-100
        
    Returns:
        True if successful
        
    Raises:
        ValueError: If level out of range
        BrightnessNotAvailableError: If brightness control not available
        AppControlError: If brightness control fails
    """
    global _brightness_available
    
    if not 0 <= level <= 100:
        raise ValueError("Brightness must be 0-100")
    
    # Check cache first
    if _brightness_available is False:
        # Already know it's not available, fail silently
        logger.debug("Brightness control cached as unavailable")
        raise BrightnessNotAvailableError("Brightness control not available (cached)")
    
    try:
        if get_platform() == "Windows":
            import wmi
            
            c = wmi.WMI(namespace='wmi')
            methods = c.WmiMonitorBrightnessMethods()[0]
            methods.WmiSetBrightness(level, 0)
            
            # Success - cache as available
            _brightness_available = True
            logger.info(f"Set brightness to {level}%")
            return True
        else:
            raise NotImplementedError("Brightness control only supported on Windows in Phase 5")
            
    except Exception as e:
        # First failure - cache as unavailable and raise user-facing error
        if _brightness_available is None:
            _brightness_available = False
            logger.warning(f"Brightness control not available: {e}")
            # This will trigger spoken warning (see SystemAgent)
            raise BrightnessNotAvailableError(
                "Brightness control isn't available on this display"
            ) from e
        else:
            # Subsequent failures - silent (already cached)
            logger.debug(f"Brightness control unavailable (cached): {e}")
            raise BrightnessNotAvailableError("Brightness control unavailable") from e


def get_brightness() -> int:
    """
    Get current screen brightness.
    
    Returns:
        Current brightness level 0-100
        
    Raises:
        BrightnessNotAvailableError: If brightness control not available
        AppControlError: If getting brightness fails
    """
    global _brightness_available
    
    # Check cache
    if _brightness_available is False:
        raise BrightnessNotAvailableError("Brightness control not available (cached)")
    
    try:
        if get_platform() == "Windows":
            import wmi
            
            c = wmi.WMI(namespace='wmi')
            brightness_methods = c.WmiMonitorBrightness()
            
            if brightness_methods:
                current = brightness_methods[0].CurrentBrightness
                logger.debug(f"Current brightness: {current}%")
                return current
            else:
                raise BrightnessNotAvailableError("No brightness methods available")
        else:
            raise NotImplementedError("Brightness control only supported on Windows in Phase 5")
            
    except Exception as e:
        if _brightness_available is None:
            _brightness_available = False
        logger.error(f"Failed to get brightness: {e}", exc_info=True)
        raise BrightnessNotAvailableError(f"Failed to get brightness: {e}") from e


def adjust_brightness(delta: int) -> bool:
    """
    Adjust brightness by delta.
    
    Args:
        delta: Amount to adjust (+/- 10 typical)
        
    Returns:
        True if successful
        
    Raises:
        BrightnessNotAvailableError: If brightness control not available
    """
    current = get_brightness()
    new_level = max(0, min(100, current + delta))
    return set_brightness(new_level)


# ============================================================================
# APP LAUNCHING (Existing Phase 4 code)
# ============================================================================

def normalize_app_name(name: str) -> Optional[str]:
    """
    Normalize app name using alias mapping.
    
    Args:
        name: User-provided app name
        
    Returns:
        Canonical app key or None if not found
    """
    normalized = name.lower().strip()
    
    # Direct match
    if normalized in APP_REGISTRY:
        return normalized
    
    # Check aliases
    for app_key, app_info in APP_REGISTRY.items():
        if "aliases" in app_info and normalized in app_info["aliases"]:
            return app_key
    
    return None


def resolve_app_path(app_key: str) -> Optional[str]:
    """
    Resolve application executable path.
    
    Uses lazy caching for performance.
    
    Args:
        app_key: Canonical app key from APP_REGISTRY
        
    Returns:
        Resolved path/command or None if not found
    """
    # Check cache first
    if app_key in _app_cache:
        return _app_cache[app_key]
    
    app_info = APP_REGISTRY.get(app_key)
    if not app_info:
        return None
    
    # For shell commands (like "start ms-camera:"), return directly
    # Don't try shutil.which() - it doesn't work for shell commands
    if app_info.get("shell_required", False):
        commands = app_info.get("commands", [])
        if commands:
            cmd = commands[0]  # Use first command
            logger.info(f"Resolved {app_key} as shell command: {cmd}")
            _app_cache[app_key] = cmd
            return cmd
    
    # Try commands via PATH
    for cmd in app_info.get("commands", []):
        resolved = shutil.which(cmd)
        if resolved:
            logger.info(f"Resolved {app_key} via PATH: {cmd}")
            _app_cache[app_key] = cmd
            return cmd
    
    # Try known paths
    for path_str in app_info.get("paths", []):
        # Replace {user} with actual home directory
        path_str = path_str.replace("{user}", Path.home().name)
        path = Path(path_str)
        
        if path.exists():
            logger.info(f"Resolved {app_key} via path: {path}")
            _app_cache[app_key] = str(path)
            return str(path)
    
    # Not found
    logger.warning(f"Could not resolve {app_key}")
    _app_cache[app_key] = None
    return None


def open_app(app_name: str) -> bool:
    """
    Open an application by name.
    
    Args:
        app_name: Application name (supports aliases)
        
    Returns:
        True if application was launched successfully
        
    Raises:
        AppNotFoundError: If application is not in registry or not found
        AppControlError: If launch fails
    """
    if not app_name:
        raise ValueError("App name cannot be empty")
    
    # Normalize app name
    app_key = normalize_app_name(app_name)
    if not app_key:
        available = ", ".join(sorted(set(
            alias
            for info in APP_REGISTRY.values()
            for alias in info.get("aliases", [])
        ))[:10])
        raise AppNotFoundError(
            f"I don't recognize '{app_name}' as an available application.\n"
            f"Try: {available}..."
        )
    
    # Get app info
    app_info = APP_REGISTRY[app_key]
    display_name = app_info.get("display", app_key)
    
    # Resolve path/command
    resolved = resolve_app_path(app_key)
    if not resolved:
        raise AppNotFoundError(
            f"{display_name} is not installed or not found in PATH.\n"
            f"Install it or check your PATH configuration."
        )
    
    # Check if shell is required
    shell_required = app_info.get("shell_required", False)
    
    try:
        logger.info(f"Launching {display_name}: {resolved}")
        
        # Launch application
        if get_platform() == "Windows":
            if shell_required:
                # Use shell for special commands
                subprocess.Popen(
                    resolved,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Direct execution (safer)
                subprocess.Popen(
                    [resolved],
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
        else:
            # macOS/Linux
            subprocess.Popen(
                [resolved],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        logger.info(f"Successfully launched {display_name}")
        return True
        
    except FileNotFoundError:
        raise AppNotFoundError(
            f"{display_name} command not found: {resolved}"
        )
    except Exception as e:
        logger.error(f"Failed to launch {display_name}: {e}", exc_info=True)
        raise AppControlError(f"Failed to launch {display_name}: {e}") from e


def open_url(url: str) -> bool:
    """
    Open a URL in the default browser.
    
    Args:
        url: URL to open (must be http:// or https://)
        
    Returns:
        True if URL was opened successfully
        
    Raises:
        ValueError: If URL is invalid
        AppControlError: If browser launch fails
    """
    if not url:
        raise ValueError("URL cannot be empty")
    
    # Check if it's a common site keyword
    url_lower = url.lower().strip()
    if url_lower in COMMON_SITES:
        url = COMMON_SITES[url_lower]
        logger.info(f"Resolved site keyword '{url_lower}' to {url}")
    
    # Validate URL scheme
    if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
        raise ValueError(
            f"Invalid URL: {url}. Only http:// and https:// are supported."
        )
    
    try:
        logger.info(f"Opening URL: {url}")
        success = webbrowser.open(url)
        
        if success:
            logger.info(f"Successfully opened URL: {url}")
            return True
        else:
            raise AppControlError(f"Browser returned failure for URL: {url}")
        
    except Exception as e:
        logger.error(f"Failed to open URL {url}: {e}", exc_info=True)
        raise AppControlError(f"Failed to open URL '{url}': {e}") from e


def is_url(target: str) -> bool:
    """Check if target string looks like a URL or site keyword"""
    target_lower = target.lower().strip()
    return (
        target_lower.startswith("http://") or
        target_lower.startswith("https://") or
        target_lower.startswith("www.") or
        target_lower in COMMON_SITES
    )


def get_available_apps() -> List[str]:
    """
    Get list of known application names (aliases).
    
    Returns:
        Sorted list of all app aliases
    """
    aliases = set()
    for app_info in APP_REGISTRY.values():
        aliases.update(app_info.get("aliases", []))
    return sorted(aliases)


def get_available_sites() -> List[str]:
    """
    Get list of known site keywords.
    
    Returns:
        Sorted list of site keywords
    """
    return sorted(COMMON_SITES.keys())


def clear_cache():
    """Clear the app path cache (useful for testing)"""
    global _app_cache
    _app_cache.clear()
    logger.debug("App cache cleared")