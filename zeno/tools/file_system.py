"""
ZENO File System - Safe File Operations

ZENO may ONLY:
- Create directories in workspace
- Create new files in workspace
- Write content to files it created
- ONLY when explicitly requested

ZENO MUST NOT:
- Modify existing files silently
- Delete files
- Read arbitrary system files
- Traverse user directories

All operations are restricted to the workspace directory (ZENO_ROOT/workspace)
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FileSystemError(Exception):
    """Base exception for file system errors"""
    pass


class FileExistsError(FileSystemError):
    """Raised when trying to create a file that already exists"""
    pass


class InvalidPathError(FileSystemError):
    """Raised when path is outside allowed workspace"""
    pass


class InvalidExtensionError(FileSystemError):
    """Raised when file extension is not allowed"""
    pass


class FileTooLargeError(FileSystemError):
    """Raised when file content exceeds size limit"""
    pass


# Workspace configuration
# Set workspace relative to ZENO project root
ZENO_ROOT = Path(__file__).parent.parent.parent  # Navigate from zeno/tools/file_system.py to root
BASE_DIR = ZENO_ROOT / "workspace"

# Allowed file extensions (whitelist)
ALLOWED_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".js",
    ".html",
    ".css",
    ".yaml",
    ".yml",
    ".xml",
    ".cpp"
}

# Maximum file size (200 KB)
MAX_FILE_SIZE = 200 * 1024  # bytes


def ensure_workspace_exists():
    """
    Ensure workspace directory exists.
    
    Creates ~/zeno_workspace if it doesn't exist.
    """
    if not BASE_DIR.exists():
        logger.info(f"Creating workspace directory: {BASE_DIR}")
        BASE_DIR.mkdir(parents=True, exist_ok=True)
    else:
        logger.debug(f"Workspace exists: {BASE_DIR}")


def validate_path(file_path: Path) -> Path:
    """
    Validate that path is within allowed workspace.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Resolved absolute path
        
    Raises:
        InvalidPathError: If path is outside workspace
    """
    try:
        # Resolve to absolute path
        resolved = file_path.resolve()
        
        # Check if it's within workspace
        if not str(resolved).startswith(str(BASE_DIR.resolve())):
            raise InvalidPathError(
                f"Path must be within workspace: {BASE_DIR}"
            )
        
        return resolved
        
    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        raise InvalidPathError(f"Invalid path: {e}") from e


def validate_extension(file_path: Path):
    """
    Validate file extension is allowed.
    
    Args:
        file_path: Path to check
        
    Raises:
        InvalidExtensionError: If extension not allowed
    """
    extension = file_path.suffix.lower()
    
    if not extension:
        raise InvalidExtensionError(
            "File must have an extension. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidExtensionError(
            f"Extension '{extension}' not allowed. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )


def validate_content_size(content: str):
    """
    Validate content size is within limits.
    
    Args:
        content: Content to validate
        
    Raises:
        FileTooLargeError: If content exceeds size limit
    """
    size = len(content.encode('utf-8'))
    
    if size > MAX_FILE_SIZE:
        raise FileTooLargeError(
            f"Content too large: {size} bytes (max: {MAX_FILE_SIZE} bytes)"
        )


def create_file(
    filename: str,
    content: str,
    subdirectory: Optional[str] = None,
    overwrite: bool = False
) -> Path:
    """
    Create a new file in the workspace.
    
    Args:
        filename: Name of file to create (e.g., "script.py")
        content: Text content to write
        subdirectory: Optional subdirectory within workspace
        overwrite: If True, allow overwriting existing files (default: False)
        
    Returns:
        Path to created file
        
    Raises:
        FileExistsError: If file exists and overwrite=False
        InvalidExtensionError: If file extension not allowed
        FileTooLargeError: If content exceeds size limit
        InvalidPathError: If path is invalid
        FileSystemError: If creation fails
        
    Example:
        >>> path = create_file("hello.py", "print('Hello')")
        >>> print(path)
        /home/user/zeno_workspace/hello.py
    """
    # Ensure workspace exists
    ensure_workspace_exists()
    
    # Build file path
    if subdirectory:
        file_path = BASE_DIR / subdirectory / filename
    else:
        file_path = BASE_DIR / filename
    
    # Validate path is in workspace
    file_path = validate_path(file_path)
    
    # Validate extension
    validate_extension(file_path)
    
    # Validate content size
    validate_content_size(content)
    
    # Check if file exists
    if file_path.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {file_path.name}. "
            f"Use a different name or allow overwrite."
        )
    
    try:
        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file (text mode, UTF-8)
        logger.info(f"Creating file: {file_path}")
        file_path.write_text(content, encoding='utf-8')
        
        logger.info(f"Successfully created: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to create file {filename}: {e}", exc_info=True)
        raise FileSystemError(f"Failed to create file: {e}") from e


def create_directory(
    dirname: str,
    parent: Optional[str] = None
) -> Path:
    """
    Create a directory in the workspace.
    
    Args:
        dirname: Name of directory to create
        parent: Optional parent directory within workspace
        
    Returns:
        Path to created directory
        
    Raises:
        InvalidPathError: If path is invalid
        FileSystemError: If creation fails
    """
    # Ensure workspace exists
    ensure_workspace_exists()
    
    # Build directory path
    if parent:
        dir_path = BASE_DIR / parent / dirname
    else:
        dir_path = BASE_DIR / dirname
    
    # Validate path is in workspace
    dir_path = validate_path(dir_path)
    
    try:
        logger.info(f"Creating directory: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Successfully created: {dir_path}")
        return dir_path
        
    except Exception as e:
        logger.error(f"Failed to create directory {dirname}: {e}", exc_info=True)
        raise FileSystemError(f"Failed to create directory: {e}") from e


def get_workspace_path() -> Path:
    """
    Get the workspace base directory path.
    
    Returns:
        Path to workspace
    """
    return BASE_DIR


def list_workspace_files(
    subdirectory: Optional[str] = None,
    extension: Optional[str] = None
) -> list[Path]:
    """
    List files in workspace.
    
    Args:
        subdirectory: Optional subdirectory to list
        extension: Optional extension filter (e.g., '.py')
        
    Returns:
        List of file paths
    """
    if not BASE_DIR.exists():
        return []
    
    # Determine which directory to list
    if subdirectory:
        list_dir = BASE_DIR / subdirectory
        if not list_dir.exists():
            return []
    else:
        list_dir = BASE_DIR
    
    # List files
    files = []
    for item in list_dir.iterdir():
        if item.is_file():
            if extension is None or item.suffix.lower() == extension.lower():
                files.append(item)
    
    return sorted(files)


def file_exists(filename: str, subdirectory: Optional[str] = None) -> bool:
    """
    Check if a file exists in workspace.
    
    Args:
        filename: Name of file to check
        subdirectory: Optional subdirectory
        
    Returns:
        True if file exists
    """
    if subdirectory:
        file_path = BASE_DIR / subdirectory / filename
    else:
        file_path = BASE_DIR / filename
    
    return file_path.exists() and file_path.is_file()


def get_workspace_stats() -> dict:
    """
    Get workspace statistics.
    
    Returns:
        Dict with file counts and sizes
    """
    if not BASE_DIR.exists():
        return {
            'exists': False,
            'total_files': 0,
            'total_size': 0
        }
    
    total_files = 0
    total_size = 0
    
    for item in BASE_DIR.rglob('*'):
        if item.is_file():
            total_files += 1
            total_size += item.stat().st_size
    
    return {
        'exists': True,
        'path': str(BASE_DIR),
        'total_files': total_files,
        'total_size': total_size,
        'total_size_mb': round(total_size / (1024 * 1024), 2)
    }