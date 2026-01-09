import re
from urllib.parse import urlparse


def is_valid_url(url:str)->bool:
    """Check if the given string is a valid URL"""
    if not url or not isinstance(url,str):
        return False
    
    parsed=urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    #Checking by regex
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
        r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # ...or TLD
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or IPv4
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return re.match(regex,url) is not None
def is_non_empty_string(s:str)->bool:
    return isinstance(s,str) and bool(s.strip())


def is_valid_date_string(date_str: str, date_format: str = "%Y-%m-%d") -> bool:
    """
    Check if a string is a valid date.
    """
    from datetime import datetime
    try:
        datetime.strptime(date_str, date_format)
        return True
    except (ValueError, TypeError):
        return False

def validate_file_path(
    path: Path,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    create_if_missing: bool = False
) -> bool:
    """
    Validate file path.
    
    Args:
        path: Path to validate
        must_exist: Path must exist
        must_be_file: Path must be a file
        must_be_dir: Path must be a directory
        create_if_missing: Create directory if missing
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If path is invalid
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        if create_if_missing and must_be_dir:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
        else:
            raise ValueError(f"Path does not exist: {path}")
    
    if path.exists():
        if must_be_file and not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        if must_be_dir and not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
    
    return True


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be dictionary, got {type(config)}")
    
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    return True


def validate_numeric_range(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value"
) -> bool:
    """
    Validate numeric value is within range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Value name for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value)}")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    
    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove invalid characters.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
    
    return filename