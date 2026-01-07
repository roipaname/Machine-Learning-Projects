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