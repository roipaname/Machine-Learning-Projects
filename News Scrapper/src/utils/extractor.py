from urllib.parse import urlparse

def extract_category(url: str) -> str:
    """
    Extract category from a news article URL.
    
    For URLs like:
        https://www.theguardian.com/football/...
        https://www.npr.org/sections/shots-health-news/...
        https://www.cnbc.com/... (uses first path part as category)
    
    Returns lowercase category string, or 'other' if cannot be determined.
    """
    try:
        path_parts = urlparse(url).path.strip("/").split("/")
        domain = urlparse(url).netloc.lower()
        
        if "theguardian.com" in domain:
            # Category is first part after base URL
            if len(path_parts) >= 1:
                return path_parts[0].lower()
        elif "npr.org" in domain:
            # NPR has '/sections/<category>/...' or '/g-s1-<id>/<slug>'
            if len(path_parts) >= 2 and path_parts[0] == "sections":
                return path_parts[1].lower()
            else:
                return "other"
        elif "cnbc.com" in domain:
            # CNBC: take first path part if it looks like category
            if len(path_parts) >= 1:
                return path_parts[0].lower()
        else:
            # Default for unknown or generic domains
            return "other"
        
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
        return "other"


