import re
import hashlib

def normalize_text(text:str)->str:
    text=text.lower().strip()
    text=re.sub(r"\s+"," ",text)
    return text

def content_hash(text:str)->str:
    if not isinstance(text,str):
        return None
    normalized=normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()