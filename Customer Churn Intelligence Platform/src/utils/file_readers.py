from pathlib import Path
from loguru import logger
from typing import List,Dict
from pypdf import PdfReader
from docx import Document


def read_txt(path:Path)->str:
    return path.read_text(encoding='utf-8',errors='ignore')

def read_md(path:Path)->str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path:Path)->str:
    reader=PdfReader(path)
    pages=[page.extract_text() for page in reader.pages if page.extract_text() is not None]

    return "\n".join(pages)

def read_docx(path:Path)->str:
    doc=Document(path)
    texts=[para.text for para in doc.paragraphs if para.text.strip()!='']
    return "\n".join(texts)


READERS:dict [str,callable[[Path],str]]={
    ".txt": read_txt,
    ".md": read_md,
    ".pdf": read_pdf,
    ".docx": read_docx,
}

def read_file_to_text(filepath:str | Path)->str:
    path=Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File {filepath} does not exist")
    reader=READERS.get(path.suffix.lower())
    if not reader:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    logger.info(f"Reading file:{path.name}")
    text = reader(path)
    if not text.strip():
        raise ValueError(f"No extractable text in file: {path.name}")

    return text