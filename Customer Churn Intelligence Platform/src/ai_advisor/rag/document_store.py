from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.settings import Settings
from chromadb.utils import embedding_functions
from loguru import logger
import os
from pathlib import Path
from typing import List,Dict 