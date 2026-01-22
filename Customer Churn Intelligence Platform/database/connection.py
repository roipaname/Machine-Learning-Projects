from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,Session
from config.settings import (
    DB_NAME,DB_HOST,DB_PASSWORD,DB_PORT,DB_USER
)

from loguru import logger
from contextlib import contextmanager



class DatabaseConnection:

    def __init__(self):
        self.localSesiion=None
        self.engine=self.init_db()

    def init_db(self):
        pass