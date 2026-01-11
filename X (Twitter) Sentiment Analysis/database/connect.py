from pymongo import MongoClient
from config.settings import MONGODBURL,DB_BNAME
from loguru import logger
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
from typing import Dict,List,Optional


class DatabaseConnection:
    def __init__(self,uri:str=MONGODBURL,db_name:str=DB_BNAME):
        self.uri=uri
        self.db_name=db_name
        self.client:None
        self.db=None
        self.connect()
        
    def connect(self):
        if self.client:
            return
        try:
            self.client=MongoClient(self.uri)
            self.db=self.client[self.db_name]
            self.db.command("ping")
            logger.info(f"Connected to MongoDB : {self.db_name}")
        except PyMongoError as e:
            logger.error(f"faied to initialize database {DB_BNAME}, {e}")
            raise
    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDb connection closed")
    def get_collection(self,name:str)->Collection:
        if not self.db:
            self.connect()
        return self.db[name]
    def get_collection_names(self) -> List[str]:
        if not self.db:
            self.connect()
        return self.db.list_collection_names()
    
    def insert_one(self,collection:str, document:Dict):
        col=self.get_collection(collection)
        return col.insert_one(document)
    def insert_many(self,collection:str,documents:List[Dict],ordered=False):
        if not documents:
            return None
        return self.get_collection(collection).insert_many(documents,ordered=ordered)
    def find_one(self,collection:str,query:Dict):
        col=self.get_collection(collection)
        return col.find_one(query)
    def find_many(
            self,
            collection:str,
            query:Dict=None,
            limit:int=0,
            projection:Dict=None
    ):
        col=self.get_collection(collection)
        cursor=col.find(query or {},projection)
        if limit:
            cursor.limit(limit)
        return list(cursor)
    def count(self, collection: str, query: Dict = None) -> int:
        col = self.get_collection(collection)
        return col.count_documents(query or {})
    def update_one(self, collection: str, query: Dict, update: Dict, upsert=False):
        col = self.get_collection(collection)
        return col.update_one(query, {"$set": update}, upsert=upsert)
    # ------------------------------------------------------------------
    # Indexes
    # ------------------------------------------------------------------
    def create_index(self, collection: str, field: str, unique=False):
        col = self.get_collection(collection)
        col.create_index(field, unique=unique)
    

