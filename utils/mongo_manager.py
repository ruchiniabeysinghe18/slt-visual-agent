"""
 * @class MongoDBmanager
 * @description MongoDBmanager use for connect and perfome operation with mongoDB database 
"""

import pathlib
from typing import List, Optional
import pymongo
from pymongo import MongoClient
from pymongo.errors import AutoReconnect, ConnectionFailure
from pymongo.command_cursor import CommandCursor
import bson
import os
from dotenv import load_dotenv
from utils.logger import get_debug_logger

from datetime import datetime
from bson.objectid import ObjectId

from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()  # take environment variables from .env.
# DB_USER = os.environ.get("DB_USER")
# DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DATABASE")
IP = os.environ.get("DB_HOST")
PORT = int(os.environ.get("DB_PORT"))

if not os.path.exists(pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "../logs")):
    os.makedirs(pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "../logs"))

logger = get_debug_logger(
    "mongo_manager", pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "../logs/server.log")
)


class MongoDBmanager:
    def __init__(self, collection, save_json=False):
        self.db = DB_NAME
        self.collection = collection
        
        # Connect to the DB
        try:
            # self.client = MongoClient(f"mongodb://{DB_USER}:{DB_PASS}@{IP}:{PORT}/{DB_NAME}")
            if not save_json:
                self.client = MongoClient(f"mongodb://{IP}:{PORT}/{DB_NAME}")
            else:
                self.client = AsyncIOMotorClient(f"mongodb://{IP}:{PORT}/{DB_NAME}")
            
            logger.debug(f"successfully connected to {DB_NAME} db")
        except (AutoReconnect, ConnectionFailure) as e:
            logger.error(f"failed to connect to {DB_NAME} db, error: {e}")
            raise Exception("DB CONNECTION ERROR")

    """
    get one document by query
    """

    def get_one_document(self, query):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        res = collection.find_one(query)
        return res
    
    def delete_one_document(self, query):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        res = collection.delete_one(query)
        return res
    
    def get_documents(self, query, sort=None, limit=0):
        if not isinstance(query, dict):
            raise ValueError("Query must be a dictionary")
        try:
            _DB = self.client[self.db]
            collection = _DB[self.collection]
            cursor = collection.find(query)

            if sort:
                cursor = cursor.sort(sort)
            if limit > 0:
                cursor = cursor.limit(limit)

            return list(cursor)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def get_user_ids(self, org_id):
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        res = collection.distinct("user_id", {"org_id": org_id})
        res = [str(x) if isinstance(x, int) else x for x in res]
        return res
    
    def insert_documents(self, documents):
        try:
            _DB = self.client[self.db]
            collection = _DB[self.collection]
            res = collection.insert_many(documents)
            return len(res.inserted_ids)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
        
    
    def add_field(self, query, field_name, field_value):
        """
        Add / or update a new field in a document
        """
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        
        update_result = collection.update_one(
            query,
            {'$set': {field_name: field_value}}
        )
        
        return update_result.modified_count > 0

    def update_one(self, query, fields_dict):
        """
        Update a document matching 'query' with 'fields_dict'.
        If no document matches, insert a new document with the fields.
        Returns True if a document was modified or inserted.
        """
        _DB = self.client[self.db]
        collection = _DB[self.collection]
        
        update_result = collection.update_one(
            query,
            {'$set': fields_dict},
            upsert=True  # create document if not existing
        )
        
        # Return True if modified or upserted
        return update_result.modified_count > 0 or update_result.upserted_id is not None

    
    def aggregate(self, query) -> Optional[List[dict]]:
        """
        A function to aggregate data using the specified query and return a list of dictionaries or None.
        """
        # Validation
        if query is None or not isinstance(query, list):
            logger.debug("aggregate | N/A | Invalid aggregation query: {}".format(query))
            return None

        _DB = self.client[self.db]
        collection = _DB[self.collection]

        try:
            cursor: CommandCursor = collection.aggregate(query)
            return list(cursor)  # Convert cursor to list
        except pymongo.errors.PyMongoError as e:
            logger.error(f"aggregate | N/A | Aggregation error: {e}")
            return None
        except Exception as e:
            logger.error(f"aggregate | N/A | Unexpected error during aggregation: {e}")
            return None

    """
    insert documents by bulk_write
    """

    def bulk_write(self, query):
        if query != None and len(query) > 0:
            _DB = self.client[self.db]

            collection = _DB[self.collection]
            ret = collection.bulk_write(query, ordered=True)
            return ret
        else:
            logger.debug("No query to bulk_write")

    """
    insert one document by insert_one
    """

    def insert_one(self, data):
        if data != None:
            _DB = self.client[self.db]
            collection = _DB[self.collection]
            ret = collection.insert_one(data)
            return ret
    """
    Get distinct values for a field
    """
    def get_distinct(self, field_name, filter_query=None):
        _DB = self.client[self.db]
        collection = _DB[self.collection]

        if filter_query is None:
            ret = collection.distinct(field_name)
        else:
            ret = collection.distinct(field_name, filter_query)

        return ret
        
    def update_field(self, user_id, field_name, new_value, clear=False):
        _DB = self.client[self.db]
        collection = _DB[self.collection]

        query = {"user_id": user_id}
        if not clear:
            update = {"$set": {field_name: new_value}}
        elif clear:
            update = {"$unset": {field_name: ""}}
        # Update the field only if the document exists (no upsert)
        result = collection.update_one(query, update, upsert=False)

        return result