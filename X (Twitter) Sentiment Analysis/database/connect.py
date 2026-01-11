from pymongo import MongoClient

client=MongoClient("mongodb://localhost:27017/")

db=client["x_sentiment"]

source_tweet=db["source_tweet"]

test_tweet = {
    "tweet_id": "123456789",
    "text": "I love working with MongoDB!",
    "user": "test_user",
    "created_at": "2026-01-11"
}

if source_tweet.count_documents({})==0:
    source_tweet.insert_one(test_tweet)
    print("Inserted test data")

print("Collections in 'twitter_sentiment' database:")
print(db.list_collection_names())

print("Setup complete!")
