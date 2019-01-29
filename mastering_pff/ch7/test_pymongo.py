import pymongo
try:
    client = pymongo.MongoClient("localhost", 27017)
    print("Connected successfully!!!")
except pymongo.errors.ConnectionFailure as e:
   print("Could not connect to MongoDB: %s" % e)