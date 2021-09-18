import json

with open("./configs/configs.json") as dataFile:
  config = json.load(dataFile)

username = config["username"]
password = config["password"]

print(f'Hello{username}')