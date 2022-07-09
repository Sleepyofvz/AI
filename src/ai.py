from spotipy.util import CLIENT_CREDS_ENV_VARS
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

birdy_uri = 'spotify:artist:2WX2uTcsvV5OnS0inACecP'
cid = 'c6cb940d34be48a4ba99edee4d43fc63'
secret = '81f29637d8e7448da3d37b7651008d3d'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
"""## Import Libraries"""

import os
import PIL
import clip
import torch
import csv
import numpy as np
import torchvision
import urllib.request
import json

f = open('./drive/MyDrive/keywords.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
text = []
for line in rdr:
    text.append(line[0])
f.close()   
"""## Model"""

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

"""## Zero-shot Prediction"""
print(text)
#for c in cifar100.classes:
#  print(f"a photo of a {c}")

#text = ["cloud", "camping", "person", "dog", "tree", "mountain", "sea", "sun", "subway", "book", "beer", "moon", "sun", "dog", "ocean", "tree", "coffee", "cafe", "computer"]#인식될 키워드 리스트 수정바람

# Prepare the inputs
image = PIL.Image.open('./drive/MyDrive/camping.jpg')
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print(100.0 * image_features @ text_features.T)

values, indices = similarity[0].topk(5)


# Print the result
print("\nTop predictions:\n")
n = 0
for value, index in zip(values, indices):
    if n==0:
        keyword = text[index]
    n= n+1
    print(f"{text[index]:>16s}: {100 * value.item():.2f}%%")

print(keyword)

results = sp.search(q=keyword+" genre: rock", limit=20)
#print(results['tracks']['items'][0]['preview_url'])
#print(results['tracks']['items'][0]['album']['images'][0]['url'])
#print(results['tracks']['items'][0]['artists'][0]['name'])
#print(results['tracks']['items'][0]['album']['name'])
#print(results['tracks']['items'][0]['id'])
print(results['tracks']['items'][0]['duration_ms'])

data = {}
data['musics'] = []
for idx, track in enumerate(results['tracks']['items']):
    print(idx, track['name'], track['preview_url'], track['album']['images'][0]['url'], track['artists'][0]['name'], track['album']['name'], track['id'], track['duration_ms'])
    urllib.request.urlretrieve(track['album']['images'][0]['url'], track['name']+".jpg")
    img = PIL.Image.open(track['name']+".jpg")
    display(img)
    data['musics'].append({
        "title": track['name'],
        "img_url":  track['album']['images'][0]['url'],
        "music_url": track['preview_url'],
        "artists": track['artists'][0]['name'],
        "album_name": track['album']['name'],
        "id": track['id'],
        "duration_ms": track['duration_ms']
    })

print(data)
with open("./result.json", 'w') as outfile:
    json.dump(data, outfile, indent=8)