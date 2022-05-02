import os
import json

video = os.listdir('../../../Celeb-df/Celeb-real')+os.listdir('../../../Celeb-df/Celeb-synthesis')+os.listdir('../../../Celeb-df/YouTube-real')
for i in range(len(video)):
    video[i] = video[i][:-4:]

with open ('celeb.json') as f:
    image = json.load(f)

for i in range(len(image)):
    sp = image[i][0].split('/')
    image[i] = sp[1]
print(len(image), len(video))

wrong = []
for i in video:
    if i in image:
        continue
    else:
        wrong.append(i)

print(wrong, len(wrong))

