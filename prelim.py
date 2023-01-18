# import pandas as pd
# import os
# from pycocotools.coco import COCO
# import skimage.io as io
# import matplotlib.pyplot as plt
# from pathlib import Path

# dataDir = Path('Data/train2017')
# annFile = Path('Data/annotations/captions_train2017.json')
# # cap = "A baker is working in the kitchen rolling dough."
# coco = COCO(annFile)
# annIDs = coco.getAnnIds()
# anns = coco.loadAnns(annIDs)

# for ann in anns:
#     if ann['caption'] == cap:
#         imgID = ann['image_id']


# imgIds = coco.getImgIds()
# imgs = coco.loadImgs(imgIds[0])
# fig, axs = plt.subplots()

# # captions/anns mapped to given image
# I = io.imread(dataDir/imgs[0]['file_name'])
# annIds = coco.getAnnIds(imgIds=[imgs[0]['id']])
# anns = coco.loadAnns(annIds)
# print((anns))
# for ann in anns: print((ann['caption']))
# axs.imshow(I)
# plt.sca(axs)
# coco.showAnns(anns, draw_bbox=False)

# ## image mapped to given caption
# img = coco.loadImgs(imgID)
# I = io.imread(dataDir/img[0]['file_name'])
# axs.imshow(I)
# plt.sca(axs)

# plt.show()

# from pprint import pprint
# import json
# json_data=open("Data/annotations/annotations.json")
# jdata = json.load(json_data)

# print(type(jdata))
# # for key, value in jdata.values():
# #    pprint("Key:")
# #    pprint(key)
'''
to verify batch generation
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

dataset = open('Data/valimglist.txt').read().splitlines()
batch_size = 8
def generate_batch(batch_size):
    batch=random.sample(dataset,batch_size) #Sample from images at random
    real_images=np.empty(shape=[batch_size,560,560,3]) #For reading groundtruth images
    encoded_sentence=np.empty(shape=[batch_size,4800]) #For reading encoded sentences
    for i in range(len(batch)):
        real_images[i]=Image.open(r'Data/val2017/'+batch[i][:-4]+'.png').convert('RGB').resize((560,560))
        sentence_list=np.load('Data/encoded_vector/val_annotations/'+batch[i][:-4]+'.npy')
        encoded_sentence[i] = sentence_list[random.randint(0,np.shape(sentence_list)[0]-1)]
    return real_images,encoded_sentence

img, vec = generate_batch(batch_size)
print(img.shape)
print(vec.shape)
# print(img[0])
'''


'''
to verify the encodings
from os import listdir
from os.path import isfile, join
import skipthoughts
import os.path
import pickle
import numpy as np
from pycocotools.coco import COCO
from pathlib import Path
from sent2vec.vectorizer import Vectorizer

''''''
    encode captions into skipthoughts vectors for each image in the dataset with same name as of image
    sampling batches selects at random numbered-file_names and load the corresponding caption vectors and image for generator
''''''

dataDir = Path('Data/train2017')    # val2017 for validation set
annFile = Path('Data/annotations/captions_train2017.json')  #captions_val2017.json for validation set annotations
# annotation_dir = 'Data/annotations/encodable'
encoded_vector_dir = 'Data/encoded_vector'    # val_annotations for validation set

vectmodel = Vectorizer()
model = skipthoughts.load_model()
coco = COCO(annFile)
imgIds = coco.getImgIds()
for imgId in imgIds:
    img = coco.loadImgs(imgId)
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img[0]['id']]))
    file_name = img[0]['file_name'][:-4] + '.npy'
    if not os.path.exists(join(encoded_vector_dir,file_name)):
        captions = [ann['caption'] for ann in anns]
        try:
            caption_vectors = skipthoughts.encode(model, captions)
            vectmodel.run(captions)
            capt_vecs = vectmodel.vectors
        except:
            pass
        print(file_name)
        np.save(join(encoded_vector_dir,file_name), caption_vectors)
        np.save(join(encoded_vector_dir,"newvecs.npy"), capt_vecs)
    else:
        print("skipped")
    break

caption_vectors = np.load(join(encoded_vector_dir,"000000391895.npy"))
capt_vecs = np.load(join(encoded_vector_dir,"newvecs.npy"))
print(caption_vectors)
print(capt_vecs)
print(caption_vectors.shape == capt_vecs.shape)
'''
