import pandas as pd
import os
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path

dataDir = Path('Data/val2017')
annFile = Path('Data/annotations/captions_val2017.json')
cap = "A baker is working in the kitchen rolling dough."
coco = COCO(annFile)
annIDs = coco.getAnnIds()
anns = coco.loadAnns(annIDs)

for ann in anns:
    if ann['caption'] == cap:
        imgID = ann['image_id']


# imgIds = coco.getImgIds()
# imgs = coco.loadImgs(imgIds[0])

fig, axs = plt.subplots()

## captions/anns mapped to given image
# I = io.imread(dataDir/imgs[0]['file_name'])
# annIds = coco.getAnnIds(imgIds=[imgs[0]['id']])
# anns = coco.loadAnns(annIds)
# axs.imshow(I)
# plt.sca(axs)
# coco.showAnns(anns, draw_bbox=False)

## image mapped to given caption
img = coco.loadImgs(imgID)
I = io.imread(dataDir/img[0]['file_name'])
axs.imshow(I)
plt.sca(axs)

plt.show()
