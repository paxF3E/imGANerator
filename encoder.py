from os import listdir
from os.path import isfile, join
import skipthoughts
import os.path
import pickle
import numpy as np
from pycocotools.coco import COCO
from pathlib import Path

'''
    encode captions into skipthoughts vectors for each image in the dataset with same name as of image
    sampling batches selects at random numbered-file_names and load the corresponding caption vectors and image for generator
'''

dataDir = Path('Data/train2017')    # val2017 for validation set
annFile = Path('Data/annotations/captions_train2017.json')  #captions_val2017.json for validation set annotations
# annotation_dir = 'Data/annotations/encodable'
encoded_vector_dir = 'Data/encoded_vector/train_annotations'    # val_annotations for validation set

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
        except:
            pass
        print(file_name)
        np.save(join(encoded_vector_dir,file_name), caption_vectors)
    else:
        print("skipped")

# onlyfiles = [f_ for f_ in listdir(annotation_dir) if isfile(join(annotation_dir, f_))]
# splitfs = [file for file in onlyfiles if 'xa' in file] # splitting 100MB json into chunks and encode and dump using *split*
# splitfs.sort()
# for files in splitfs:
#     files_ = files+'.pkl' #files[0:-4]+'pkl'
#     print(files_)
#     if not os.path.exists(join(encoded_vector_dir,files_)):
#         with open(join(annotation_dir,files)) as f:
#             print(join(annotation_dir,files))
#             captions = f.read().split(',')
#             f.close()
#         captions = [cap for cap in captions if len(cap.strip()) > 0]

#         try:
#             caption_vectors = skipthoughts.encode(model, captions)
#         except:
#             # with open(join(garbage,files), mode='w') as myfile:
#             #     myfile.write(' ')
#             pass
#         print(files_)
#         with open(join(encoded_vector_dir,"annotations_capt_train.pkl"), mode='ab+') as myfile:
#             pickle.dump(caption_vectors, myfile)
#             myfile.close()
#     else:
#         print("skipped")
