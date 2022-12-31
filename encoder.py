from os import listdir
from os.path import isfile, join
import skipthoughts
import os.path
import pickle

annotation_dir = 'Data/annotations/encodable'
encoded_vector_dir = 'Data/encoded_vector'
# garbage = 'Data/test_garbage'

model = skipthoughts.load_model()
onlyfiles = [f_ for f_ in listdir(annotation_dir) if isfile(join(annotation_dir, f_))]
splitfs = [file for file in onlyfiles if 'xa' in file] # splitting 100MB json into chunks and encode and dump using *split*
splitfs.sort()
for files in splitfs:
    files_ = files+'.pkl' #files[0:-4]+'pkl'
    print(files_)
    if not os.path.exists(join(encoded_vector_dir,files_)):
        with open(join(annotation_dir,files)) as f:
            print(join(annotation_dir,files))
            captions = f.read().split(',')
            f.close()
        captions = [cap for cap in captions if len(cap.strip()) > 0]

        try:
            caption_vectors = skipthoughts.encode(model, captions)
        except:
            # with open(join(garbage,files), mode='w') as myfile:
            #     myfile.write(' ')
            pass
        print(files_)
        with open(join(encoded_vector_dir,"annotations_capt_train.pkl"), mode='ab+') as myfile:
            pickle.dump(caption_vectors, myfile)
            myfile.close()
    else:
        print("skipped")
