# T2I-Service
API served service to generate images from text captions 

- `dcganimp1.py` rewritten code with tf v2, Tape and some other twweaks
- `dcganv1.py` forked implementation, build using CUDA, NVidia drivers and tensorflow v1
- `dcganv2.py` modified version with tensorflow v2 compatibility and executable for an AMD machine
- `skipthoughts.py` main code for `skipthoughts` encoding; including assest dictionaries from `Data/skipthougts` (gitignored; can be downloaded from <https://github.com/ryankiros/skip-thoughts>)
- `encoder.py` driver code for text vectorization/encoding (similar to Word2Vec)
- `prelim.py` redundant code for testing things out separately without including in the core flow
- `report.txt` report file of tensorflow compatibility script for `v1` to `v2`; script can be found in tf docs<br>
P.S. script was necessary, but not sufficient in my case; had to edit the code anyhow at some places
- `testing.py` some code to perform post-training actions like pinging the model to generate image from input text (can be optimized for final deployment to serve the flow)
<br><br>
- `Data/` contains the train/val COCO imageset, skipthoughts dictionaries, annotations from COCO imageset
visualiazation of annotations and other COCO features can be done with some code chunks in `prelim.py`
