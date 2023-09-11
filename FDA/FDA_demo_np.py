import numpy as np
from PIL import Image
from utils import FDA_source_to_target_np
import utils.scipy as sp

im_src = Image.open("demo_images/source.png").convert('RGB')
im_trg = Image.open("demo_images/target.png").convert('RGB')

im_src = im_src.resize( (1024,512), Image.Resampling.BICUBIC )
im_trg = im_trg.resize( (1024,512), Image.Resampling.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

src_in_trg = src_in_trg.transpose((1,2,0))
sp.toimage(src_in_trg, cmin=0.0, cmax=255.0).save('demo_images/src_in_tar.png')

#TO DO:
#Gather Dataset, develop NN to learn - exclude the segemntation in paper?
