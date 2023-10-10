import numpy as np
from PIL import Image
from utils import FDA_source_to_target
import utils.scipy as sp
import torch

im_src = Image.open("demo_images/src_900.png").convert('RGB')
im_trg = Image.open("demo_images/trg_900.png").convert('RGB')

im_src = im_src.resize( (1024,512), Image.Resampling.BICUBIC )
im_trg = im_trg.resize( (1024,512), Image.Resampling.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

im_src = torch.from_numpy(im_src).cuda()
im_trg = torch.from_numpy(im_trg).cuda()
beta = torch.tensor([0.5]).cuda()
#[[0.1838],
#        [0.3036]]

print(im_src.shape)
print(im_trg.shape)

src_in_trg = FDA_source_to_target( im_src, im_trg, L=beta )

src_in_trg = src_in_trg.permute((1,2,0))
sp.toimage(src_in_trg.cpu().detach(), cmin=0.0, cmax=255.0).save('demo_images/src_in_tar_torch.png')
