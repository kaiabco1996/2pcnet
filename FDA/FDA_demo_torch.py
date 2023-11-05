import numpy as np
from PIL import Image
from utils import FDA_source_to_target, combine_fourier_images, visual_ampl_phase
import utils.scipy as sp
import torch
from torchvision.utils import save_image

im_src = Image.open("demo_images/0a56c2e8-1290ac29.jpg").convert('RGB')
im_trg = Image.open("demo_images/0a8aa157-38e8ed64.jpg").convert('RGB')

im_src = im_src.resize( (1024,512), Image.Resampling.BICUBIC )
im_trg = im_trg.resize( (1024,512), Image.Resampling.BICUBIC )

im_src = np.asarray(im_src, np.float32)
im_trg = np.asarray(im_trg, np.float32)

im_src = im_src.transpose((2, 0, 1))
im_trg = im_trg.transpose((2, 0, 1))

im_src = torch.from_numpy(im_src).cpu()
im_trg = torch.from_numpy(im_trg).cpu()
beta = torch.tensor([0.5]).cuda()

print(im_src.shape)
print(im_trg.shape)
low_radii = 120 #the actual road
high_radii = 5 #the reflection control

src_in_trg = combine_fourier_images(im_src, im_trg, low_radii, high_radii)
# trg_in_src = combine_fourier_images(im_trg, im_src, low_radii, high_radii)
with torch.no_grad():
    fft_src = torch.fft.fft2( im_src, dim=(-2, -1), norm="ortho").cpu()    
    fft_trg = torch.fft.fft2( im_trg, dim=(-2, -1), norm="ortho").cpu()
    fft_src_in_trg = torch.fft.fft2( src_in_trg, dim=(-2, -1), norm="ortho").cpu()
    

    src_amp, src_phase = visual_ampl_phase(fft_src)
    trg_amp, trg_phase = visual_ampl_phase(fft_trg)
    src_in_trg_amp, src_in_trg_phase = visual_ampl_phase(fft_src_in_trg)
    

    src_amp = (src_amp - src_amp.min()) / (src_amp.max() - src_amp.min())
    src_phase = (src_phase - src_phase.min()) / (src_phase.max() - src_phase.min())
    trg_amp = (trg_amp - trg_amp.min()) / (trg_amp.max() - trg_amp.min())
    trg_phase = (trg_phase - trg_phase.min()) / (trg_phase.max() - trg_phase.min())
    im_src = (im_src - im_src.min()) / (im_src.max() - im_src.min())
    im_trg = (im_trg - im_trg.min()) / (im_trg.max() - im_trg.min())
    src_in_trg = (src_in_trg - src_in_trg.min()) / (src_in_trg.max() - src_in_trg.min())
    src_in_trg_amp = (src_in_trg_amp - src_in_trg_amp.min()) / (src_in_trg_amp.max() - src_in_trg_amp.min())
    src_in_trg_phase = (src_in_trg_phase - src_in_trg_phase.min()) / (src_in_trg_phase.max() - src_in_trg_phase.min())
    
    
    

    # src_amp = src_amp.permute((1,2,0))
    # src_phase = src_phase.permute((1,2,0))
    # trg_amp = trg_amp.permute((1,2,0))
    # trg_phase = trg_phase.permute((1,2,0))
    
    print(src_amp.shape)
    print(src_phase.shape)
    print(trg_amp.shape)
    print(trg_phase.shape)
    print(im_src.shape)
    print(im_trg.shape)
    print(src_in_trg_phase.shape)
    print(src_in_trg_amp.shape)
    print(src_in_trg.shape)
    
    

    save_image([im_src, src_amp.type(torch.float32), 
                src_phase.cpu().detach().type(torch.float32),
                im_trg,
                trg_amp.cpu().detach().type(torch.float32),
                trg_phase.cpu().detach().type(torch.float32),
                src_in_trg,
                src_in_trg_amp,
                src_in_trg_phase], 
               'demo_images/fft_visual.png', nrow=3)

# sp.toimage(src_amp.cpu().detach(), cmin=0.0, cmax=1.0).save('demo_images/src_amp.png')
# sp.toimage(src_phase.cpu().detach(), cmin=0.0, cmax=1.0).save('demo_images/src_phase.png')
# sp.toimage(trg_amp.cpu().detach(), cmin=0.0, cmax=1.0).save('demo_images/trg_amp.png')
# sp.toimage(trg_phase.cpu().detach(), cmin=0.0, cmax=1.0).save('demo_images/trg_phase.png')

