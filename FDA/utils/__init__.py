import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import center_crop

def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = torch.abs(fft_im)#fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    #fft_amp = torch.sqrt(fft_amp)
    #fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    fft_pha = torch.angle(fft_im)
    return fft_amp, fft_pha

def gaussian_heatmap(x,y,sig):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    #sig = torch.randint(low=1,high=150,size=(1,)).cuda()[0]
    image_size = x.shape[1:]
    sig = 1 + (150 - 1) * sig
    #print("debugging")
    center = (torch.tensor(image_size[0]/2, dtype=torch.int).cuda(), torch.tensor(image_size[1]/2, dtype=torch.int).cuda())
    #center = (torch.randint(image_size[0],(1,))[0].cuda(), torch.randint(image_size[1],(1,))[0].cuda())
    #print(center)
    x_axis = torch.linspace(0, image_size[0]-1, image_size[0]).cuda() - center[0]
    # print("x-axis")
    # print(x_axis)
    # print(x_axis.shape)
    y_axis = torch.linspace(0, image_size[1]-1, image_size[1]).cuda() - center[1]
    # print("y-axis")
    # print(y_axis)
    # print(y_axis.shape)
    xx, yy = torch.meshgrid(x_axis, y_axis)
    # print(xx)
    # print(xx.shape)
    # print(yy)
    # print(yy.shape)
    # print(torch.square(xx))
    # print(torch.square(yy))
    # print(-0.5 * (torch.square(xx)+torch.square(yy)) / torch.square(sig))
    # print(sig)
    kernel = torch.exp(-0.5 * (torch.square(xx) + torch.square(yy)) / torch.square(sig))
    #print("kernel")
    #torch.set_printoptions(profile="full")    
    #print(kernel)
    #np.savetxt('my_file.txt', kernel.cpu().detach().numpy())
    #print(kernel.shape)
    new_img = (x*(1-kernel) + y*kernel).type(torch.float32)
    #new_img = (x*(1-kernel) + 255*kernel).type(torch.uint8)
    #print(new_img)
    #print(x)
    return new_img

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    # _, h, w = amp_src.size()
    # b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    # amp_src[:,0:b,0:b]     = amp_trg[:,0:b,0:b]      # top left
    # amp_src[:,0:b,w-b:w]   = amp_trg[:,0:b,w-b:w]    # top right
    # amp_src[:,h-b:h,0:b]   = amp_trg[:,h-b:h,0:b]    # bottom left
    # amp_src[:,h-b:h,w-b:w] = amp_trg[:,h-b:h,w-b:w]  # bottom right
    # return amp_src
    a_src = torch.fft.fftshift(amp_src, dim=(-2, -1))
    a_trg = torch.fft.fftshift(amp_trg, dim=(-2, -1))
    
    _, h, w = a_src.size()
    bw = (  np.floor(w*L.item()*0.5)  ).astype(int)
    bh = (  np.floor(h*L.item()*0.5)  ).astype(int) 
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-bh
    h2 = c_h+bh
    w1 = c_w-bw
    w2 = c_w+bw

    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = gaussian_heatmap(a_src, a_trg, L)
    a_src = torch.fft.ifftshift(a_src, dim=(-2, -1))
    #print(a_src.size())
    return a_src

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    bw = (  np.floor(w*L*0.5)  ).astype(int)
    bh = (  np.floor(h*L*0.5)  ).astype(int) 
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-bh
    h2 = c_h+bh
    w1 = c_w-bw
    w2 = c_w+bw

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    print(a_src.shape)
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    
    return a_src

def FDA_source_to_target_2(src_img, trg_img, L=0.01):
    # exchange magnitude
    # input: src_img, trg_img
    #print('PROCESS')
    #print(src_img.shape)
    if not torch.is_tensor(src_img):
        #print('PROCESSTORCH')
        src_img = torch.from_numpy(src_img)
    if not torch.is_tensor(trg_img):
        #print('PROCESSTORCH')
        trg_img = torch.from_numpy(trg_img)

    # get fft of both source and target
    #print('PROCESS2')
    fft_src = torch.fft.fft2( src_img.cuda(), dim=(-2, -1)).cuda()
    fft_trg = torch.fft.fft2( trg_img.cuda(), dim=(-2, -1)).cuda()

    # extract amplitude and phase of both ffts
    #print('PROCESS3')
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())
    
    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )
    #print(amp_src_)
    # recompose fft of source
    #fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    # real = torch.cos(pha_src.clone()) * amp_src_.clone()
    # imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    # fft_y = torch.complex(real, imag)
    fft_y = amp_src_ * torch.exp(1j * pha_src)
    #print(fft_y.size())
    # get the recomposed image: source content, target style
    #_, imgH, imgW = src_img_tensor.size()
    _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(fft_y, dim=(-2, -1), s=[imgH, imgW])
    src_in_trg = torch.tensor(src_in_trg.cpu().detach().numpy(), requires_grad=True).cuda()
    outmap_min = torch.min(src_in_trg).cuda()
    outmap_max = torch.max(src_in_trg).cuda()
    src_in_trg = (src_in_trg.clone().detach() - outmap_min) / (outmap_max - outmap_min)
    return src_in_trg

def FDA_source_to_target(src_img, trg_img, L=0.01):
    # exchange magnitude
    # input: src_img, trg_img
    #print('PROCESS')
    #print(src_img.shape)
    if not torch.is_tensor(src_img):
        #print('PROCESSTORCH')
        src_img = torch.from_numpy(src_img)
    if not torch.is_tensor(trg_img):
        #print('PROCESSTORCH')
        trg_img = torch.from_numpy(trg_img)

    # get fft of both source and target
    #print('PROCESS2')
    fft_src = torch.fft.fft2( src_img.cuda(), dim=(-2, -1)).cuda()
    fft_trg = torch.fft.fft2( trg_img.cuda(), dim=(-2, -1)).cuda()

    # extract amplitude and phase of both ffts
    #print('PROCESS3')
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())
    
    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )
    #print(amp_src_)
    # recompose fft of source
    #fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    # real = torch.cos(pha_src.clone()) * amp_src_.clone()
    # imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    # fft_y = torch.complex(real, imag)
    fft_y = amp_src_ * torch.exp(1j * pha_src)
    #print(fft_y.size())
    # get the recomposed image: source content, target style
    #_, imgH, imgW = src_img_tensor.size()
    _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(fft_y, dim=(-2, -1), s=[imgH, imgW])
    src_in_trg = torch.tensor(src_in_trg.cpu().detach().numpy(), requires_grad=True)
    return src_in_trg

def FDA_source_to_target_unet(src_img, trg_img, unet_model, L=0.01):
    # exchange magnitude
    # input: src_img, trg_img
    #print('PROCESS')
    #print(src_img.shape)
    if not torch.is_tensor(src_img):
        #print('PROCESSTORCH')
        src_img = torch.from_numpy(src_img)
    if not torch.is_tensor(trg_img):
        #print('PROCESSTORCH')
        trg_img = torch.from_numpy(trg_img)

    # get fft of both source and target
    #print('PROCESS2')
    fft_src = torch.fft.fft2( src_img.cuda(), dim=(-2, -1)).cuda()
    fft_trg = torch.fft.fft2( trg_img.cuda(), dim=(-2, -1)).cuda()

    # extract amplitude and phase of both ffts
    #print('PROCESS3')
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())
    
    # replace the low frequency amplitude part of source with that from target
    #amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )
    transform = transforms.Pad([10, 4, 11, 4])
    amp_src2 = transform(amp_src.cuda())
    amp_src_ = unet_model(torch.unsqueeze(amp_src2.cuda(), 0))
    
    amp_src_ = center_crop(torch.squeeze(amp_src_), [600, 1067]).cuda()
    #print(amp_src_)
    # recompose fft of source
    #fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    # real = torch.cos(pha_src.clone()) * amp_src_.clone()
    # imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    # fft_y = torch.complex(real, imag)
    fft_y = amp_src_.cuda() * torch.exp(1j * pha_src)
    #print(fft_y.size())
    # get the recomposed image: source content, target style
    #_, imgH, imgW = src_img_tensor.size()
    _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(fft_y.cuda(), dim=(-2, -1), s=[imgH, imgW])
    src_in_trg = torch.tensor(src_in_trg.cpu().detach().numpy(), requires_grad=True)
    result_map = {
        "image": src_in_trg,
        "src_amp": amp_src,
        "fake_amp": amp_src_
    }
    return result_map

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img
    
    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)
    
    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )
    #print("amp src", amp_src_)
    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)
    #print(src_in_trg)
    return src_in_trg

