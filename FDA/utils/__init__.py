import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import center_crop
from torchvision.utils import save_image
from torchvision.transforms import Resize

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

# def FDA_source_to_target_2(src_img, trg_img, L=0.01):
#     # exchange magnitude
#     # input: src_img, trg_img
#     #print('PROCESS')
#     #print(src_img.shape)
#     if not torch.is_tensor(src_img):
#         #print('PROCESSTORCH')
#         src_img = torch.from_numpy(src_img)
#     if not torch.is_tensor(trg_img):
#         #print('PROCESSTORCH')
#         trg_img = torch.from_numpy(trg_img)

#     # get fft of both source and target
#     #print('PROCESS2')
#     fft_src = torch.fft.fft2( src_img.cuda(), dim=(-2, -1)).cuda()
#     fft_trg = torch.fft.fft2( trg_img.cuda(), dim=(-2, -1)).cuda()

#     # extract amplitude and phase of both ffts
#     #print('PROCESS3')
#     amp_src, pha_src = extract_ampl_phase( fft_src.clone())
#     amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())
    
#     # replace the low frequency amplitude part of source with that from target
#     amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )
#     #print(amp_src_)
#     # recompose fft of source
#     #fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
#     # real = torch.cos(pha_src.clone()) * amp_src_.clone()
#     # imag = torch.sin(pha_src.clone()) * amp_src_.clone()
#     # fft_y = torch.complex(real, imag)
#     fft_y = amp_src_ * torch.exp(1j * pha_src)
#     #print(fft_y.size())
#     # get the recomposed image: source content, target style
#     #_, imgH, imgW = src_img_tensor.size()
#     _, imgH, imgW = src_img.size()
#     src_in_trg = torch.fft.irfft2(fft_y, dim=(-2, -1), s=[imgH, imgW])
#     src_in_trg = torch.tensor(src_in_trg.cpu().detach().numpy(), requires_grad=True).cuda()
#     outmap_min = torch.min(src_in_trg).cuda()
#     outmap_max = torch.max(src_in_trg).cuda()
#     src_in_trg = (src_in_trg.clone().detach() - outmap_min) / (outmap_max - outmap_min)
#     return src_in_trg

# def FDA_source_to_target(src_img, trg_img, L=0.01):
#     # exchange magnitude
#     # input: src_img, trg_img
#     #print('PROCESS')
#     #print(src_img.shape)
#     if not torch.is_tensor(src_img):
#         #print('PROCESSTORCH')
#         src_img = torch.from_numpy(src_img)
#     if not torch.is_tensor(trg_img):
#         #print('PROCESSTORCH')
#         trg_img = torch.from_numpy(trg_img)

#     # get fft of both source and target
#     #print('PROCESS2')
#     fft_src = torch.fft.fft2( src_img.cuda(), dim=(-2, -1)).cuda()
#     fft_trg = torch.fft.fft2( trg_img.cuda(), dim=(-2, -1)).cuda()

#     # extract amplitude and phase of both ffts
#     #print('PROCESS3')
#     amp_src, pha_src = extract_ampl_phase( fft_src.clone())
#     amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())
    
#     # replace the low frequency amplitude part of source with that from target
#     amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )
#     #print(amp_src_)
#     # recompose fft of source
#     #fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
#     # real = torch.cos(pha_src.clone()) * amp_src_.clone()
#     # imag = torch.sin(pha_src.clone()) * amp_src_.clone()
#     # fft_y = torch.complex(real, imag)
#     fft_y = amp_src_ * torch.exp(1j * pha_src)
#     #print(fft_y.size())
#     # get the recomposed image: source content, target style
#     #_, imgH, imgW = src_img_tensor.size()
#     _, imgH, imgW = src_img.size()
#     src_in_trg = torch.fft.irfft2(fft_y, dim=(-2, -1), s=[imgH, imgW])
#     src_in_trg = torch.tensor(src_in_trg.cpu().detach().numpy(), requires_grad=True)
#     return src_in_trg\
    
def FDA_source_to_target_unet(src_img, trg_img, unet_model, start_iter, L=0.01):

    transform = transforms.Pad([10, 4, 11, 4])
    transform_grey = transforms.Grayscale(num_output_channels=1)
    if not torch.is_tensor(src_img):
        src_img = torch.from_numpy(src_img)
    if not torch.is_tensor(trg_img):
        trg_img = torch.from_numpy(trg_img)

    src_img = src_img.float()/255.0
    trg_img = trg_img.float()/255.0
    fft_src = torch.fft.fft2( src_img.cuda(), dim=(-2, -1), norm="ortho").cuda()    
    fft_trg = torch.fft.fft2( trg_img.cuda(), dim=(-2, -1), norm="ortho").cuda()

    amp_src, pha_src = extract_ampl_phase( fft_src.cuda())    
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.cuda())
    # amp_src.requires_grad_()
    # amp_trg.requires_grad_()
    # pha_src.requires_grad_()
    # pha_trg.requires_grad_()

    amp_src = (amp_src - amp_src.min()) / (amp_src.max() - amp_src.min())
    amp_trg = (amp_trg - amp_trg.min()) / (amp_trg.max() - amp_trg.min())
    org_pha_src = pha_src.cuda()
    # org_pha_src.requires_grad_()
    pha_src = (pha_src - pha_src.min()) / (pha_src.max() - pha_src.min())
    
    
    src_img_pad = transform(src_img.cuda())
    trg_img_pad = transform(trg_img.cuda())
    amp_fusion = torch.unsqueeze(src_img_pad.cuda(), 0) #torch.cat((torch.unsqueeze(src_img_pad.cuda(), 0), torch.unsqueeze(trg_img_pad.cuda(), 0)), dim=0).cuda()
    # amp_fusion.requires_grad_()
    # print("amp_fusion: "+str(amp_fusion.requires_grad))
    
    src_in_trg = unet_model(amp_fusion.cuda()).cuda()    
    # print("src_in_trg: "+str(src_in_trg.requires_grad))
    src_in_trg = torch.mean(src_in_trg, dim=0)
    # print("src_in_trg: "+str(src_in_trg.requires_grad))
    split_tensors = torch.split(src_in_trg, 1, dim=0)
    # print("split_tensors: "+str(split_tensors[0].requires_grad))
    
    mean_tensors = [torch.mean(split_tensor, dim=0) for split_tensor in split_tensors]
    # print("mean_tensors[0]: "+str(mean_tensors[0].requires_grad))
    src_in_trg = torch.stack(mean_tensors, dim=0)
    # print("src_in_trg: "+str(src_in_trg.requires_grad))
    
        
    src_in_trg = torch.nn.functional.interpolate(src_in_trg.unsqueeze(0), size=(600, 1067)).squeeze().cuda()
    # print("src_in_trg: "+str(src_in_trg.requires_grad))
    #src_in_trg = (src_in_trg - src_in_trg.min()) / (src_in_trg.max() - src_in_trg.min())
    #src_in_trg = torch.nn.Sigmoid()(src_in_trg)
    # print("src_in_trg: "+str(src_in_trg.requires_grad))
    fft_src_in_trg = torch.fft.fft2( src_in_trg.cuda(), dim=(-2, -1), norm="ortho").cuda()  
    # print("fft_src_in_trg: "+str(fft_src_in_trg.requires_grad))
    
    
    amp_src_in_trg, pha_src_in_trg = extract_ampl_phase( fft_src_in_trg.cuda()) 
    # print("amp_src_in_trg: "+str(amp_src_in_trg.requires_grad))
    # print("pha_src_in_trg: "+str(pha_src_in_trg.requires_grad))
    
    pha_src_in_trg = (pha_src_in_trg - pha_src_in_trg.min()) / (pha_src_in_trg.max() - pha_src_in_trg.min())
    amp_src_in_trg = (amp_src_in_trg - amp_src_in_trg.min()) / (amp_src_in_trg.max() - amp_src_in_trg.min())
    src_in_trg = (src_in_trg - src_in_trg.min()) / (src_in_trg.max() - src_in_trg.min())
    
    
    result_map = {
        "src_amp": amp_src.cuda(),
        "fake_amp": amp_src_in_trg.cuda(),
        "trg_amp": amp_trg.cuda(),
        "src_org": src_in_trg.cuda(),
        "pha_fake": pha_src_in_trg.cuda(),
        "pha_src": pha_src.cuda(),
        "trg_img": trg_img,
    }
    return result_map

def unet_helper(src_in_trg):
    src_in_trg = torch.mean(src_in_trg, dim=0)
    # print("src_in_trg: "+str(src_in_trg.requires_grad))
    split_tensors = torch.split(src_in_trg, 2, dim=0)
    # print("split_tensors: "+str(split_tensors[0].requires_grad))
    
    mean_tensors = [torch.mean(split_tensor, dim=0) for split_tensor in split_tensors]
    # print("mean_tensors[0]: "+str(mean_tensors[0].requires_grad))
    src_in_trg = torch.stack(mean_tensors, dim=0)
    # print("src_in_trg: "+str(src_in_trg.requires_grad))
        
    src_in_trg = torch.nn.functional.interpolate(src_in_trg.unsqueeze(0), size=(600, 1067)).squeeze().cuda()
    src_in_trg = (src_in_trg - src_in_trg.min()) / (src_in_trg.max() - src_in_trg.min())
    
    return src_in_trg

def FDA_output(src_img, trg_img):
    src_img = src_img.float()/255.0
    
    fft_src = torch.fft.fft2( src_img.cuda(), dim=(-2, -1)).cuda()    
    fft_trg = torch.fft.fft2( trg_img.cuda(), dim=(-2, -1)).cuda()

    # extract amplitude and phase of both ffts
    #print('PROCESS3')
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())    
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    amp_src = (amp_src - amp_src.min()) / (amp_src.max() - amp_src.min())
    amp_trg = (amp_trg - amp_trg.min()) / (amp_trg.max() - amp_trg.min())
    pha_src = (pha_src - pha_src.min()) / (pha_src.max() - pha_src.min())
    pha_trg = (pha_trg - pha_trg.min()) / (pha_trg.max() - pha_trg.min())
    
    result = {
        "amp_src": amp_src,
        "amp_trg": amp_trg,
        "pha_src": pha_src,
        "pha_trg": pha_trg
    }
    return result
    

# def FDA_source_to_target_unet(src_img, trg_img, unet_model, start_iter, L=0.01):
#     # exchange magnitude
#     # input: src_img, trg_img
#     #print('PROCESS')
#     #print(src_img.shape)
#     transform = transforms.Pad([10, 4, 11, 4])
#     transform_grey = transforms.Grayscale(num_output_channels=1)
#     if not torch.is_tensor(src_img):
#         #print('PROCESSTORCH')
#         src_img = torch.from_numpy(src_img)
#     if not torch.is_tensor(trg_img):
#         #print('PROCESSTORCH')
#         trg_img = torch.from_numpy(trg_img)

#     # get fft of both source and target
#     #print('PROCESS2')
#     src_img = src_img.float()/255.0
    
#     fft_src = torch.fft.fft2( src_img.cuda(), dim=(-2, -1)).cuda()    
#     fft_trg = torch.fft.fft2( trg_img.cuda(), dim=(-2, -1)).cuda()

#     # extract amplitude and phase of both ffts
#     #print('PROCESS3')
#     amp_src, pha_src = extract_ampl_phase( fft_src.clone())    
#     amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

#     amp_src = (amp_src - amp_src.min()) / (amp_src.max() - amp_src.min())
#     amp_trg = (amp_trg - amp_trg.min()) / (amp_trg.max() - amp_trg.min())
    
    
#     #     # Apply a fat filter to the magnitude of the Fourier transform
#     # fat_filter = torch.ones_like(amp_src)
#     # fat_filter = fat_filter.roll(shifts=(0, 0), dims=(1, 2))
#     # fat_filter[0, 0, :] = 0
#     # fat_filter[:, :, 0] = 0

#     # # Apply a Laplacian filter to the magnitude of the Fourier transform
#     # laplacian_filter = torch.zeros_like(amp_src)
#     # laplacian_filter[1, 1, :] = -4
#     # laplacian_filter[0, 1, :] = 1
#     # laplacian_filter[2, 1, :] = 1
#     # laplacian_filter[1, 0, :] = 1
#     # laplacian_filter[1, 2, :] = 1

#     # # Combine the fat filter and the Laplacian filter
#     # magnitude = amp_src * fat_filter + laplacian_filter
#     # # Inverse Fourier transform the filtered magnitude
#     # filtered_image = torch.fft.ifft2(magnitude)

#     # replace the low frequency amplitude part of source with that from target
#     #amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )
#     amp_src_pad = transform(amp_src.cuda())
#     amp_trg_pad = transform(amp_trg.cuda())
#     amp_fusion = torch.cat((torch.unsqueeze(amp_src_pad.cuda(), 0), torch.unsqueeze(amp_trg_pad.cuda(), 0)), dim=0).cuda()
#     #print("amp_fusion shape:" + str(amp_fusion.shape))
    
#     amp_src_out = unet_model(amp_fusion)
#     #print("amp_src_out shape:" + str(amp_src_out.shape))
    
#     amp_src_out = torch.mean(amp_src_out, dim=0)
#     split_tensors = torch.split(amp_src_out, 2, dim=0)
#     #print("split_tensors shape:" + str(len(split_tensors)))
#     mean_tensors = [torch.mean(split_tensor, dim=0) for split_tensor in split_tensors]
#     amp_src_out = torch.stack(mean_tensors, dim=0)
    
#     #print("amp_src_out shape:" + str(amp_src_out.shape))
    
#     amp_src_out_norm = (amp_src_out - amp_src_out.min()) / (amp_src_out.max() - amp_src_out.min())
    
    
#     # if start_iter%250 == 0:
#     #     print("printing: "+ str(start_iter))
#     #     # Save the image of the amplitude of the filtered image
#     #     save_image(transform_grey(amp_src).type(torch.float32),'./demo_images/'+str(start_iter)+"_src_amp_before"+'.png')
#     #     save_image(transform_grey(amp_trg).type(torch.float32),'./demo_images/'+str(start_iter)+"_src_trg_before"+'.png')
#     #     save_image(transform_grey(amp_src_out_norm).type(torch.float32),'./demo_images/'+str(start_iter)+"_src_amp_after"+'.png')

    
#     #amp_src_out_corp = center_crop(torch.squeeze(amp_src_out), [600, 1067]).cuda()
#     #print("amp_src_out shape:" + str(amp_src_out.shape))
#     #resize_transform = Resize(size=(3, 600, 1067))
#     amp_src_out_corp = torch.nn.functional.interpolate(amp_src_out_norm.unsqueeze(0), size=(600, 1067)).squeeze().cuda()
#     amp_src_out_corp = ((amp_src_out_corp - amp_src_out_corp.min()) / (amp_src_out_corp.max() - amp_src_out_corp.min()))
    
#     #print("amp_src_out_crop shape:" + str(amp_src_out_corp.shape))
#     #print(amp_src_)
#     # recompose fft of source
#     #fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
#     # real = torch.cos(pha_src.clone()) * amp_src_.clone()
#     # imag = torch.sin(pha_src.clone()) * amp_src_.clone()
#     # fft_y = torch.complex(real, imag)
#     fft_y = amp_src_out_corp.cuda() * torch.exp(1j * pha_src)
#     fft_y_org = amp_trg * torch.exp(1j * pha_src)
    
#     #print(fft_y.size())
#     # get the recomposed image: source content, target style
#     #_, imgH, imgW = src_img_tensor.size()
#     _, imgH, imgW = src_img.size()
#     src_in_trg = torch.fft.irfft2(fft_y.cuda(), dim=(-2, -1), s=[imgH, imgW]).cuda()
#     src_org = torch.fft.irfft2(fft_y_org.cuda(), dim=(-2, -1), s=[imgH, imgW])
    
#     #src_in_trg = torch.tensor(src_in_trg.cpu().detach().numpy(), requires_grad=True)
#     src_org = torch.tensor(src_org.cpu().detach().numpy(), requires_grad=True)
    
    
#     src_in_trg = (src_in_trg - src_in_trg.min()) / (src_in_trg.max() - src_in_trg.min())
#     src_org = (src_org - src_org.min()) / (src_org.max() - src_org.min())
    
#     result_map = {
#         "image": src_in_trg,
#         "src_amp": amp_src,
#         "fake_amp": amp_src_out_corp,
#         "trg_amp": amp_trg.cuda(),
#         "src_org": src_org
#     }
#     return result_map

# def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
#     # exchange magnitude
#     # input: src_img, trg_img
    
#     src_img_np = src_img #.cpu().numpy()
#     trg_img_np = trg_img #.cpu().numpy()

#     # get fft of both source and target
#     fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
#     fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

#     # extract amplitude and phase of both ffts
#     amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
#     amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)
    
#     # mutate the amplitude part of source with target
#     amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )
#     #print("amp src", amp_src_)
#     # mutated fft of source
#     fft_src_ = amp_src_ * np.exp( 1j * pha_src )

#     # get the mutated image
#     src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
#     src_in_trg = np.real(src_in_trg)
#     #print(src_in_trg)
#     return src_in_trg

