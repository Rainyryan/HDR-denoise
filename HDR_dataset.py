import os
import glob
import numpy as np
import cv2
# PyTorch
import torch
from torch.utils.data import Dataset
from my_GBTF import GBTF_CFAInterpolation

##########################################################################
## File I/O
##########################################################################
def load_hdr(filename, normalize=False, resize=True, resize_factor=2, verbose=False, clip_min=True):
    image = cv2.imread(filename, -1)[..., ::-1]
    if clip_min:
        image[image <= 0] = 200
        image[image == 200] = image.min()
    if verbose: print(f"[Dataset preparation]: HDR data range: {np.log10(image.min()):.4f} to {np.log10(image.max()):.4f}, {np.log2(image.max())-np.log2(image.min()):.2f} bits")
    if resize:
        image = cv2.resize(image, (int(image.shape[1]//resize_factor), int(image.shape[0]//resize_factor)))
    if normalize:
        image = np.log10(image)
        image = (image - image.min()) / (image.max() - image.min())
        image = 10**image
    return image

def load_img(filename, normalize=True, resize=True, resize_factor=2):
    image = cv2.imread(filename)[..., ::-1]
    if resize:
        image = cv2.resize(image, (int(image.shape[1]//resize_factor), int(image.shape[0]//resize_factor)))
    if normalize:
        image = image / 255.
    return image

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def write_hdr(filename, image):
    with open(filename, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))
        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)

##########################################################################
## Image Operations
##########################################################################
def random_crop(image, height, width, J, augment=False, center=False):
    if center and image.shape[0] > 1.5*height*J and image.shape[1] > 2*width*J and np.random.rand() > 0.33:
        row = int(np.random.randint(image.shape[0]*2//3-height*J+1) + image.shape[0]*1//3)
        col = int(np.random.randint(image.shape[1]//2-width*J+1) + image.shape[1]//4)
    else:
        row = np.random.randint(image.shape[0]-height*J+1)
        col = np.random.randint(image.shape[1]-width*J+1)
    image = image[row:row+height*J, col:col+width*J, ...]
    if augment:
        if np.random.rand() < 0.5:
            image = np.fliplr(image)
        k = int(np.random.rand()*4)
        image = np.rot90(image, k)
    if J != 1:
        image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LANCZOS4)
    return image

def modcrop(image, modulo, channel_first=False):
    shape = image.shape
    id0, id1 = (-2, -1) if channel_first else (0, 1)
    size0 = shape[id0] - shape[id0] % modulo
    size1 = shape[id1] - shape[id1] % modulo
    if len(image.shape) == 2:
        out = image[0:size0, 0:size1]
    elif channel_first:
        out = image[..., 0:size0, 0:size1]
    else:
        out = image[0:size0, 0:size1, ...]
    return out

def add_photon_noise(image, read_noise, nbits=10, alpha=1, random_alpha=True):
    pix_max = 2**nbits-1
    image -= image.min(); image /= image.max()
    image *= pix_max
    if random_alpha:
        alpha = np.random.random() * (1-0.25) + 0.25
    gt = image * alpha
    clean = image * alpha
    read_noise = np.random.random() * (160-135) + 135
    noise_sigma = np.sqrt(14 * clean + read_noise)
    noisy = clean + noise_sigma*np.random.randn(*clean.shape)
    noisy = np.round(noisy)
    noisy = np.clip(noisy, 0, pix_max)
    noisy = noisy / pix_max
    gt = np.clip(gt, 0, pix_max)
    gt = gt / pix_max
    if noisy.mean() > 0.99:
        print(alpha, image.mean(), clean.mean(), noisy.mean())
    return noisy, gt

def add_photon_noise_expand(image, read_noise, nbits=20, nbits_data=10, alpha=1, random_alpha=True):
    pix_max = 2**nbits-1
    data_max = 2**nbits_data-1
    image = image * data_max
    if random_alpha:
        alpha = np.random.triangular(1, 2, 8) 
    gt = image * alpha
    clean = image * alpha
    if np.random.rand() < 0.3:
        offset = 2 ** (np.random.rand() * 6 + 5)
        gt = gt + offset
        clean = clean + offset
    clean = clean / 0.5
    clean = np.clip(clean, 0, None)
    clean_poisson = np.clip(clean, 0, 1000)
    noisy = np.zeros_like(clean)
    noisy[clean < 1000]  = np.random.poisson(clean_poisson)[clean < 1000]
    noisy[clean >= 1000] = (clean + np.sqrt(clean) * np.random.randn(*clean.shape))[clean >= 1000] 
    noisy = noisy * 0.5 + read_noise*data_max*np.random.randn(*clean.shape) 
    noisy = np.round(noisy)
    noisy = np.clip(noisy, 0, pix_max)
    noisy = noisy / pix_max
    gt = np.clip(gt, 0, pix_max)
    gt = gt / pix_max
    return noisy, gt

def RGB2GBRG(im):
    H, W, C = im.shape
    if H == 3 and C != 3: 
        im = np.transpose(im, [1,2,0])
        H, W, C = im.shape
    hgb  = np.tile(np.array([[1,0],[0,0]]), (H//2, W//2))
    hb   = np.tile(np.array([[0,1],[0,0]]), (H//2, W//2))
    hr   = np.tile(np.array([[0,0],[1,0]]), (H//2, W//2))
    hgr  = np.tile(np.array([[0,0],[0,1]]), (H//2, W//2))
    CFA  = cv2.merge((hr, hgr+hgb, hb))
    xCFA = np.sum(im*CFA,axis=2).astype(np.float32)
    return xCFA

def GBRG2disp(im):
    if len(im.shape) == 3 and im.shape[0] == 1:
        channel_first = True
        im = im[0]
    else:
        channel_first = False
        if len(im.shape) == 3 and im.shape[2] == 1:
            im = im[:,:,0]
    assert len(im.shape) == 2
    H, W = im.shape
    if channel_first: 
        image_bayer = np.zeros((3, H, W))
        image_bayer[1, 0::2, 0::2] = im[0::2, 0::2]
        image_bayer[2, 0::2, 1::2] = im[0::2, 1::2]
        image_bayer[0, 1::2, 0::2] = im[1::2, 0::2]
        image_bayer[1, 1::2, 1::2] = im[1::2, 1::2]
    else: 
        image_bayer = np.zeros((H, W, 3))
        image_bayer[0::2, 0::2, 1] = im[0::2, 0::2]
        image_bayer[0::2, 1::2, 2] = im[0::2, 1::2]
        image_bayer[1::2, 0::2, 0] = im[1::2, 0::2]
        image_bayer[1::2, 1::2, 1] = im[1::2, 1::2]
    return image_bayer

def load_crops(directory, HDR_name, patch_sz, num_patch, J):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    folders = sorted(folders)
    size = num_patch*len(folders)
    crops = np.empty([size, 3, patch_sz, patch_sz], dtype=np.float32) 
    window_sz = patch_sz * J
    cnt = 0
    for folder in folders:
        HDR_fname = os.path.join(directory, folder, HDR_name)
        image = load_hdr(HDR_fname, verbose=False)
        [height, width, _]  = image.shape
        if height < window_sz or width < window_sz:
            continue
        assert image.min() >= 0

        for i in range(num_patch):
            crop = random_crop(image, patch_sz, patch_sz, J, augment=True, center=False)
            crop = np.clip(crop, image.min(), None) 
            crops[cnt,:,:,:] = crop.transpose((2,0,1))
            cnt += 1
    crops = crops[:cnt]
    shuf = np.arange(cnt)
    np.random.shuffle(shuf)
    crops = crops[shuf]
    print(f"[Dataset preparation]: Loaded {cnt} crops")
    return crops, cnt

def gen_data(crops, nbits_data, read_noise):
    B, C, H, W = crops.shape
    noisy, noisy_dm, gt = np.zeros((B,1,H,W)), np.zeros((B,3,H,W)), np.zeros((B,C,H,W))
    for i in range(len(crops)):
        noisy_rgb, gt[i] = add_photon_noise(crops[i], read_noise, nbits=nbits_data)
        noisy[i] = RGB2GBRG(noisy_rgb)
        noisy_dm[i] = np.transpose(GBTF_CFAInterpolation(noisy[i,0], BayerPatternFlag=2), (2,0,1))
    y = gt     
    x = noisy  
    xm = noisy_dm 
    return y, x, xm

def gen_data_expand(crops, nbits, nbits_data, read_noise):
    B, C, H, W = crops.shape
    noisy, noisy_dm, gt = np.zeros((B,1,H,W)), np.zeros((B,3,H,W)), np.zeros((B,C,H,W))
    for i in range(len(crops)):
        noisy_rgb, gt[i] = add_photon_noise_expand(crops[i], read_noise, nbits=nbits, nbits_data=nbits_data)
        noisy[i] = RGB2GBRG(noisy_rgb)
        noisy_dm[i] = np.transpose(GBTF_CFAInterpolation(noisy[i,0], BayerPatternFlag=2), (2,0,1))
    y = gt     
    x = noisy  
    xm = noisy_dm 
    return y, x, xm

##########################################################################
## Optimized PyTorch Dataset (Single-Item Loader)
##########################################################################
class HDRDataset(Dataset):
    def __init__(self, directory, patch_sz, num_patch, batch_sz, J, nbits, nbits_data, read_noise, do_expand):
        self.HDR_name = "ref_hdr_aligned.hdr"
        self.directory = directory
        self.crops, self.num_crops = load_crops(directory, self.HDR_name, patch_sz, num_patch, J)
        if do_expand:
            self.y, self.x, self.xm = gen_data_expand(self.crops, nbits, nbits_data, read_noise)
        else:
            self.y, self.x, self.xm = gen_data(self.crops, nbits_data, read_noise)
            
        self.patch_sz = patch_sz
        self.num_patch = num_patch
        self.J = J
        self.batch_sz = batch_sz
        self.nbits = nbits
        self.nbits_data = nbits_data
        self.read_noise = read_noise
        self.do_expand = do_expand

    def __len__(self):
        # OPTIMIZATION: Return the EXACT number of individual items!
        return self.num_crops

    def __getitem__(self, idx):
        # OPTIMIZATION: Return ONE sample at a time. The DataLoader will stack them!
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        y_single = self.y[idx]
        x_single = self.x[idx]
        xm_single = self.xm[idx]
        
        # Return clean standard tensors (The DataLoader will apply pin_memory safely)
        y_tensor = torch.tensor(y_single, dtype=torch.float32)
        x_tensor = torch.tensor(x_single, dtype=torch.float32)
        xm_tensor = torch.tensor(xm_single, dtype=torch.float32)
        
        return {'x': x_tensor, 'y': y_tensor, 'xm': xm_tensor}

    def regen_crops(self):
        del self.crops
        torch.cuda.empty_cache()
        self.crops, self.num_crops = load_crops(self.directory, self.HDR_name, self.patch_sz, self.num_patch, self.J)
        if self.do_expand:
            self.y, self.x, self.xm = gen_data_expand(self.crops, self.nbits, self.nbits_data, self.read_noise)
        else:
            self.y, self.x, self.xm = gen_data(self.crops, self.nbits_data, self.read_noise)

    def regen_noise(self):
        del self.xm
        del self.x
        del self.y
        torch.cuda.empty_cache()
        if self.do_expand:
            self.y, self.x, self.xm = gen_data_expand(self.crops, self.nbits, self.nbits_data, self.read_noise)
        else:
            self.y, self.x, self.xm = gen_data(self.crops, self.nbits_data, self.read_noise)