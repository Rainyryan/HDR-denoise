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

# # Add Poisson-Gaussian noise to [Channel Height Width] HDR images normalized to [0, 1]
# def add_photon_noise(image, read_noise, nbits=8, alpha=1, random_alpha=True):
# 	pix_max = 2**nbits-1
# 	# Randomize scene intensity, to be recovered
# 	image = image * pix_max
# 	if random_alpha:
# 		alpha = np.random.triangular(1, 2, 8) # more low-light
# 	gt = image * alpha
# 	clean = image * alpha
# 	# Rescale to photon - divide by QE (quantum efficiency), normally ~50%
# 	clean = clean / 0.5
# 	clean = np.clip(clean, 0, None)
# 	# Random photon count
# 	clean_poisson = np.clip(clean, 0, 1000)
# 	noisy = np.zeros_like(clean)
# 	noisy[clean < 1000]  = np.random.poisson(clean_poisson)[clean < 1000]
# 	noisy[clean >= 1000] = (clean + np.sqrt(clean) * np.random.randn(*clean.shape))[clean >= 1000] # Approx Poisson for large values
# 	# QE, ADC, read noise
# 	noisy = noisy * 0.5 + read_noise*pix_max*np.random.randn(*clean.shape) # read_noise related to gain 
# 	noisy = np.round(noisy)
# 	noisy = np.clip(noisy, 0, pix_max)
# 	# Back to [0, 1] image
# 	noisy = noisy / pix_max
# 	# Return gt as well
# 	gt = np.clip(gt, 0, pix_max)
# 	gt = gt / pix_max
# 	return noisy, gt


# Add Poisson-Gaussian noise to [Channel Height Width] HDR images normalized to [0, 1]
def add_photon_noise(image, read_noise, nbits=10, alpha=1, random_alpha=True):
	pix_max = 2**nbits-1
	image -= image.min(); image /= image.max()
	image *= pix_max
	# Randomize scene intensity, to be recovered
	if random_alpha:
		alpha = np.random.random() * (1-0.25) + 0.25 #np.random.triangular(0.5, 1, 2)
	gt = image * alpha
	clean = image * alpha
	# Note on read noise:
	#  As Enze measured on 2/5/2026, noise variance = 13.899 * X_dn + 154.981, while the line looks like
	#  it may be a bit steeper. So I will fit it as 14 * X_dn + Uniform(135, 160)
	#  sigma will be sqrt of the variance
	read_noise = np.random.random() * (160-135) + 135
	noise_sigma = np.sqrt(14 * clean + read_noise)
	noisy = clean + noise_sigma*np.random.randn(*clean.shape)
	noisy = np.round(noisy)
	noisy = np.clip(noisy, 0, pix_max)
	noisy = noisy / pix_max
	# Return gt as well
	gt = np.clip(gt, 0, pix_max)
	gt = gt / pix_max
	if noisy.mean() > 0.99:
		print(alpha, image.mean(), clean.mean(), noisy.mean())
	return noisy, gt

# Original expand: Poisson-Gaussian + optional offset, output in nbits (e.g. 20-bit)
def add_photon_noise_expand(image, read_noise, nbits=20, nbits_data=10, alpha=1, random_alpha=True):
	pix_max = 2**nbits-1
	data_max = 2**nbits_data-1
	# Randomize scene intensity, to be recovered
	image = image * data_max
	if random_alpha:
		alpha = np.random.triangular(1, 2, 8) # more low-light
	gt = image * alpha
	clean = image * alpha
	# Randomize expanded dynamic range - global, 30% high-light data, adds 5-bit to 11-bit
	if np.random.rand() < 0.3:
		offset = 2 ** (np.random.rand() * 6 + 5)
		gt = gt + offset
		clean = clean + offset
	# Rescale to photon - divide by QE (quantum efficiency), normally ~50%
	clean = clean / 0.5
	clean = np.clip(clean, 0, None)
	# Random photon count
	clean_poisson = np.clip(clean, 0, 1000)
	noisy = np.zeros_like(clean)
	noisy[clean < 1000]  = np.random.poisson(clean_poisson)[clean < 1000]
	noisy[clean >= 1000] = (clean + np.sqrt(clean) * np.random.randn(*clean.shape))[clean >= 1000] # Approx Poisson for large values
	# QE, ADC, read noise
	noisy = noisy * 0.5 + read_noise*data_max*np.random.randn(*clean.shape) # read_noise related to gain 
	noisy = np.round(noisy)
	noisy = np.clip(noisy, 0, pix_max)
	# Back to [0, 1] image
	noisy = noisy / pix_max
	# Return gt as well
	gt = np.clip(gt, 0, pix_max)
	gt = gt / pix_max
	return noisy, gt

# # Expand version using same camera noise model as add_photon_noise (variance = 14*X_dn + read_noise)
# def add_photon_noise_expand(image, read_noise, nbits=20, nbits_data=10, alpha=1, random_alpha=True):
# 	pix_max = 2**nbits - 1
# 	data_max = 2**nbits_data - 1
# 	image = np.asarray(image, dtype=np.float64).copy()
# 	im_min, im_max = image.min(), image.max()
# 	if im_max > im_min:
# 		image = (image - im_min) / (im_max - im_min)
# 	else:
# 		image = np.zeros_like(image)
# 	image *= data_max
# 	# Randomize scene intensity, to be recovered
# 	if random_alpha:
# 		alpha = np.random.random() * (1 - 0.25) + 0.25
# 	gt = image * alpha
# 	clean = image * alpha
# 	# Randomize expanded dynamic range - global, 30% high-light data, adds 5-bit to 11-bit
# 	if np.random.rand() < 0.3:
# 		offset = 2 ** (np.random.rand() * 6 + 5)
# 		gt = gt + offset
# 		clean = clean + offset
# 	# Same camera noise model as add_photon_noise: variance = 14 * X_dn + read_noise
# 	read_noise = np.random.random() * (160 - 135) + 135
# 	noise_sigma = np.sqrt(14 * clean + read_noise)
# 	noisy = clean + noise_sigma * np.random.randn(*clean.shape)
# 	noisy = np.round(noisy)
# 	noisy = np.clip(noisy, 0, pix_max)
# 	noisy = noisy / pix_max
# 	gt = np.clip(gt, 0, pix_max)
# 	gt = gt / pix_max
# 	return noisy, gt

def calculate_snr_map(noisy_image):
    """
    Calculates a 1-channel SNR map from a 3-channel noisy input using the 
    sensor physics model: Variance = 14 * X_dn + 147.5
    """
    # Convert from [C, H, W] to [H, W, C] for OpenCV
    img_np = np.transpose(noisy_image, (1, 2, 0))
    
    # Apply a mild Gaussian Blur to stabilize the proxy signal and suppress read noise spikes
    img_blurred = cv2.GaussianBlur(img_np, (5, 5), 1.0)
    if len(img_blurred.shape) == 2:
        img_blurred = np.expand_dims(img_blurred, axis=-1)
        
    # Convert back to [C, H, W]
    img_blurred = np.transpose(img_blurred, (2, 0, 1))
    
    # Apply the physics model: variance = 14 * Signal + mean(read_noise)
    variance = 14.0 * np.clip(img_blurred, 0, None) + 147.5
    sigma = np.sqrt(variance)
    
    # Calculate SNR
    snr = img_blurred / (sigma + 1e-6)
    
    # Average across the 3 RGB channels to get a 1-channel map [1, H, W]
    snr_map = np.mean(snr, axis=0, keepdims=True)
    
    # Normalize to [0, 1] for stable gradient flow
    snr_max = snr_map.max()
    if snr_max > 0:
        snr_map = snr_map / snr_max
        
    return snr_map

def RGB2GBRG(im):
	H, W, C = im.shape
	if H == 3 and C != 3: # channel first
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



# Generate images for testing
def load_test_samples(directory, HDR_name, nbits, read_noise, alpha=4):
	folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
	folders = sorted(folders)
	size = len(folders)
	samples = []
	for folder in folders:
		HDR_fname = os.path.join(directory, folder, HDR_name)
		image = load_hdr(HDR_fname, resize=False)
		image = modcrop(image, 2)
		# print(image.min(), image.max())
		# image_temp = image * (1/(image.min()*(2**nbits-1)))
		# print(image_temp.min(), image_temp.max())
		# Noisy images
		if alpha is None:
			noisy, gt = add_photon_noise(image, read_noise, nbits=nbits)
		elif alpha == "full":
			noisy, gt = add_photon_noise(image, read_noise, nbits=nbits, alpha=1/(image.min()*(2**nbits-1)), random_alpha=False)
		else:
			noisy, gt = add_photon_noise(image, read_noise, nbits=nbits, alpha=alpha, random_alpha=False)
		# noisy, gt = noisy * (2**nbits-1) / (2**20-1), gt * (2**nbits-1) / (2**20-1)
		noisy, gt = noisy * (2**nbits-1) / (2**10-1), gt * (2**nbits-1) / (2**10-1)
		noisy_bayer = RGB2GBRG(noisy)
		noisy_bayer = noisy_bayer[np.newaxis,...]
		noisy_dm = GBTF_CFAInterpolation(noisy_bayer[0], BayerPatternFlag=2)
		gt = np.transpose(gt, (2,0,1))
		noisy = np.transpose(noisy, (2,0,1))
		noisy_dm = np.transpose(noisy_dm, (2,0,1))
		samples.append({'x': noisy_bayer, 'xm': noisy_dm, 'xt': noisy, 'y': gt})
	return samples

def load_test_samples_HDR(HDR_names, nbits, read_noise, alpha=4):
	samples = []
	for HDR_name in HDR_names:
		image = load_hdr(HDR_name, resize=False, verbose=True)
		image = image / image.max()
		image = modcrop(image, 2)
		# Noisy images
		if alpha is None:
			noisy, gt = add_photon_noise(image, read_noise, nbits=nbits)
		elif alpha == "full":
			noisy, gt = add_photon_noise(image, read_noise, nbits=nbits, alpha=1/(image.min()*(2**nbits-1)), random_alpha=False)
		else:
			noisy, gt = add_photon_noise(image, read_noise, nbits=nbits, alpha=alpha, random_alpha=False)
		noisy_bayer = RGB2GBRG(noisy)
		noisy_bayer = noisy_bayer[np.newaxis,...]
		noisy_dm = GBTF_CFAInterpolation(noisy_bayer[0], BayerPatternFlag=2)
		gt = np.transpose(gt, (2,0,1))
		noisy = np.transpose(noisy, (2,0,1))
		noisy_dm = np.transpose(noisy_dm, (2,0,1))
		samples.append({'x': noisy_bayer, 'xm': noisy_dm, 'xt': noisy, 'y': gt})
	return samples

def load_test_samples_LDR(LDR_names, nbits, read_noise, alpha=4, gamma=4.2):
	samples = []
	for LDR_name in LDR_names:
		image = load_img(LDR_name, resize=False)
		image = modcrop(image, 2)
		print(image.min(), image.max(), image.mean())
		image = np.clip(image, 1/255., None)
		image = image ** gamma
		print(image.min(), image.max(), image.mean())
		print(1/image.min())
		# Noisy images
		if alpha is None:
			noisy, gt = add_photon_noise(image, read_noise, nbits=nbits)
		elif alpha == "full":
			noisy, gt = add_photon_noise(image, read_noise, nbits=nbits, alpha=1/image.min(), random_alpha=False)
		else:
			noisy, gt = add_photon_noise(image, read_noise, nbits=nbits, alpha=alpha, random_alpha=False)
		noisy_bayer = RGB2GBRG(noisy)
		noisy_bayer = noisy_bayer[np.newaxis,...]
		noisy_dm = GBTF_CFAInterpolation(noisy_bayer[0], BayerPatternFlag=2)
		gt = np.transpose(gt, (2,0,1))
		noisy = np.transpose(noisy, (2,0,1))
		noisy_dm = np.transpose(noisy_dm, (2,0,1))
		samples.append({'x': noisy_bayer, 'xm': noisy_dm, 'xt': noisy, 'y': gt})
	return samples

# Generate batches for training
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
		[height, width, _]	= image.shape
		if height < window_sz or width < window_sz:
			continue
		assert image.min() >= 0

		for i in range(num_patch):
			crop = random_crop(image, patch_sz, patch_sz, J, augment=True, center=False)
			crop = np.clip(crop, image.min(), None) # When J > 1, openCV resize may generate negative values in GT images
			crops[cnt,:,:,:] = crop.transpose((2,0,1))
			cnt += 1
	crops = crops[:cnt]
	shuf = np.arange(cnt)
	np.random.shuffle(shuf)
	crops = crops[shuf]
	print(f"[Dataset preparation]: Loaded {cnt} crops")
	return crops, cnt


# def gen_data(crops, nbits_data, read_noise):
# 	B, C, H, W = crops.shape
# 	noisy, noisy_dm, gt = np.zeros((B,1,H,W)), np.zeros((B,3,H,W)), np.zeros((B,C,H,W))
# 	for i in range(len(crops)):
# 		noisy_rgb, gt[i] = add_photon_noise(crops[i], read_noise, nbits=nbits_data)
# 		noisy[i] = RGB2GBRG(noisy_rgb)
# 		noisy_dm[i] = np.transpose(GBTF_CFAInterpolation(noisy[i,0], BayerPatternFlag=2), (2,0,1))
# 	y = gt     # clean HDR
# 	x = noisy  # noisy HDR
# 	xm = noisy_dm # noisy demosaicked HDR
# 	return y, x, xm

# def gen_data_expand(crops, nbits, nbits_data, read_noise):
# 	B, C, H, W = crops.shape
# 	noisy, noisy_dm, gt = np.zeros((B,1,H,W)), np.zeros((B,3,H,W)), np.zeros((B,C,H,W))
# 	for i in range(len(crops)):
# 		noisy_rgb, gt[i] = add_photon_noise_expand(crops[i], read_noise, nbits=nbits, nbits_data=nbits_data)
# 		noisy[i] = RGB2GBRG(noisy_rgb)
# 		noisy_dm[i] = np.transpose(GBTF_CFAInterpolation(noisy[i,0], BayerPatternFlag=2), (2,0,1))
# 	y = gt     # clean HDR
# 	x = noisy  # noisy HDR
# 	xm = noisy_dm # noisy demosaicked HDR
# 	return y, x, xm

def gen_data(crops, nbits_data, read_noise):
    B, C, H, W = crops.shape
    noisy, noisy_dm, gt = np.zeros((B,1,H,W)), np.zeros((B,3,H,W)), np.zeros((B,C,H,W))
    snr_maps = np.zeros((B,1,H,W)) # NEW: Allocate SNR array
    
    for i in range(len(crops)):
        noisy_rgb, gt[i] = add_photon_noise(crops[i], read_noise, nbits=nbits_data)
        noisy[i] = RGB2GBRG(noisy_rgb)
        noisy_dm[i] = np.transpose(GBTF_CFAInterpolation(noisy[i,0], BayerPatternFlag=2), (2,0,1))
        
        # NEW: Calculate SNR map from the demosaicked noisy image
        snr_maps[i] = calculate_snr_map(noisy_dm[i])
        
    y = gt     # clean HDR
    x = noisy  # noisy HDR
    xm = noisy_dm # noisy demosaicked HDR
    return y, x, xm, snr_maps # NEW: Return snr_maps

def gen_data_expand(crops, nbits, nbits_data, read_noise):
    B, C, H, W = crops.shape
    noisy, noisy_dm, gt = np.zeros((B,1,H,W)), np.zeros((B,3,H,W)), np.zeros((B,C,H,W))
    snr_maps = np.zeros((B,1,H,W)) # NEW: Allocate SNR array
    
    for i in range(len(crops)):
        noisy_rgb, gt[i] = add_photon_noise_expand(crops[i], read_noise, nbits=nbits, nbits_data=nbits_data)
        noisy[i] = RGB2GBRG(noisy_rgb)
        noisy_dm[i] = np.transpose(GBTF_CFAInterpolation(noisy[i,0], BayerPatternFlag=2), (2,0,1))
        
        # NEW: Calculate SNR map from the demosaicked noisy image
        snr_maps[i] = calculate_snr_map(noisy_dm[i])
        
    y = gt     # clean HDR
    x = noisy  # noisy HDR
    xm = noisy_dm # noisy demosaicked HDR
    return y, x, xm, snr_maps # NEW: Return snr_maps

class HDRDataset(Dataset):
    def __init__(self, directory, patch_sz, num_patch, batch_sz, J, nbits, nbits_data, read_noise, do_expand):
        self.HDR_name = "ref_hdr_aligned.hdr"
        self.directory = directory
        self.crops, self.num_crops = load_crops(directory, self.HDR_name, patch_sz, num_patch, J)
        if do_expand:
            # NEW: Unpack the 4th variable (snr)
            self.y, self.x, self.xm, self.snr = gen_data_expand(self.crops, nbits, nbits_data, read_noise)
        else:
            self.y, self.x, self.xm, self.snr = gen_data(self.crops, nbits_data, read_noise)
            
        # Store param for data generating
        self.patch_sz = patch_sz
        self.num_patch = num_patch
        self.J = J
        # Store other param
        self.batch_sz = batch_sz
        self.nbits = nbits
        self.nbits_data = nbits_data
        self.read_noise = read_noise
        self.do_expand = do_expand

    def __len__(self):
        return int(np.floor(self.num_crops / float(self.batch_sz)))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        y = self.y[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        x = self.x[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        xm = self.xm[idx * self.batch_sz:(idx + 1) * self.batch_sz]
        snr = self.snr[idx * self.batch_sz:(idx + 1) * self.batch_sz] # NEW: Grab the SNR batch
        
        device = torch.device("cuda:0")
        y = torch.tensor(y, dtype=torch.float).to(device)
        x = torch.tensor(x, dtype=torch.float).to(device)
        xm = torch.tensor(xm, dtype=torch.float).to(device)
        snr = torch.tensor(snr, dtype=torch.float).to(device) # NEW: Send to device
        
        # NEW: Add snr_map to the returned dictionary
        sample = {'x': x, 'y': y, 'xm': xm, 'snr_map': snr}
        return sample

    def regen_crops(self):
        del self.crops
        torch.cuda.empty_cache()
        self.crops, self.num_crops = load_crops(self.directory, self.HDR_name, self.patch_sz, self.num_patch, self.J)
        if self.do_expand:
            # NEW
            self.y, self.x, self.xm, self.snr = gen_data_expand(self.crops, self.nbits, self.nbits_data, self.read_noise)
        else:
            self.y, self.x, self.xm, self.snr = gen_data(self.crops, self.nbits_data, self.read_noise)

    def regen_noise(self):
        del self.xm
        del self.x
        del self.y
        # NEW: delete old snr
        del self.snr 
        torch.cuda.empty_cache()
        if self.do_expand:
            self.y, self.x, self.xm, self.snr = gen_data_expand(self.crops, self.nbits, self.nbits_data, self.read_noise)
        else:
            self.y, self.x, self.xm, self.snr = gen_data(self.crops, self.nbits_data, self.read_noise)
