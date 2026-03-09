import os
import glob
import numpy as np
import time
import cv2
# Prevent trained model overwrite
from datetime import datetime
# PyTorch
import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image

#from model import *
from HDR_model_switch import HDR_model
from HDR_dataset import *

##########################################################################
## Main function util
##########################################################################
def print_running_loss(running_loss, running_psnr, batch_sz, i, print_every_epoch=10):
	psnr_str = "\tPSNR = %.2f" % (running_psnr/(i+1)) if running_psnr else ""
	if i % print_every_epoch == (print_every_epoch-1):
		print("\r\tLoss = %15.4f\t%s" % (running_loss/(batch_sz*(i+1)), psnr_str) , end="")

def print_epoch_loss(epoch_num, avg_loss, avg_psnr, time_spent=None):
	time_spent = "Unknown" if time_spent is None else "%f"%time_spent
	psnr_str = "\tPSNR = %.2f" % avg_psnr if avg_psnr else ""
	epoch_loss = "\rEpoch #%d: Loss = %12.4f \t%s\t(Time: %s)" % (epoch_num, avg_loss, psnr_str, time_spent)
	print(epoch_loss)
	return epoch_loss

def write_log(filename, s, newfile=False, end="\n"):
	if newfile:
		with open(filename, "w") as ofile:
			pass
	with open(filename, "a") as ofile:
		ofile.write("%s%s" % (s, end))


##########################################################################
## Training util
##########################################################################
def hdr_tonemap(hdr_image, nbits=18):
	mu = 2**nbits-1
	return torch.log10(1.0 + mu * hdr_image) / torch.log10(torch.tensor(1.0 + mu))

def hdr_tonemap_np(hdr_image, nbits=18):
	mu = 2**nbits-1
	return np.log10(1.0 + mu * hdr_image) / np.log10(1.0 + mu)

# Metrics
def batch_psnr(img, gt, data_range=1):
	def compare_psnr(img1, img2, data_range=1):
		mse = np.mean((img1.astype(float) - img2.astype(float))**2)
		if mse == 0:
			return 100
		return 10 * np.log10((data_range**2) / mse)
	img = img.data.cpu().numpy().astype(np.float32)
	gt = gt.data.cpu().numpy().astype(np.float32)
	PSNR = 0
	for i in range(img.shape[0]):
		PSNR += compare_psnr(gt[i,:,:,:], img[i,:,:,:], data_range=1)
	return (PSNR/img.shape[0])

# Learning rate
def learning_rate(lr, epoch):
	#factor =  [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 80, 120, 120, 160, 160, 200, 200]
	factor =  [1, 1, 2, 2, 5, 5, 10, 20, 30, 50, \
			   2, 2, 5, 5, 10, 20, 30, 50, 60, 70, \
			   5, 5, 10, 20, 30, 50, 60, 70, 80, 100, \
			   10, 20, 50, 70, 100, 120, 150, 170, 200, 300]
	return lr / factor[epoch//10] if epoch//10 < len(factor) else lr / 400

# Given numpy image (in channel-first format), return the tonemap image
def to_tonemap(img):
	assert len(img.shape) == 3
	img = np.transpose(img, [1,2,0])[:,:,::-1]
	img = np.clip(img, 0, None)
	img = np.float32(img)
	# tonemap = cv2.createTonemapDurand(2.2)
	tonemap = cv2.createTonemap(gamma=2.2)
	tonemapped_image = tonemap.process(img)
	return tonemapped_image



if __name__ == "__main__":
	#torch.autograd.set_detect_anomaly(True)
	# Set up gpu
	device = torch.device("cuda:0")

	# Set seed
	seed = 1
	torch.manual_seed(seed)
	np.random.seed(seed)


	# Hyperparameters
	# Phase
	phase = "test"
	# Filenames
	save_folder = "models_rest_150_100_hi_noise_dim32_4444/"
	create_folder(save_folder)
	save_path = save_folder + "hdr_best.pth"
	output_setting = "rest_150_100_hi_noise_dim32_4444"
	# Data loading
	alpha = 1
	nbits = 10
	read_noise = 0 

	####################################################################################################################################################
	## Testing
	####################################################################################################################################################

	if phase == "test":
		patch_sz = 512

		# Load model
		model = HDR_model()
		checkpoint = torch.load(save_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.eval()
		model.to(device)

		for alpha in [1, 3, "full"]:
			# Output folder
			if alpha == "full":
				output_name = "full"
			elif type(alpha) == type(0):
				output_name = "a%d" % int(alpha)
			else:
				output_name = "a%.2f" % alpha
			output_dir = f"results_{nbits}bit_{output_setting}/{output_name}/"
			create_folder(output_dir)
			logfile = output_dir + "psnr.txt"

			# Load test samples
			directory = "/mnt/c/Users/rc615/Documents/GitHub/DeepHDR/dataset/test"
			HDR_name = "ref_hdr_aligned.hdr"
			samples = load_test_samples(directory, HDR_name, nbits, read_noise, alpha=alpha)
			running_psnr = 0.0
			
			for i, sample in enumerate(samples):
				x, y = sample["xm"], sample["y"]
				print(x.shape, x.min(), x.max())
				print(y.shape, y.min(), y.max())
				x = modcrop(x, 64, channel_first=True)
				y = modcrop(y, 64, channel_first=True)
				C, H, W = x.shape
				x = torch.tensor(x[np.newaxis,...], dtype=torch.float).to(device)
				y_pred = np.zeros_like(y)

				pd1 = 8; pd2 = pd1*2
				ii_idx = 0
				for ii in range(0, H, patch_sz-pd2):
					jj_idx = 0
					for jj in range(0, W, patch_sz-pd2):
						x_patch = torch.tensor(np.zeros((1,C,patch_sz,patch_sz)), dtype=torch.float).to(device)
						ii_end, jj_end = min(ii+patch_sz,H), min(jj+patch_sz,W)
						x_patch[:, :, :(ii_end-ii), :(jj_end-jj)] = x[:, :, ii:ii_end, jj:jj_end]
						with torch.no_grad():
							y_patch = model(x_patch)
						y_patch = y_patch.detach().cpu().numpy()
						y_patch = y_patch[:, :, :(ii_end-ii), :(jj_end-jj)]

						y_pred[:, ii_idx*(patch_sz-pd2)+pd1:(ii_idx+1)*(patch_sz-pd2)+pd1, jj_idx*(patch_sz-pd2)+pd1:(jj_idx+1)*(patch_sz-pd2)+pd1] = y_patch[0, :, pd1:patch_sz-pd1, pd1:patch_sz-pd1]
						if ii_idx == 0: y_pred[:, :pd1, jj_idx*(patch_sz-pd2)+pd1:(jj_idx+1)*(patch_sz-pd2)+pd1] = y_patch[0, :, :pd1, pd1:patch_sz-pd1]
						if jj_idx == 0: y_pred[:, ii_idx*(patch_sz-pd2)+pd1:(ii_idx+1)*(patch_sz-pd2)+pd1, :pd1] = y_patch[0, :, pd1:patch_sz-pd1, :pd1]
						if ii_idx == 0 and jj_idx == 0: y_pred[:, :pd1, :pd1] = y_patch[0, :, :pd1, :pd1]
						jj_idx += 1
					ii_idx += 1


				y_pred = torch.tensor(y_pred[np.newaxis,...], dtype=torch.float).to(device)
				y = torch.tensor(y[np.newaxis,...], dtype=torch.float).to(device)
				running_psnr += batch_psnr(y_pred, y)

				# Save images (noisy, prediction, ground truth)
				image = np.transpose(y[0].detach().cpu().numpy(), [1,2,0])[:,:,::-1]
				image = np.clip(hdr_tonemap_np(image)*255, 0, 255).astype(np.uint8)
				cv2.imwrite(os.path.join(output_dir, "%d_tonemap_gt.png"%i), image)

				image = np.transpose(y[0].detach().cpu().numpy(), [1,2,0])[:,:,::-1]
				cv2.imwrite(os.path.join(output_dir, "%d_gt.png"%i), image*255)
				image = to_tonemap(y[0].detach().cpu().numpy())
				cv2.imwrite(os.path.join(output_dir, "%d_tonemap_cv_gt.png"%i), image*255)

				image = np.transpose(y_pred[0].detach().cpu().numpy(), [1,2,0])[:,:,::-1]
				# cv2.imwrite(os.path.join(output_dir, "%d_pred.png"%i), image*255)
				image = np.clip(hdr_tonemap_np(image)*255, 0, 255).astype(np.uint8)
				cv2.imwrite(os.path.join(output_dir, "%d_tonemap_pred.png"%i), image)

				image = np.transpose(x[0].detach().cpu().numpy(), [1,2,0])[:,:,::-1]
				cv2.imwrite(os.path.join(output_dir, "%d_noisy.png"%i), image*255)
				image = np.clip(hdr_tonemap_np(image)*255, 0, 255).astype(np.uint8)
				cv2.imwrite(os.path.join(output_dir, "%d_tonemap_noisy.png"%i), image)

				image = np.transpose(sample["x"], [1,2,0])
				image = GBRG2disp(image)
				cv2.imwrite(os.path.join(output_dir, "%d_noisy_bayer.png"%i), image[:,:,::-1]*255)
				image = np.clip(hdr_tonemap_np(image[:,:,::-1])*255, 0, 255).astype(np.uint8)
				cv2.imwrite(os.path.join(output_dir, "%d_tonemap_noisy_bayer.png"%i), image)

				image = to_tonemap(y_pred[0].detach().cpu().numpy())
				cv2.imwrite(os.path.join(output_dir, "%d_tonemap_cv_pred.png"%i), image*255)

				image = np.transpose(sample["xt"], [1,2,0])
				cv2.imwrite(os.path.join(output_dir, "%d_noisy_color.png"%i), image[:,:,::-1]*255)
				image = np.clip(hdr_tonemap_np(image)*255, 0, 255).astype(np.uint8)
				cv2.imwrite(os.path.join(output_dir, "%d_tonemap_noisy_color.png"%i), image[:,:,::-1])


			print(alpha)
			print("Avg PSNR = %.4f dB"%(running_psnr/i))
			write_log(logfile, "\nTest Avg PSNR = %.4f dB"%(running_psnr/i))
			


