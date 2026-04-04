import os
import glob
import numpy as np
import time
import cv2
import lpips
# Prevent trained model overwrite
from datetime import datetime
# PyTorch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import save_image

from HDR_model_switch import HDR_model
from HDR_dataset import *
from distiller import RestormerDistiller

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
def hdr_tonemap(hdr_image, nbits=10):
	mu = 2**nbits-1
	return torch.log10(1.0 + mu * hdr_image) / torch.log10(torch.tensor(1.0 + mu))

def hdr_tonemap_np(hdr_image, nbits=10):
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
	tonemap = cv2.createTonemapDurand(2.2)
	tonemapped_image = tonemap.process(img)
	return tonemapped_image


if __name__ == "__main__":
	# Set up gpu
	device = torch.device("cuda:0")

	# Set seed
	seed = 21 
	torch.manual_seed(seed)
	np.random.seed(seed)

	# Hyperparameters
	phase = "train"
	# Filenames
	save_folder = "distill_models_%s/" % datetime.now().strftime("%Y%m%d_%H%M%S")
	pretrained_path = "./distill_models_3/preexpand_hdr_best.pth"
	create_folder(save_folder)
	save_path_preexpand = save_folder + "preexpand_hdr.pth"
	best_save_path_preexpand = save_folder + "preexpand_hdr_best.pth"
	save_path = save_folder + "hdr.pth"
	best_save_path = save_folder + "hdr_best.pth"
	last_path = None
 
	# Data loading
	nbits = 20 
	nbits_data = 10
	batch_sz = 8
	read_noise = 0
	regen_crops_every_epoch = 10 
	regen_noise_every_epoch = 5  
	# Optimizing
	lr = 2e-4 
	optimizer_choice = "Adam" 
	# Loss schedule
	num_epochs = 300
	start_epoch = 0 
	epoch_expand_mode = 999 
	# Misc.
	best_running_psnr = 10.0
	last_epoch_psnr, last_epoch_loss = 0.0, 1e6

	####################################################################################################################################################
	## Training
	####################################################################################################################################################
	if phase == "train":
		print("Preparing models for Distillation")
		
		# 1. Initialize Teacher and Student
		teacher_model = HDR_model(dim=32, num_blocks=[4,4,4,4]) 
		student_model = HDR_model(dim=8, num_blocks=[2,2,2,1]) 

		# Load Teacher weights (Assuming you have a trained teacher here)
		teacher_checkpoint_path = "./models_rest_300_999_hi_noise_dim32_4444/preexpand_hdr_best.pth"
		if os.path.exists(teacher_checkpoint_path):
			print(f"Loading Teacher weights from {teacher_checkpoint_path}")
			checkpoint = torch.load(teacher_checkpoint_path)
			teacher_model.load_state_dict(checkpoint['model_state_dict'])
		else:
			print(f"WARNING: Teacher weights not found at {teacher_checkpoint_path}. Using random weights.")

		# 2. Wrap in Distiller
		distiller = RestormerDistiller(
			teacher_model=teacher_model, 
			student_model=student_model, 
			teacher_dim=32, 
			student_dim=8
		).to(device)

		# Ensure optimizer only targets the student and the alignment layers
		if optimizer_choice == "RMSprop":
			optimizer = torch.optim.RMSprop([p for p in distiller.parameters() if p.requires_grad], lr=learning_rate(lr, start_epoch))
		else:
			optimizer = torch.optim.Adam([p for p in distiller.parameters() if p.requires_grad], lr=learning_rate(lr, start_epoch))

		loss_lpips = lpips.LPIPS(net='vgg').to(device)

		# Write log
		logfile = "log.txt"
		write_log(logfile, "", newfile=True)
		pytorch_total_params = sum(p.numel() for p in distiller.student.parameters() if p.requires_grad)
		write_log(logfile, f"Total Student params = {pytorch_total_params}")
		training_t0 = time.time()

		# Checkpoint Loading logic for Student/Distiller
		if os.path.exists(pretrained_path):
			print("Loading pretrained student model")
			checkpoint = torch.load(pretrained_path)
			if 'student_state_dict' in checkpoint:
				distiller.student.load_state_dict(checkpoint['student_state_dict'])
			elif 'model_state_dict' in checkpoint: # Fallback if loading a standard trained model
				distiller.student.load_state_dict(checkpoint['model_state_dict'])

		if os.path.exists(pretrained_path):
			print("Loading continued training model")
			checkpoint = torch.load(pretrained_path)
			
			# Load the full distiller state if available (preserves alignment layers)
			if 'distiller_state_dict' in checkpoint:
				distiller.load_state_dict(checkpoint['distiller_state_dict'])
			else:
				distiller.student.load_state_dict(checkpoint['student_state_dict'])
				
			start_epoch = checkpoint.get('epoch', 0)
			if 'optimizer_state_dict' in checkpoint:
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			print(f"Resuming from epoch {start_epoch}")

		distiller.train()

		# Dataset
		print("Creating dataset")
		directory = "/mnt/c/Users/rc615/Documents/GitHub/DeepHDR/dataset/train"
		dataset = HDRDataset(directory, patch_sz=128, num_patch=64, batch_sz=batch_sz, J=1, 
							nbits=nbits, nbits_data=nbits_data, read_noise=read_noise, do_expand=False)

		# Train
		print("Training started")

		for epoch in range(start_epoch, num_epochs):
			# Regen crops or noise every several epochs
			if epoch >= epoch_expand_mode:
				dataset.do_expand = True
			if regen_crops_every_epoch != 0 and epoch % regen_crops_every_epoch == 0 and epoch != 0:
				dataset.regen_crops()
			if regen_noise_every_epoch != 0 and epoch % regen_noise_every_epoch == 0 and epoch != 0:
				dataset.regen_noise()
			
			improved = False
			while not improved:
				running_loss = 0.0
				running_psnr = 0.0
				t1 = time.time()
				
				for i in range(len(dataset)):
					sample = dataset[i]
					x, y, xm = sample["x"], sample["y"], sample["xm"]

					optimizer.zero_grad()
					
					# Forward pass through distiller
					s_out, t_out, s_maps_aligned, t_maps = distiller(xm)
					
					# A. Task Loss (existing HDR tonemap loss against Ground Truth)
					loss_task = (hdr_tonemap(s_out, nbits) - hdr_tonemap(y, nbits)) ** 2
					loss_task = torch.mean(loss_task)
					
					# B. Output Distillation Loss (Student output mimicking Teacher output)
					loss_out_distill = F.l1_loss(s_out, t_out)
					
					# C. Feature Distillation Loss (Matching intermediate U-Net layers)
					loss_feat = 0
					for key in ["x_level1", "x_level2", "x_latent"]:
						loss_feat += F.mse_loss(s_maps_aligned[key], t_maps[key])
						
					# D. Total Loss (schedule)
					alpha = 0.0   # Weight for output distillation
					beta = 0.05   # Weight for feature distillation
					
					ttl_loss = loss_task + (alpha * loss_out_distill) + (beta * loss_feat)
					
					# Backward pass
					ttl_loss.backward()
					optimizer.step()
					
					running_loss += ttl_loss.item()
					running_psnr += batch_psnr(s_out, y)
					print_running_loss(running_loss, running_psnr, batch_sz, i)

				# Save tmp image (Fixed variable name from y_pred to s_out)
				output_dir = "tmp/"
				create_folder(output_dir)
				for i in range(min(5, batch_sz)):
					image = hdr_tonemap_np(y[i].detach().cpu().numpy())
					cv2.imwrite(f"{output_dir}{i}np_gt.png", np.transpose(image, [1,2,0])[:,:,::-1]*255)
					image = hdr_tonemap_np(s_out[i].detach().cpu().numpy())
					cv2.imwrite(f"{output_dir}{i}np_pred.png", np.transpose(image, [1,2,0])[:,:,::-1]*255)
					image = hdr_tonemap_np(x[i].detach().cpu().numpy())
					cv2.imwrite(f"{output_dir}{i}np_noisy.png", np.transpose(image, [1,2,0])[:,:,::-1]*255)
					image = hdr_tonemap_np(xm[i].detach().cpu().numpy())
					cv2.imwrite(f"{output_dir}{i}np_dm.png", np.transpose(image, [1,2,0])[:,:,::-1]*255)

				t2 = time.time()
				this_epoch_loss = running_loss/(batch_sz*len(dataset))
				this_epoch_psnr = running_psnr/len(dataset)
				epoch_loss_str = print_epoch_loss(epoch+1, this_epoch_loss, this_epoch_psnr, t2-t1)
				write_log(logfile, epoch_loss_str, end="")

				improved = True
				
				# Rollback condition if loss explodes
				if ((epoch < epoch_expand_mode or epoch >= epoch_expand_mode + 10) and last_epoch_psnr > 10 and this_epoch_loss > 2 * last_epoch_loss) or \
					(epoch >= epoch_expand_mode and epoch < epoch_expand_mode + 10 and this_epoch_loss > 1000):
					print(last_epoch_psnr, this_epoch_psnr, "do not save")
					improved = False
					dataset.regen_crops()
					dataset.regen_noise()
					
					# Load FULL distiller state so alignment layers aren't scrambled
					checkpoint = torch.load(last_path)
					if 'distiller_state_dict' in checkpoint:
						distiller.load_state_dict(checkpoint['distiller_state_dict'])
					else:
						distiller.student.load_state_dict(checkpoint['student_state_dict'])
				else:
					last_epoch_psnr = this_epoch_psnr
					last_epoch_loss = this_epoch_loss

				# Save model
				if improved:
					sp = save_path if epoch >= epoch_expand_mode else save_path_preexpand
					bsp = best_save_path if epoch >= epoch_expand_mode else best_save_path_preexpand
					last_path = sp
					
					# Save both the student (for inference) and the full distiller (for resuming training)
					save_dict = {
						'epoch': epoch+1,
						'student_state_dict': distiller.student.state_dict(),
						'distiller_state_dict': distiller.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'loss': this_epoch_loss
					}
					
					torch.save(save_dict, sp)
					if running_psnr > best_running_psnr:
						best_running_psnr = running_psnr
						torch.save(save_dict, bsp)

		training_t1 = time.time()
		write_log(logfile, "\nTotal training time = %.2f" % (training_t1 - training_t0))
		print("Training finished")