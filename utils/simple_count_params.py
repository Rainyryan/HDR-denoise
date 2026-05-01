"""
Simple Model Parameter Counter
Quick and simple parameter counting script
"""
import torch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HDR_model_switch import HDR_model
from HDR_model_hybrid_student import Hybrid_Student_HDR


def count_and_print(model, model_name="Model", log_file=None):
	"""Count parameters and print/log results"""
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	size_mb = total * 4 / (1024 * 1024)  # FP32
	
	print(f"\n{'='*60}")
	print(f"{model_name}")
	print(f"{'='*60}")
	print(f"Total Parameters:      {total:>15,}")
	print(f"Trainable Parameters:  {trainable:>15,}")
	print(f"Model Size (FP32):     {size_mb:>15.2f} MB")
	print(f"{'='*60}\n")
	
	if log_file:
		with open(log_file, 'a') as f:
			f.write(f"{model_name}\n")
			f.write(f"Total Parameters: {total:,}\n")
			f.write(f"Trainable Parameters: {trainable:,}\n")
			f.write(f"Model Size (FP32): {size_mb:.2f} MB\n")
			f.write("-" * 60 + "\n\n")
		print(f"✓ Logged to {log_file}")


if __name__ == "__main__":
	# Count original model
	print("Counting Original Model...")
	model_orig = HDR_model(dim=32, num_blocks=[4,4,4,4])
	count_and_print(model_orig, "Original HDR Model", "params_log.txt")
	
	# Count lightweight model
	print("Counting Lightweight Model...")
	model_light = Hybrid_Student_HDR(dim=8, num_blocks=[2,2,2,1], num_refinement_blocks=2, heads=[1,2,4,8])
	count_and_print(model_light, "Lightweight HDR Model", "params_log.txt")
	
	# Calculate reduction
	orig_params = sum(p.numel() for p in model_orig.parameters())
	light_params = sum(p.numel() for p in model_light.parameters())
	reduction = orig_params / light_params
	
	print(f"\n{'='*60}")
	print("COMPARISON")
	print(f"{'='*60}")
	print(f"Original:    {orig_params:>15,} parameters")
	print(f"Lightweight: {light_params:>15,} parameters")
	print(f"Reduction:    {reduction:>15.2f}x")
	print(f"{'='*60}\n")
