import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_log(file_path):
    epochs, losses, psnrs = [], [], []
    pattern = re.compile(r"Epoch #(\d+): Loss =\s+([\d\.]+)\s+PSNR =\s+([\d\.]+)")
    
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                psnrs.append(float(match.group(3)))
    
    # Take the last entry for each epoch to handle re-logs
    df = pd.DataFrame({'Epoch': epochs, 'Loss': losses, 'PSNR': psnrs})
    return df.groupby('Epoch').last().reset_index()

# Load data
logs = {
    'Restormer (150 Epochs)': 'log_rest_150_100.txt',
    'Restormer (400 Epochs)': 'log_rest_400_200.txt',
    # 'SwinIR (150 Epochs)': 'log_swin_150_100.txt',
    # 'Code (150 Epochs)': 'log_code_100_50.txt',
    # 'Rest (150 hi noise)': 'log_rest_150_100_hi_noise.txt',
    'Rest (150 16 4442 hi noise)': 'log_rest_150_100_hi_noise_dim16_4442.txt',
    'Rest (200 32 4444 hi noise)': 'log_rest_200_999_hi_noise_dim32_4444.txt',
    'Rest (400 16 4442 hi noise)': 'log_rest_400_999_hi_noise_dim16_4442.txt',
}

# Parse all data first
data = {label: parse_log(path) for label, path in logs.items()}

plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: Training Loss
plt.figure(figsize=(13, 6))
for label, df in data.items():
    plt.plot(df['Epoch'], df['Loss'], label=label)

plt.title('Training Loss (Log Scale)', fontsize=14)
plt.yscale('log')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('loss_comparison.jpg') # Saves as a separate file
plt.close() # Closes the figure to free memory

# Figure 2: PSNR Comparison
plt.figure(figsize=(13, 6))
for label, df in data.items():
    plt.plot(df['Epoch'], df['PSNR'], label=label)

plt.title('PSNR Comparison', fontsize=14)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('psnr_comparison.jpg') # Saves as a separate file
plt.close()