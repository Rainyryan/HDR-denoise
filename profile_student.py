import os
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch.profiler import profile, record_function, ProfilerActivity

from HDR_model_switch import HDR_model
from HDR_model_hybrid_student import Hybrid_Student_HDR

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on: {device}")

    # 1. Initialize the Student Model
    dim = 8
    num_blocks = [2, 2, 2, 1]
    model = Hybrid_Student_HDR(dim=dim, num_blocks=num_blocks, num_refinement_blocks=2, heads=[1,2,4,8]).to(device)

    # Load weights
    ckpt_path = "./distill_models_20260417_134836/preexpand_hdr_best.pth"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['student_state_dict'])
        print(f"Successfully loaded checkpoint: {ckpt_path}")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Using random weights.")
    
    model.eval()

    # 2. Define Dummy Input (using your patch_sz=512)
    # Note: Assuming 3 channels based on your PixelUnshuffle(2) -> 12 channels logic
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    # ==========================================
    # Part A: FLOPs and Parameter Analysis
    # ==========================================
    print("\n" + "="*50)
    print("Running FLOP Analysis (fvcore)...")
    print("="*50)
    
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_input)
        # flop_count_table gives a beautiful hierarchical breakdown of your network
        print(flop_count_table(flops, max_depth=3)) 

    # ==========================================
    # Part B: Execution Latency Profiling
    # ==========================================
    print("\n" + "="*50)
    print("Running Latency Trace (torch.profiler)...")
    print("="*50)

    # Warmup to avoid cold-start bias in CUDA
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()

    # Profile GPU and CPU execution
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("student_model_inference"):
            with torch.no_grad():
                _ = model(dummy_input)

    # Print a summary of the most time-consuming operations
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Save a trace file for visual inspection
    trace_file = "student_profile_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace saved to {trace_file}")
    print("Open Google Chrome and navigate to: chrome://tracing to view the timeline.")

if __name__ == "__main__":
    main()