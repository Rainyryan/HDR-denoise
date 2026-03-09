import torch
import torch_pruning as tp

from HDR_model_switch import HDR_model
from HDR_dataset import *

# 1. Initialize your Restormer model/block
# Using the parameters from your test script

model = HDR_model()

save_folder = "models_rest_300_999_hi_noise_dim16_4442/"

save_path = save_folder + "preexpand_hdr_best.pth"

checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().cpu()
# model.to(torch.device("cuda:0"))



# 1. Identify layers to ignore
customized_pruners = {}
ignored_layers = []
for name, m in model.named_modules():
    # Ignore Attention (MDTA) - The primary cause of the hang
    # if "attn" in name or "MDTA" in str(type(m)):
    #     ignored_layers.append(m)
    # Ignore Deformable Convs - The secondary cause
    if "Deformable" in str(type(m)):
        ignored_layers.append(m)
    # Ignore Norms and Output
    if isinstance(m, torch.nn.LayerNorm) or m == model.reduce_chan_final:
        ignored_layers.append(m)
    if "GDFN" in str(type(m)):
        # customized_pruners[type(m.project_in)] = GDFNPruner()
        ignored_layers.append(m)



# 3. Setup Pruner with NO external DG (let it build its own mini-graphs)
print("Setting up pruner (Lightweight Mode)...")
patch_sz = 512
example_inputs = torch.randn(1, 3, patch_sz, patch_sz) # Smallest possible for logic check


imp = tp.importance.GroupMagnitudeImportance(p=2)
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    pruning_ratio=0.5,
    ignored_layers=ignored_layers,
    customized_pruners=customized_pruners,
    # Crucial: Only look for standard Convs to start pruning
    # root_module_types=[torch.nn.Conv2d], 
    round_to=None,
)


print("Pruner setup done")
# 3. Prune the model
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
tp.utils.print_tool.before_pruning(model) # or print(model)
pruner.step()
# group.prune() # Prune the entire group at once, instead of one layer at a time. This ensures the GDFN branches are pruned together.
tp.utils.print_tool.after_pruning(model) # or print(model), this util will show the difference before and after pruning
macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")

model.cuda()
example_inputs = example_inputs.to("cuda:0")
# 5. Verify forward pass
with torch.no_grad():
    output = model(example_inputs)
    print(f"Pruning successful. Output shape: {output.shape}")