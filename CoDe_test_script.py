import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from CoDe_Container import CoDe_Container

def test_code_module(image_path):
    # 1. Load and Preprocess Image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 256, 256]

    # 2. Define a simple Base Model (just an Identity for testing)
    class IdentityModel(torch.nn.Module):
        def forward(self, x): return x

    # 3. Initialize the CoDe Container
    # We use n_components=2 to get Low-Freq and High-Freq
    model = CoDe_Container(base_model=IdentityModel(), n_components=2)
    model.eval()

    # 4. Extract Sub-images via the Content Decoupling Module
    with torch.no_grad():
        # Get the sub-images directly from the CDM for visualization
        sub_images = model.cdm(input_tensor)
        # Get the final combined output from the full container
        final_output = model(input_tensor)

    # 5. Helper to convert tensor back to displayable image
    def to_img(tensor):
        # Clamp and squeeze for visualization
        return tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    # 6. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(to_img(input_tensor))
    axes[0].set_title("Original Image")
    
    axes[1].imshow(to_img(sub_images[0]))
    axes[1].set_title("Sub-image 1 (Low Freq)")
    
    # High freq often looks dark/gray; we scale it for better visibility
    axes[2].imshow(to_img(sub_images[1] * 3)) 
    axes[2].set_title("Sub-image 2 (High Freq - Boosted)")
    
    axes[3].imshow(to_img(final_output))
    axes[3].set_title("Reconstructed Output")

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

# Run the test (provide a valid path to an image)
# test_code_module('path_to_your_image.jpg')

if __name__ == "__main__":
    test_code_module('./0_tonemap_noisy.png')  # Replace with your test image path