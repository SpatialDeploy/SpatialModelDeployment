from deploy_imports import *
from deploy_models import *
from deploy_utils import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Config:
    def __init__(self):

        # Diffusion settings
        self.diffusion_steps = 1000
        self.sampling_steps = 10
        self.ddim = True
        self.use_cfg = False
        self.guidance_scale = 3.0
        
        # Inference settings
        self.batch_size = 1
        self.output_dir = "outputs"
        self.input_dir = "inputs"
        self.display_images = False
        
        # Model settings
        self.device = "cuda:0"
        self.latent_image_size = 16
        self.latent_vector_channels = 32
        self.base_channels = 128
        self.image_size = 128
        self.model_name = "UNet3DInpaintingL"
        self.model_path = "models/diffusion_UNet3DInpaintingL_humans_ema_epoch_9.pth"
        self.vae_path = "models/vae_32x16v_128b_sparse_humans_ema.pth"
        self.vae_norm_path = "models/vae_32x16v_128b_sparse_humans_ema_normalization.pth"
        self.seed = 42

config = Config()

def sparse_encode(vae, vdb_tensor, norm_params=None):
    """Encode a sparse VDB tensor using the sparse VAE"""
    with torch.no_grad():
        vae.eval()
        mu, log_var = vae.encode(vdb_tensor)
        latents = vae.reparametrize(mu, log_var)
    if norm_params is not None:
        num_channels = latents.shape[1]
        for c in range(num_channels):
            shift_c = norm_params[f'channel_{c}']['shift']
            scale_c = norm_params[f'channel_{c}']['scale']
            latents[:, c] = (latents[:, c] + shift_c) * scale_c
    return latents

def sparse_decode(vae, latents, norm_params=None):
    """Decode sparse latents back to VDB tensor"""
    if norm_params is not None:
        num_channels = latents.shape[1]
        for c in range(num_channels):
            shift_c = norm_params[f'channel_{c}']['shift']
            scale_c = norm_params[f'channel_{c}']['scale']
            latents[:, c] = (latents[:, c]) / scale_c - shift_c
    with torch.no_grad():
        vae.eval()
        output = vae.decode(latents)
    return output["color"]

def sample_N_images(
    N,
    model,
    diffusion,
    vae,
    cond,
    xT=None,
    sampling_steps=100,
    batch_size=1,
    config=None,
    guidance_scale=3.0,
    norm_params=None,
):
    """Sample images from the diffusion model"""
    samples, num_samples = [], 0

    with torch.no_grad():
        with tqdm(total=N) as pbar:
            while num_samples < N:
                y = sparse_encode(vae, cond, norm_params=norm_params)

                if xT is None:
                    xT = 2*((
                        torch.rand(batch_size, config.latent_vector_channels, config.latent_image_size, config.latent_image_size, config.latent_image_size)
                        .float()
                        .to(config.device)
                    ) - 0.5)
                if config.use_cfg:
                    y_uncond = torch.zeros_like(y)
                else:
                    y_uncond = None

                denoised_latents = diffusion.sample_from_reverse_process(
                    model, xT, sampling_steps, 
                    {"y": y, "y_uncond": y_uncond}, 
                    config.ddim,
                    use_cfg=config.use_cfg
                )
                
                outputs = sparse_decode(vae, denoised_latents, norm_params=norm_params)

                samples.append(outputs)

                num_samples += len(xT)
                pbar.update(num_samples)
            
        return samples

def load_models(config):
    """Load all required models"""
    print("Loading models...")

    # Load diffusion model
    model = globals()[config.model_name](
        image_size=config.latent_image_size,
        in_channels=config.latent_vector_channels,
        out_channels=config.latent_vector_channels,
    ).to(config.device)

    print("Model parameters: ", sum(p.numel() for p in model.parameters()))
    # Load model weights
    if os.path.exists(config.model_path):
        print(f"Loading model from {config.model_path}")
        model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    else:
        print(f"Model path {config.model_path} not found")
        exit()
    
    model.eval()
    
    # Load VAE
    vae = SparseCoarseVAE(
        color_channels=3,
        base_ch=config.base_channels,
        levels=3,
        latent_ch=config.latent_vector_channels,
    ).to(config.device)
    
    # Compile VAE first, then load weights
    vae = torch.compile(vae)
    vae.load_state_dict(torch.load(config.vae_path), strict=True)
    vae.eval()
    
    # Load normalization parameters
    stats = torch.load(config.vae_norm_path)
    norm_params = stats['normalization']['per_channel']
    
    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False
    
    # Load diffusion process
    diffusion = GaussianDiffusion(config.diffusion_steps, config.device)
    

    return model, vae, diffusion, norm_params


def run_inference_custom(model, vae, diffusion, config, norm_params):
    """Run inference with custom inputs"""
    print("Running inference with custom inputs...")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # check if input_dir is empty
    if not os.listdir(config.input_dir):
        print("Input directory is empty. Please provide a directory with .vv files.")
        exit()
    else:
        print(f"Running inference on {len(os.listdir(config.input_dir))} .vv files.")

    # Get list of VV files and create names list
    vv_files = [f for f in os.listdir(config.input_dir) if f.endswith('.vv')]
    names = [os.path.splitext(f)[0] for f in vv_files]  # Remove .vv extension
    
    grid, feats = vv_files_to_gridbatch(config.input_dir)
    cond = fvnn.VDBTensor(grid, feats)
    num_samples = len(vv_files)
    
    sample_vdbs = sample_N_images(num_samples, model, diffusion, vae, cond, sampling_steps=config.sampling_steps, batch_size=config.batch_size, config=config, guidance_scale=config.guidance_scale, norm_params=norm_params)
    
    # Save results
    for i in range(len(sample_vdbs)):
        sample_vdb = sample_vdbs[i]
        for j in range(sample_vdb.grid_count):
            output = sample_vdb[j]
            num = i * config.batch_size + j
            name = names[num] if num < len(names) else f"sample_{num}"
            output_path = os.path.join(config.output_dir, f"{name}.png")
            vv_path = os.path.join(config.output_dir, f"{name}.vv")
            
            # Visualize using the same function as training
            if config.display_images:
                visualize_vdb_with_polyscope(output, out_path=output_path, batch_element=0)
            
            fvdb_to_vv(output.grid, output.data, vv_path)

def main():
    print("Starting inpainting inference...")
    start_time = time.time()
    # Setup
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Enable efficient attention mechanisms
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    
    # Load models
    model, vae, diffusion, norm_params = load_models(config)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Run inference
    run_inference_custom(model, vae, diffusion, config, norm_params)
    
    print("Inference completed in {:.2f} seconds!".format(time.time() - start_time))

if __name__ == "__main__":
    main() 