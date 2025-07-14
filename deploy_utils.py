from deploy_imports import *
import spatialstudio as splv

def vv_to_gridbatch(frame):
    """Convert VV frame to GridBatch"""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        frame.save_to_nvdb(tmp_file.name)
        grid, feats, _ = fvdb.load(tmp_file.name, device="cuda:0")
    return grid, feats

def vv_file_to_gridbatch(vv_file: str):
    """Convert VV file to GridBatch"""
    frame = splv.frame_load(vv_file)
    return vv_to_gridbatch(frame)

def vv_files_to_gridbatch(vv_files):
    """Convert directory of VV files to GridBatch"""
    
    # If vv_files is a directory, get all .vv files
    if isinstance(vv_files, str) and os.path.isdir(vv_files):
        vv_files = [os.path.join(vv_files, f) for f in os.listdir(vv_files) if f.endswith('.vv')]
    
    grids, feats = [], []
    for vv_file in vv_files:
        grid, feat = vv_file_to_gridbatch(vv_file)
        grids.append(grid)
        feats.append(feat)
    
    if not grids:
        raise ValueError("No VV files found")
    
    grid = fvdb.jcat(grids)
    feat = fvdb.jcat(feats)
    return grid, feat

unsqueeze3x = lambda x: x[..., None, None, None]
unsqueeze4x = lambda x: x[..., None, None, None, None]
unsqueeze5x = lambda x: x[..., None, None, None, None, None]
device = "cuda:0"
# unsqueeze = unsqueeze5x if (args.dimensions == 3 and args.num_frames > 1) else unsqueeze4x if args.dimensions == 3 else unsqueeze3x
unsqueeze = unsqueeze4x  

class GaussianDiffusion:
    """Gaussian diffusion process with 1) Cosine schedule for beta values (https://arxiv.org/abs/2102.09672)
    2) L_simple training objective from https://arxiv.org/abs/2006.11239.
    """

    def __init__(self, timesteps=1000, device="cuda:0"):
        self.timesteps = timesteps
        self.device = device
        self.alpha_bar_scheduler = (
            lambda t: math.cos((t / self.timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        self.scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, self.timesteps, self.device
        )

        self.clamp_x0 = lambda x: x.clamp(-1, 1)
        self.get_x0_from_xt_eps = lambda xt, eps, t, scalars: (
            self.clamp_x0(
                1
                / unsqueeze(scalars.alpha_bar[t].sqrt())
                * (xt - unsqueeze((1 - scalars.alpha_bar[t]).sqrt()) * eps)
            )
        )
        self.get_pred_mean_from_x0_xt = (
            lambda xt, x0, t, scalars: unsqueeze(
                (scalars.alpha_bar[t].sqrt() * scalars.beta[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * x0
            + unsqueeze(
                (scalars.alpha[t] - scalars.alpha_bar[t])
                / ((1 - scalars.alpha_bar[t]) * scalars.alpha[t].sqrt())
            )
            * xt
        )

    def get_all_scalars(self, alpha_bar_scheduler, timesteps, device, betas=None):
        """
        Using alpha_bar_scheduler, get values of all scalars, such as beta, beta_hat, alpha, alpha_hat, etc.
        """
        all_scalars = {}
        if betas is None:
            all_scalars["beta"] = torch.from_numpy(
                np.array(
                    [
                        min(
                            1 - alpha_bar_scheduler(t + 1) / alpha_bar_scheduler(t),
                            0.999,
                        )
                        for t in range(timesteps)
                    ]
                )
            ).to(
                device
            )  # hardcoding beta_max to 0.999
        else:
            all_scalars["beta"] = betas
        all_scalars["beta_log"] = torch.log(all_scalars["beta"])
        all_scalars["alpha"] = 1 - all_scalars["beta"]
        all_scalars["alpha_bar"] = torch.cumprod(all_scalars["alpha"], dim=0)
        all_scalars["beta_tilde"] = (
            all_scalars["beta"][1:]
            * (1 - all_scalars["alpha_bar"][:-1])
            / (1 - all_scalars["alpha_bar"][1:])
        )
        all_scalars["beta_tilde"] = torch.cat(
            [all_scalars["beta_tilde"][0:1], all_scalars["beta_tilde"]]
        )
        all_scalars["beta_tilde_log"] = torch.log(all_scalars["beta_tilde"])
        return EasyDict(dict([(k, v.float()) for (k, v) in all_scalars.items()]))

    def sample_from_forward_process(self, x0, t):
        """Single step of the forward process, where we add noise in the image.
        Note that we will use this paritcular realization of noise vector (eps) in training.
        """
        if x0.ndim == 4:
            x0 = x0.squeeze(0)
        eps = torch.randn_like(x0)
        xt = (
            unsqueeze(self.scalars.alpha_bar[t].sqrt()) * x0
            + unsqueeze((1 - self.scalars.alpha_bar[t]).sqrt()) * eps
        )
        assert xt.shape == x0.shape
        if xt.ndim == 4:
            xt = xt.squeeze(0)
        return xt.float(), eps

    def sample_from_reverse_process(
        self, model, xT, timesteps=None, model_kwargs={}, ddim=False, use_cfg=False
    ):
        """Sampling images by iterating over all timesteps.

        model: diffusion model
        xT: Starting noise vector.
        timesteps: Number of sampling steps (can be smaller the default,
            i.e., timesteps in the diffusion process).
        model_kwargs: Additional kwargs for model (using it for conditioning)
        ddim: Use ddim sampling (https://arxiv.org/abs/2010.02502).
        cfg_scale: Scale for classifier-free guidance (1.0 means no guidance)

        Return: An image tensor with identical shape as XT.
        """
        model.eval()
        final = xT
        if final.ndim == 4:
            final = final.unsqueeze(0)
        
        if use_cfg:
            cfg_scale = args.guidance_scale
            uncond_kwargs = model_kwargs.copy()
            uncond_kwargs.update({"y": uncond_kwargs["y_uncond"]})
            del uncond_kwargs["y_uncond"]
            
            def cfg_model(x, t, **kwargs):
                eps_cond = model(x, t, **kwargs)
                
                uncond_k = uncond_kwargs.copy()

                eps_uncond = model(x, t, **uncond_k)
                
                return eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            
            model_fn = cfg_model
            del model_kwargs["y_uncond"]
        else:
            model_fn = model
            if "y_uncond" in model_kwargs:
                del model_kwargs["y_uncond"]

        # sub-sampling timesteps for faster sampling
        timesteps = timesteps or self.timesteps
        new_timesteps = np.linspace(
            0, self.timesteps - 1, num=timesteps, endpoint=True, dtype=int
        )
        alpha_bar = self.scalars["alpha_bar"][new_timesteps]
        new_betas = 1 - (
            alpha_bar / torch.nn.functional.pad(alpha_bar, [1, 0], value=1.0)[:-1]
        )
        scalars = self.get_all_scalars(
            self.alpha_bar_scheduler, timesteps, self.device, new_betas
        )

        for i, t in zip(np.arange(timesteps)[::-1], new_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.tensor([t] * len(final), device=final.device)
                current_sub_t = torch.tensor([i] * len(final), device=final.device)
                pred_epsilon = model_fn(final, current_t, **model_kwargs)
                # using xt+x0 to derive mu_t, instead of using xt+eps (former is more stable)
                pred_x0 = self.get_x0_from_xt_eps(
                    final, pred_epsilon, current_sub_t, scalars
                )
                pred_mean = self.get_pred_mean_from_x0_xt(
                    final, pred_x0, current_sub_t, scalars
                )
                if i == 0:
                    final = pred_mean
                else:
                    if ddim:
                        final = (
                            unsqueeze(scalars["alpha_bar"][current_sub_t - 1]).sqrt()
                            * pred_x0
                            + (
                                1 - unsqueeze(scalars["alpha_bar"][current_sub_t - 1])
                            ).sqrt()
                            * pred_epsilon
                        )
                    else:
                        final = pred_mean + unsqueeze(
                            scalars.beta_tilde[current_sub_t].sqrt()
                        ) * torch.randn_like(final)
                final = final.detach()

        if final.ndim == 4:
            final = final.squeeze(0)
        return final

def fvdb_file_to_vv(path, output_path=None):

    grid, feats, _ = fvdb.load(path, device="cpu")

    new_frame = splv.SPLVframe(128, 128, 128)
    voxels = []

    for i in range(grid.ijk.jdata.shape[0]):
        xyz = grid.ijk.jdata[i]
        rgb = feats.jdata[i]
        r = int(rgb[0] * 255)
        g = int(rgb[1] * 255)
        b = int(rgb[2] * 255)
        new_frame.set_voxel(xyz[0], xyz[1], xyz[2], (r, g, b))

    if output_path is None:
        return new_frame
    else:
        new_frame.save(output_path)

def fvdb_to_vv(grid, feats, output_path=None):
    new_frame = splv.SPLVframe(128, 128, 128)
    for i in range(grid.ijk.jdata.shape[0]):
        xyz = grid.ijk.jdata[i]
        rgb = feats.jdata[i]
        r = int(rgb[0] * 255)
        g = int(rgb[1] * 255)
        b = int(rgb[2] * 255)
        new_frame.set_voxel(xyz[0], xyz[1], xyz[2], (r, g, b))
    if output_path is None:
        return new_frame
    else:
        new_frame.save(output_path)


def visualize_vdb_with_polyscope(vdb_tensor: fvnn.VDBTensor, 
                                out_path: str = "debug/example_vdb.png",
                                batch_element: int = 0):
    """
    Visualize a VDB tensor using polyscope in a separate process to avoid state issues
    """
    import subprocess
    import tempfile
    
    # Handle batch selection
    if vdb_tensor.grid_count > 1:
        single_vdb = fvnn.VDBTensor(vdb_tensor.grid[batch_element], vdb_tensor.data[batch_element])
    else:
        single_vdb = vdb_tensor
    
    # Save VDB tensor to temporary file using fvdb API
    with tempfile.NamedTemporaryFile(suffix='.nvdb', delete=False) as f:
        temp_path = f.name
    
    # Save using fvdb.save
    fvdb.save(temp_path, single_vdb.grid, single_vdb.data)
    
    # Create the visualization script path
    save_path = f"{out_path.replace(' ', '_')}"
    name = save_path.split(".")[0]
    save_dir = os.path.dirname(save_path)
    if save_dir:  # Only create directory if there is one
        os.makedirs(save_dir, exist_ok=True)
    
    # Run visualization in subprocess
    try:
        result = subprocess.run([
            'python', '-c', f'''
import fvdb
from fvdb import nn as fvnn
import polyscope as ps
import numpy as np
import torch
import os
from pathlib import Path

# Load VDB data using fvdb API
grid, data, _ = fvdb.load("{temp_path}", device=torch.device("cpu"))
vdb_tensor = fvnn.VDBTensor(grid, data)

# Initialize polyscope
ps.init()
ps.set_ground_plane_mode("none")

# Extract coordinates and colors
xyz_coords = vdb_tensor.grid.grid_to_world(vdb_tensor.grid.ijk.jdata.float()).jdata.cpu().numpy()
rgb_colors = vdb_tensor.data.jdata[:, :3].detach().cpu().numpy()
rgb_colors = np.clip(rgb_colors, 0.0, 1.0)

# Create cube geometry
cube_verts = []
cube_faces = []
cube_colors = []

unit_cube_verts = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]
])

unit_cube_faces = np.array([
    [0,1,2], [0,2,3], [4,7,6], [4,6,5], [0,4,5], [0,5,1],
    [2,6,7], [2,7,3], [0,3,7], [0,7,4], [1,5,6], [1,6,2]
])

for i, (pos, color) in enumerate(zip(xyz_coords, rgb_colors)):
    verts = unit_cube_verts + pos
    faces = unit_cube_faces + len(cube_verts)
    cube_verts.extend(verts)
    cube_faces.extend(faces)
    cube_colors.extend([color] * 8)

cube_verts = np.array(cube_verts)
cube_faces = np.array(cube_faces)
cube_colors = np.array(cube_colors)

# Register mesh
mesh = ps.register_surface_mesh("{name}", cube_verts, cube_faces)
mesh.add_color_quantity("rgb", cube_colors, defined_on="vertices", enabled=True)

# Set camera to match O3D view (rotated ~135 degrees horizontally, 15 degrees down)
if len(cube_verts) > 0:
    bbox_center = np.mean(cube_verts, axis=0)
    bbox_size = np.max(cube_verts, axis=0) - np.min(cube_verts, axis=0)
    distance = np.linalg.norm(bbox_size) * 1.5
    
    # 135 degree rotation around Y axis, 15 degrees down from horizontal
    horizontal_angle = np.radians(135)
    vertical_angle = np.radians(15)  # negative for looking down
    
    # Calculate camera position
    camera_pos = bbox_center + distance * np.array([
        np.sin(horizontal_angle) * np.cos(vertical_angle),
        np.sin(vertical_angle),
        np.cos(horizontal_angle) * np.cos(vertical_angle)
    ])
    
    ps.look_at(camera_pos, bbox_center)

ps.screenshot("{save_path}")
print("Saved polyscope visualization to {save_path}")
'''
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Subprocess failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Visualization subprocess timed out")
        return None
    finally:
        # Clean up temp file
        os.unlink(temp_path)
    # print(f"Saved polyscope visualization to {save_path}")
    
    return save_path
