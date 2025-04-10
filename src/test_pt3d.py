import torch
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
verts = torch.rand((1, 3, 3), device=device)
faces = torch.tensor([[[0, 1, 2]]], device=device)

meshes = Meshes(verts=verts, faces=faces)

settings = RasterizationSettings(
    image_size=64,
    blur_radius=0.0,
    faces_per_pixel=1,
)

fragments = rasterize_meshes(
    meshes=meshes,
    image_size=settings.image_size,
    blur_radius=settings.blur_radius,
    faces_per_pixel=settings.faces_per_pixel,
)

print("✅ PyTorch3D is using GPU!")
