#%% 1. Import libraries

from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
import open3d.camera as o3d_camera
from transformers import DPTImageProcessor, DPTForDepthEstimation
import os

# Change to the project directory automatically
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)

#%% 2. Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "images/sample2.JPG"

#%% 3. Load and resize image

image = Image.open(image_path).convert("RGB")
max_height = 480
aspect_ratio = image.width / image.height
new_height = min(image.height, max_height)
new_width = int(new_height * aspect_ratio)
image = image.resize((new_width, new_height))

#%% 4. Load model

processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)

#%% 5. Predict depth

inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    predicted_depth = model(**inputs).predicted_depth

depth_map = predicted_depth.squeeze().cpu().numpy() * 1000.0  # to mm

# Resize image to match depth map (384 x 384)
image = image.resize((depth_map.shape[1], depth_map.shape[0]))
image_np = np.asarray(image, dtype=np.uint8)

# Debug print
print("✅ Depth map ready | Shape:", depth_map.shape, "| Any NaNs:", np.isnan(depth_map).any())

#%% 6. Show depth map

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(depth_map, cmap='plasma')
plt.title("Depth Map")
plt.axis('off')
plt.colorbar()
plt.tight_layout()
plt.show()

#%% 7 preparing depth image for open3d

width, height = image.size
depth_image = (depth_map*255/np.max(depth_map)).astype('uint8')
image = np.array(image)

#create a rgbd image 
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

#%% 8 creating a camera

camera_intrinsic = o3d_camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

#%% 9 Creating a o3d point cloud

pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
o3d.visualization.draw_geometries([pcd_raw])

#%% 10 Post processing the 3D Cloud

cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors = 20, std_ratio=6.0)
pcd = pcd_raw.select_by_index(ind)

#estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

o3d.visualization.draw_geometries([pcd])

#%% 11 Surface reconstruction

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 10, n_threads = 1)[0]

rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center=(0, 0, 0))

# visualise the mesh

o3d.visualization.draw_geometries([mesh], mesh_show_back_face = True)

#%% 12 Exports

import os
from matplotlib import cm

# Create output folder if needed
os.makedirs("outputs", exist_ok=True)

#Saving 3D Mesh
o3d.io.write_triangle_mesh('outputs/mesh_output.obj', mesh, write_vertex_colors=True)

# Save RGB image
Image.fromarray(image).save("outputs/example_rgb.jpg")

# Normalize depth for visualization
depth_norm = depth_map - depth_map.min()
depth_norm /= depth_norm.max()
depth_colored = cm.plasma(depth_norm)[:, :, :3]  # Drop alpha channel

# Convert to uint8 image
depth_colored_img = (depth_colored * 255).astype(np.uint8)
Image.fromarray(depth_colored_img).save("outputs/example_depth.png")
Image.fromarray(depth_colored_img).save("outputs/sample_depthmap.png")


