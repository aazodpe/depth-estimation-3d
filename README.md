# 🧠 3D Depth Estimation from a Single Image using DPT + Open3D

This project demonstrates how to estimate a depth map from a **single RGB image** using Intel's `dpt-hybrid-midas` model (a transformer-based monocular depth estimator), and then reconstruct a 3D **point cloud** using Open3D.

<img src="outputs/sample_depthmap.png" width="500">

---

## 📦 Features

- 🔍 **Transformer-based depth estimation** (DPT model from Intel)
- 🌍 **Point cloud reconstruction** from RGB + depth with Open3D
- 📊 Visual output: depth map + 3D visualization
- ✅ Works in a Python virtual environment

---

## 🚀 Getting Started

### 1. 📁 Clone the Repository

```bash
git clone https://github.com/your-username/depth-estimation-3d.git
cd depth-estimation-3d
```

### 2. 🧪 Create a Virtual Environment

We recommend using `venv` or `conda`:

<details>
<summary>Using <code>venv</code> (built-in)</summary>

```bash
python3 -m venv reconst
source reconst/bin/activate  # On Windows: reconst\Scripts\activate
```

</details>

<details>
<summary>Using <code>conda</code></summary>

```bash
conda create -n reconst python=3.11
conda activate reconst
```

</details>

---

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. 🏃 Run the Program

Place your input image in the project folder and update the `image_path` in `depth_estimation.py`.

```bash
python depth_estimation.py
```

---

## 🧠 How It Works

1. Loads a single image (`.jpg`, `.png`) using Pillow
2. Predicts **relative depth** using the DPT MiDaS model from Hugging Face:
   > [`Intel/dpt-hybrid-midas`](https://huggingface.co/Intel/dpt-hybrid-midas)
3. Converts depth + RGB image to an Open3D `RGBDImage`
4. Uses **pinhole camera intrinsics** to project a 3D point cloud
5. Visualizes the output with Open3D

⚠️ Note: The depth is **relative**, not metric — it's scaled for visualization.

---

## 📁 File Structure

```
depth-estimation-3d/
├── depth_estimation.py     # Main script
├── requirements.txt        # Python dependencies
├── test_image.jpg          # (Optional) Sample input image
├── outputs/                # (Optional) Output point clouds or visuals
└── README.md
```

---

## 🧩 Dependencies

- `torch`
- `transformers`
- `open3d`
- `matplotlib`
- `numpy`
- `Pillow`

Install with:

```bash
pip install -r requirements.txt
```

---

## 🖼️ Example Output

| Original Image | Depth Map | Point Cloud |
|----------------|-----------|-------------|
| ![rgb](outputs/example_rgb.jpg) | ![depth](outputs/example_depth.png) | *(opens Open3D viewer)* |

---

## 📝 License

This project is open-source and available under the **MIT License**.

---

## 🙏 Acknowledgements

- Intel + Hugging Face for [`dpt-hybrid-midas`](https://huggingface.co/Intel/dpt-hybrid-midas)
- Open3D for point cloud and visualization tools

---

Happy reconstructing! 🧱🔍
