# Mock OCR Model - Performance Evaluation on CPU vs GPU

## Overview
This repository contains the implementation of a Mock OCR Model designed for Optical Character Recognition (OCR) tasks. The project includes performance evaluation of the model on different devices (CPU vs GPU) using PyTorch. The evaluation focuses on Frames Per Second (FPS) and Latency measurements to compare the computational efficiency of the model.

In this repository, you will also find memory usage tracking and basic OCR output comparison. This project can be expanded for more sophisticated OCR models or used for evaluating model performance across various platforms.

## Features
- **Mock OCR Model**: A convolutional neural network (CNN) model designed for OCR-like tasks.
- **Device Performance Evaluation**: Compare the FPS and latency of the model on CPU and GPU (or MPS if using Mac devices with Apple Silicon).
- **Memory Usage Tracking**: Monitor memory usage on the CPU during model inference.
- **OCR Output Comparison**: Sample OCR outputs for testing and processing comparison.
- **Matplotlib Plots**: Visualize FPS and Latency for different devices.

## Prerequisites
To run this repository, you'll need the following:

- Python 3.7+
- PyTorch (for model implementation)
- Matplotlib (for plotting)
- psutil (for memory tracking)
- Tesseract (for OCR-related tasks)

Dependencies are listed in `requirements.txt`. You can install the necessary libraries by running:

```bash
pip install -r requirements.txt```

Additional Setup
Tesseract OCR Installation:
To enable OCR functionality, you must have Tesseract installed on your system. You can install it using the following methods:

Linux/Mac:

bash
Copy code
brew install tesseract
Windows:

Download and install Tesseract from Tesseract’s official page.
After installation, add the Tesseract binary path to your system’s PATH environment variable.
Optional Setup for MacOS (Apple Silicon): If you are using a Mac with an Apple Silicon chip, ensure that you have set up the MPS (Metal Performance Shaders) backend for PyTorch, which can significantly improve performance over the standard CPU backend. To enable this:

Install the nightly version of PyTorch:

bash
Copy code
pip install torch==1.13.0+cpu
Set the environment variable for PyTorch to use MPS as the backend:

bash
Copy code
export PYTORCH_ENABLE_MPS_FALLBACK=1
CUDA for GPU Setup (Optional): For utilizing GPU on Linux or Windows with CUDA-enabled hardware:

First, install the appropriate CUDA version as per your system’s GPU model and PyTorch’s supported CUDA version (refer to PyTorch’s official installation page).

Install the CUDA-enabled version of PyTorch:

bash
Copy code
pip install torch==<version>+cu<cuda_version>
Replace <version> with the desired PyTorch version and <cuda_version> with the specific CUDA version (e.g., cu113 for CUDA 11.3).

Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/mock-ocr-model.git
cd mock-ocr-model
Install Dependencies:

bash
Copy code
pip install torch torchvision torchaudio
pip install psutil matplotlib pytesseract
Usage
1. Model Definition
The MockOCRModel class defines a simple CNN-based model with two convolutional layers and a fully connected layer. The architecture is intended to simulate an OCR model’s basic structure.

2. Evaluating Performance
You can evaluate the model's performance on CPU and GPU by running the following code:

python
Copy code
# Evaluate performance on CPU and GPU
fps_gpu, latency_gpu = evaluate_performance(gpu_model, processed_image, device_gpu)
fps_cpu, latency_cpu = evaluate_performance(cpu_model, processed_image, device_cpu)

# Print the results
print(f"FPS (GPU): {fps_gpu:.2f}, Latency (GPU): {latency_gpu:.4f} seconds per image")
print(f"FPS (CPU): {fps_cpu:.2f}, Latency (CPU): {latency_cpu:.4f} seconds per image")
3. Memory Usage
The memory usage of the CPU model is tracked using psutil. You can see the memory consumption in the terminal as the model is evaluated:

python
Copy code
memory_cpu = process.memory_info().rss / (1024 ** 2)  # in MB
print(f"Memory usage (CPU): {memory_cpu:.2f} MB")
4. OCR Output Comparison
The repository includes some sample OCR outputs to showcase the differences between the original and processed OCR text.

5. Plotting Performance
The FPS and Latency results are plotted using Matplotlib:

python
Copy code
# Plotting FPS and Latency for GPU vs CPU
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# FPS Plot
ax[0].bar(devices, fps_values, color=['blue', 'green'])
ax[0].set_title('FPS Comparison')
ax[0].set_xlabel('Device')
ax[0].set_ylabel('Frames Per Second')

# Latency Plot
ax[1].bar(devices, latency_values, color=['blue', 'green'])
ax[1].set_title('Latency Comparison')
ax[1].set_xlabel('Device')
ax[1].set_ylabel('Seconds per Image')

plt.tight_layout()
plt.show()
Example Output
When you run the model, you'll receive output similar to:

bash
Copy code
FPS (GPU): 120.45, Latency (GPU): 0.0084 seconds per image
FPS (CPU): 55.23, Latency (CPU): 0.0181 seconds per image
Memory usage (CPU): 58.12 MB
Original OCR Output:

Copy code
GABRIEL Mepmall...
Processed OCR Output:

Copy code
GABRIEL MermaLll...
Contributing
Feel free to contribute to this project by opening issues or submitting pull requests. All contributions are welcome!

Fork the repository
Create your feature branch (git checkout -b feature-branch)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature-branch)
Create a new Pull Request
License
This project is licensed under the MIT License - see the LICENSE file for details.

vbnet
Copy code

This version of the README includes the additional setup instructions, such as how to set up Tesseract, CUDA for GPU, and MPS for Mac devices with Apple Silicon. Let me know if you need further modifications!
