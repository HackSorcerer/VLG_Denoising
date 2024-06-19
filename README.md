This repository features a TensorFlow-based image enhancement model designed to significantly improve the quality of low-resolution images through a recursive approach. The model employs a custom Peak Signal-to-Noise Ratio (PSNR) metric to evaluate the quality of the enhanced images. To get started, clone the repository using the command git clone https://github.com/yourusername/image-enhancement.git and navigate into the directory. Ensure you have all necessary dependencies installed by running pip install -r requirements.txt. You will need to download or provide your trained model file named mymodel2.h5 and place it in the Models directory for the enhancement process to function correctly.

To use the enhancement tool, prepare your low-resolution images by placing them into the test/low directory. Once your images are in place, execute the enhancement script with the command python enhance_images.py. This script processes the images in the test/low directory and saves the enhanced versions in the test/predicted directory. The enhancement process involves several key functions and steps. The compute_psnr function calculates the PSNR value between the original and processed images, providing a quantitative measure of the enhancement quality.

The script loads the pre-trained enhancement model using TensorFlow’s load_model function, including the custom PSNR metric. The core of the enhancement process is a recursive function, enhance_image_recursively, which iteratively improves the image quality over multiple steps. This function adapts based on whether it is processing the initial image or subsequent iterations, ensuring optimal enhancement at each step. The script reads images from the specified folder, processes them, and saves the enhanced images back to the specified output directory.

Additionally, the script ensures that the output directory exists, creating it if necessary, and saves the enhanced image with appropriate scaling to maintain image integrity. This repository encourages contributions, welcoming any suggestions for improvements or reports of issues via the opening of issues or submission of pull requests. The project is distributed under the MIT License, allowing for broad usage and modification. For detailed instructions and more information, refer to the README file and the code comments within the script.





