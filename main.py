import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
import imageio
import numpy as np
import os

def compute_psnr(original, processed):
    return tf.image.psnr(original, processed, max_val=1.0)

enhancement_model = load_model('Models/model.h5', custom_objects={'psnr_metric': compute_psnr})

def enhance_image_recursively(input_image, steps, initial_flag):
    if steps == 0:
        return input_image

    height, width, channels = input_image.shape
    if initial_flag == 1:
        prediction = enhancement_model.predict(input_image.reshape(1, height, width, 3))
        normalized_input = input_image / 255.0
        enhanced = normalized_input + ((prediction[0] * normalized_input) * (1 - normalized_input))
        psnr = compute_psnr(tf.convert_to_tensor(input_image, dtype=tf.float32), tf.convert_to_tensor(enhanced * 255, dtype=tf.float32)).numpy()
        print(f"PSNR: {psnr:.4f}")
        return enhance_image_recursively(enhanced, steps - 1, 0)
    else:
        prediction = enhancement_model.predict(input_image.reshape(1, height, width, 3))
        enhanced = input_image + ((prediction[0] * input_image) * (1 - input_image))
        psnr = compute_psnr(tf.convert_to_tensor(input_image, dtype=tf.float32), tf.convert_to_tensor(enhanced, dtype=tf.float32)).numpy()
        print(f"PSNR: {psnr:.4f}")
        return enhance_image_recursively(enhanced, steps - 1, initial_flag)

source_dir = 'test/low'
image_paths = glob.glob(source_dir + "/*")
images_array = []

image_paths.sort()
for path in image_paths:
    image = imageio.imread(path)
    images_array.append(image)
images_np = np.array(images_array)

first_image = images_np[0]
result_image = enhance_image_recursively(first_image, 8, 1)

save_dir = 'test/predicted'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'enhanced_image.png')
imageio.imwrite(save_path, (result_image * 255).astype(np.uint8))
print(f"Enhanced image saved at {save_path}")
