import os
import sys
import cv2
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Models.CNN import CNN

image_path = os.path.join(project_root, 'Data', 'test.jpg')

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found")

img_pil = Image.fromarray(img)

cnn = CNN(img_pil)
cnn.forward()

print("CNN output shape:", cnn.output.shape)
print(cnn.output)