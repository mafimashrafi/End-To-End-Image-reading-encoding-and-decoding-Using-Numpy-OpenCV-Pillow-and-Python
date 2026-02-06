import numpy as np 
import time 
from PIL import Image
import cv2

class CNN:
    def __init__(self, img, pad=0, s=2):
        self.image = img.convert("L").resize((128,128))
        self.padding = pad
        self.stride = s
        self.kernel_size = 3
        self.output = []

    def convo_ReLU(self, img_array):
        kernel = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

        if self.padding > 0:
            img_array = np.pad(
                img_array,
                ((self.padding, self.padding),
                (self.padding, self.padding)),
                mode='constant'
            )

        out_size = ((img_array.shape[0] - self.kernel_size) // self.stride) + 1
        output = np.zeros((out_size, out_size))

        out_i = 0
        for i in range(0, img_array.shape[0]-2, self.stride):
            out_j = 0
            for j in range(0, img_array.shape[1]-2, self.stride):
                patch = img_array[i:i+3, j:j+3]
                val = np.sum(patch * kernel)
                output[out_i, out_j] = max(0, val) 
                out_j += 1
            out_i += 1

        return output

    def max_pool(self, img_array, pool_size = 2, stride = 2):
        h, w = img_array.shape
        out_h = (h-pool_size) // stride +1
        out_w = (w-pool_size) // stride +1
        
        output = np.zeros((out_h, out_w))
        
        for i in range(0, h-pool_size + 1, stride):
            for j in range(0, w - pool_size +1, stride):
                output[i//stride, j//stride] = np.max(
                    img_array[i:i+pool_size, j:j+pool_size]
                )
                
        return output
    
    def forward(self):
        img_array = np.array(self.image)
        conv1 = self.convo_ReLU(img_array)
        pool1=self.max_pool(conv1)
        conv2=self.convo_ReLU(pool1)
        pool2=self.max_pool(conv2)
        
        self.output = pool2.flatten()