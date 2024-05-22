# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:44:07 2024

@author: Renan Miranda Portela
"""

''' Import libraries '''
from PIL import Image
import numpy as np

''' Import image and convert to greyscale '''
address = r'C:\Users\rmportel\OneDrive - University of Waterloo\2024 - Winter Semester\Microscopic assessment\Bindered Fabric Inactivated 02\Inactivated_02_Sample_02_500x_008_scale_copy.jpg'
img = Image.open(address).convert('L')

''' Listing pixels in greyscale '''
pixels = list(img.getdata())

''' Evaluate fiber volume fraction '''
Red = 190
Blue = 190
Green = 190
fiber_greyscale = 0.299 * Red + 0.587 * Green + 0.114 * Blue
list_benchmark = np.ones(len(pixels))*fiber_greyscale
fiber = pixels > list_benchmark    
fvf = fiber.sum()/len(fiber)

print('Fiber volume content = {:.2f}%'.format(fvf*100))

''' Color fiber in the figure '''
img = Image.open(address).convert('RGB')
new_pixel = img.load()

for i in range(img.size[0]): # for every pixel:
    for j in range(img.size[1]):
        if new_pixel[i,j] >= (Red, Green, Blue):
            # change the fiber color to red
            new_pixel[i,j] = (255, 0 ,0)
            
img.show()

''' Evaluate void content '''
Red = 70
Blue = 70
Green = 70
void_greyscale = 0.299 * Red + 0.587 * Green + 0.114 * Blue
list_benchmark = np.ones(len(pixels))*void_greyscale
void = pixels < list_benchmark    
void_content = void.sum()/len(void)

print('Void content = {:.2f}%'.format(void_content*100))

''' Color void in the figure '''
img = Image.open(address).convert('RGB')
new_pixel = img.load()

for i in range(img.size[0]): # for every pixel:
    for j in range(img.size[1]):
        if new_pixel[i,j] <= (Red, Green, Blue):
            # change the voids color to yellow
            new_pixel[i,j] = (255, 255 ,0)
            
img.show()