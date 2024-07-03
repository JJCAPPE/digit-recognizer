import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Plotting:
    @staticmethod
    def showDrawing(csv):
        csv_data = pd.read_csv(csv)
        pixel_data = np.array(csv_data.iloc[0, 1:])
        pixel_data = np.where(pixel_data < 1, 0, 255)
        image = pixel_data.reshape(28, 28)
 
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title('Handwritten Digit')
        plt.show()