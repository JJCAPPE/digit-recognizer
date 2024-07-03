import tkinter as tk
import numpy as np
import csv
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DrawingApp:
    def __init__(self, root):
        self.root = root
    
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.left_frame, width=280, height=280, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_paint)

        self.pixel_data = np.zeros((28, 28), dtype=int)
        self.last_x = None
        self.last_y = None
        self.update_pending = False
        
        self.reset_button = tk.Button(self.left_frame, text="Reset", command=self.reset_canvas)
        self.reset_button.pack()
        self.save_button = tk.Button(self.left_frame, text="Save", command=self.save_drawing_and_predict)
        self.save_button.pack()
        self.prediction_label = tk.Label(self.left_frame, text="", font=("Helvetica", 24, "bold"))  
        self.info_label = tk.Label(self.left_frame, text="")
        self.prediction_label.pack()
        self.info_label.pack()
        
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas_widget = self.canvas_plot.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=30, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.update_pixel_data(self.last_x, self.last_y, x, y)
        self.last_x = x
        self.last_y = y
        
        if not self.update_pending:
            self.update_pending = True
            self.root.after(10, self.update_canvas)

    def reset_paint(self, event):
        self.last_x = None
        self.last_y = None

    def update_canvas(self):
        self.update_pending = False
        self.canvas.update_idletasks()

    def update_pixel_data(self, x1, y1, x2, y2):
        # Calculate the slope and intercept of the line
        if x2 - x1 == 0:  # Vertical line
            x_coords = [x1] * max(abs(y2 - y1) + 1, 1)
            y_coords = list(range(min(y1, y2), max(y1, y2) + 1))
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Generate points along the line
            x_coords = list(range(min(x1, x2), max(x1, x2) + 1))
            y_coords = [int(slope * x + intercept) for x in x_coords]

        # Update pixel_data
        for x, y in zip(x_coords, y_coords):
            if 0 <= x < 280 and 0 <= y < 280:
                self.pixel_data[y // 10, x // 10] = 255

    def reset_canvas(self):
        self.canvas.delete("all")
        self.pixel_data.fill(0)
        self.last_x = None
        self.last_y = None

    def save_drawing_and_predict(self):
        flattened_data = self.pixel_data.flatten()
        headers = [f'pixel{i}' for i in range(784)]
        with open('drawing.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(flattened_data)
        
        with open('parameters_stoch_250.pkl', 'rb') as f:
            W1, b1, W2, b2, W3, b3 = pickle.load(f)

        if os.path.exists('drawing.csv'):
            start = time.time()

            drawing = pd.read_csv('drawing.csv')
            drawing = np.array(drawing).T

            X_test = drawing / 255.0

            _, _, _, _, _, A3 = self.forward_prop(W1, b1, W2, b2, W3, b3, X_test)

            prediction = self.get_predictions(A3)

            predicted = prediction[0]

            certainty = A3[predicted]

            taken = time.time() - start

            self.prediction_label.config(text=f"Predicted digit: {predicted}")

            self.info_label.config(text=f"Certainty: {certainty[0]:.4f} Time Taken: {taken:.4f} s")

            A3_flat = A3.flatten()

            labels = list(range(len(A3_flat)))

            self.ax.clear()

            plt.bar(labels, A3_flat)

            self.ax.bar(labels, A3_flat)
            self.ax.set_title('Bar Chart of Array A3')
            self.ax.set_xlabel('Index')
            self.ax.set_ylabel('Value')

            for i, v in enumerate(A3_flat):
                self.ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
            
            self.ax.set_xticks(labels)

            self.fig.tight_layout()
            self.canvas_plot.draw()
    
            os.remove('drawing.csv')

    def forward_prop(self, W1, b1, W2, b2, W3, b3, X):
        Z1 = W1.dot(X) + b1
        A1 = self.leaky_relu(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.leaky_relu(Z2)
        Z3 = W3.dot(A2) + b3
        A3 = self.softmax(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    def leaky_relu(self, Z, alpha=0.01):
        return np.maximum(alpha * Z, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def get_predictions(self, A3):
        return np.argmax(A3, axis=0)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1500x800") 
    app = DrawingApp(root)
    root.mainloop()
