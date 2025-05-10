import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F


# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 36)  # 36 classes (0-9, A-Z)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(-1, 64 * 5 * 5)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the trained model
model = CNN()
model.load_state_dict(torch.load("emnist_cnn.pth"))
model.eval()

# Label map (same as training)
labels_map = {i: chr(48 + i) if i < 10 else chr(65 + i - 10) for i in range(36)}  # 0-9, A-Z


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Character Prediction")

        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack(pady=10)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        self.predict_btn = tk.Button(button_frame, text="🔍 Predict", command=self.predict,
                                     bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.predict_btn.grid(row=0, column=0, padx=5)

        self.clear_btn = tk.Button(button_frame, text="🧹 Clear", command=self.clear_canvas,
                                   bg="#f44336", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.clear_btn.grid(row=0, column=1, padx=5)

        self.result_label = tk.Label(root, text="Prediction: None", font=("Arial", 16, "bold"), fg="blue")
        self.result_label.pack(pady=10)

        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        radius = 8
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='black')
        self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: None", fg="blue")

    def predict(self):
        self.image.save("drawed-original.png")

        # Resize and invert for EMNIST
        img_resized = self.image.resize((28, 28))
        img_inverted = ImageOps.invert(img_resized)
        img_inverted.save("drawed-processed.png")

        arr = torch.tensor(np.array(img_inverted), dtype=torch.float32)
        img_tensor = arr.unsqueeze(0)  # [1, 28, 28]

        # Apply horizontal flip and 90° rotation
        img_tensor = F.rotate(F.hflip(img_tensor), 90)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)  # [1, 1, 28, 28]

        with torch.no_grad():
            output = model(img_tensor)
            prediction = output.argmax(dim=1).item()
            label = labels_map[prediction]
            print(f"Predicted class: {prediction}, Label: {label}")
            self.result_label.config(text=f"🧠 Prediction: {label}", fg="green")


if __name__ == "__main__":
    # Launch the app
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
