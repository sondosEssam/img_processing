import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        self.filters = [
            "LPF", "HPF", "MEAN", "MEDIAN",
            "Roberts", "Prewitt", "Sobel",
            "Erosion", "Dilation", "Open", "Close",
            "Hough Circles", "Segmentation Thresholding"
        ]

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.load_image_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_image_button.pack()

        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack()

        for filter_name in self.filters:
            button = tk.Button(self.buttons_frame, text=filter_name, command=lambda f=filter_name: self.apply_filter(f))
            button.pack(side=tk.LEFT)

        self.original_image = None
        self.display_image = None
        self.filtered_image_window = None


    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[
                ("All Files", "*.*"),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpeg"),
                ("JPG Files", "*.jpg"),
                ("Bitmap Files", "*.bmp"),
                ("TIFF Files", "*.tiff"),
                ("GIF Files", "*.gif"),
            ]
        )
        if file_path:
            self.original_image = Image.open(file_path).convert('RGB')
            self.display_image_on_label(self.original_image)

    def display_image_on_label(self, image):
        self.display_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.display_image)

    def apply_filter(self, filter_name):
        if self.original_image:
            image_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            if filter_name == "LPF":
                filtered_image = self.LPF(image_cv)
            elif filter_name == "HPF":
                filtered_image = self.HPF(image_cv)
            elif filter_name == "MEAN":
                filtered_image = self.MEAN(image_cv)
            elif filter_name == "MEDIAN":
                filtered_image = self.MEDIAN(image_cv)
            elif filter_name == "Roberts":
                filtered_image = self.Roberts(image_cv)
            elif filter_name == "Prewitt":
                filtered_image = self.Prewitt(image_cv)
            elif filter_name == "Sobel":
                filtered_image = self.Sobel(image_cv)
            elif filter_name == "Erosion":
                filtered_image = self.Erosion(image_cv)
            elif filter_name == "Dilation":
                filtered_image = self.Dilation(image_cv)
            elif filter_name == "Open":
                filtered_image = self.Open(image_cv)
            elif filter_name == "Close":
                filtered_image = self.Close(image_cv)
            elif filter_name == "Hough Circles":
                filtered_image = self.Hough_Circles(image_cv)
            elif filter_name == "Segmentation Thresholding":
                filtered_image = self.Segmentation_Thresholding(image_cv)
            else:
                filtered_image = image_cv

            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
            filtered_image_pil = Image.fromarray(filtered_image)
            self.show_filtered_image(filtered_image_pil)


    def show_filtered_image(self, image):
        if self.filtered_image_window is not None:
            self.filtered_image_window.destroy()

        self.filtered_image_window = tk.Toplevel(self.root)
        self.filtered_image_window.title("Filtered Image")
        filtered_image_label = tk.Label(self.filtered_image_window)
        filtered_image_label.pack()
        display_image = ImageTk.PhotoImage(image)
        filtered_image_label.config(image=display_image)
        filtered_image_label.image = display_image  # Keep a reference to avoid garbage collection

    def LPF(self, image):
        # Low-Pass Filter (Gaussian Blur)
        return cv2.GaussianBlur(image, (15, 15), 0)

    def HPF(self, image):
        # High-Pass Filter (Laplacian)
        lpf_image = self.LPF(image)
        return cv2.subtract(image, lpf_image)

    def MEAN(self, image):
        # Mean Filter (Average Blur)
        return cv2.blur(image, (5, 5))

    def MEDIAN(self, image):
        # Median Filter
        return cv2.medianBlur(image, 5)

    def Roberts(self, image):
        # Roberts Cross Edge Detector
        kernelx = np.array([[1, 0], [0, -1]], dtype=int)
        kernely = np.array([[0, 1], [-1, 0]], dtype=int)
        x = cv2.filter2D(image, cv2.CV_16S, kernelx)
        y = cv2.filter2D(image, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    def Prewitt(self, image):
        # Prewitt Edge Detector
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv2.filter2D(image, cv2.CV_16S, kernelx)
        y = cv2.filter2D(image, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    def Sobel(self, image):
        # Sobel Edge Detector
        grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    def Erosion(self, image):
        # Erosion
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def Dilation(self, image):
        # Dilation
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def Open(self, image):
        # Opening
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def Close(self, image):
        # Closing
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def Hough_Circles(self, image):
        # Hough Transform for Circles
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=15, maxRadius=40)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        return image

    def Segmentation_Thresholding(self, image):
        # Segmentation using thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


if __name__== "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)

    root.mainloop()
