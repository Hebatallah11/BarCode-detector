
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.signal import convolve2d
from skimage.measure import label, regionprops
import os


class MyButton(ttk.Button):
    """Styled button class"""
    def __init__(self, master=None, **kwargs):
        style = ttk.Style()
        style.configure("Modern.TButton", padding=10, font=('Helvetica', 10))
        super().__init__(master, style="Modern.TButton", **kwargs)

class BarcodeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Barcode Detector")
        self.root.geometry("1200x800")
        
        # Configure style
        self.setup_styles()
        
        # Initialize variables
        self.current_image_path = None
        self.current_image = None
        self.processed_image = None
        self.zoom_factor = 1.0
        
        # Create main layout
        self.create_layout()
        
        # Create menu bar
        self.create_menu()
        
        # Initialize image history
        self.image_history = []
        self.current_history_index = -1
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure("Header.TLabel", font=('Helvetica', 16, 'bold'), padding=10)
        style.configure("Status.TLabel", font=('Helvetica', 10), padding=5)
        style.configure("Modern.TRadiobutton", font=('Helvetica', 10), padding=5)
        style.configure("Controls.TFrame", padding=10)
        
    def create_layout(self):
        # Create main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create left and right panels
        self.create_left_panel()
        self.create_right_panel()
        
        # Configure main container grid weights
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        
    def create_left_panel(self):
        # Control panel (left side)
        control_panel = ttk.Frame(self.main_container, style="Controls.TFrame")
        control_panel.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        
        # Header
        ttk.Label(control_panel, text="Controls", style="Header.TLabel").pack(pady=(0, 10))
        
        # Browse button
        MyButton(control_panel, text="Browse Image", 
                    command=self.browse_image).pack(fill="x", pady=5)
        
        # Method selection frame
        method_frame = ttk.LabelFrame(control_panel, text="Detection Method", padding=10)
        method_frame.pack(fill="x", pady=10)
        
        self.method_var = tk.StringVar(value="opencv")
        ttk.Radiobutton(method_frame, text="OpenCV Method", 
                       variable=self.method_var, value="opencv",
                       style="Modern.TRadiobutton").pack(fill="x", pady=2)
        ttk.Radiobutton(method_frame, text="Low-Level Method", 
                       variable=self.method_var, value="lowlevel",
                       style="Modern.TRadiobutton").pack(fill="x", pady=2)
        
        # Detection button
        MyButton(control_panel, text="Detect Barcode",
                    command=self.detect_barcode).pack(fill="x", pady=5)
        
        # Zoom controls
        zoom_frame = ttk.LabelFrame(control_panel, text="Zoom Controls", padding=10)
        zoom_frame.pack(fill="x", pady=10)
        
        MyButton(zoom_frame, text="Zoom In", 
                    command=lambda: self.zoom_image(1.2)).pack(fill="x", pady=2)
        MyButton(zoom_frame, text="Zoom Out", 
                    command=lambda: self.zoom_image(0.8)).pack(fill="x", pady=2)
        MyButton(zoom_frame, text="Reset Zoom", 
                    command=lambda: self.zoom_image(reset=True)).pack(fill="x", pady=2)
        
        # History controls
        history_frame = ttk.LabelFrame(control_panel, text="History", padding=10)
        history_frame.pack(fill="x", pady=10)
        
        MyButton(history_frame, text="Undo", 
                    command=self.undo).pack(fill="x", pady=2)
        MyButton(history_frame, text="Redo", 
                    command=self.redo).pack(fill="x", pady=2)
        
        # Status label
        self.status_label = ttk.Label(control_panel, text="Ready", 
                                    style="Status.TLabel", wraplength=200)
        self.status_label.pack(fill="x", pady=10)
        
    def create_right_panel(self):
        # Image panel (right side)
        self.image_frame = ttk.Frame(self.main_container)
        self.image_frame.grid(row=0, column=1, sticky="nsew")
        
        # Canvas for image display with scrollbars
        self.canvas = tk.Canvas(self.image_frame, bg='#f0f0f0')
        self.scrollbar_y = ttk.Scrollbar(self.image_frame, orient="vertical", 
                                       command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self.image_frame, orient="horizontal", 
                                       command=self.canvas.xview)
        
        # Configure canvas
        self.canvas.configure(xscrollcommand=self.scrollbar_x.set, 
                            yscrollcommand=self.scrollbar_y.set)
        
        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        # Bind mouse events for panning
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.browse_image)
        file_menu.add_command(label="Save Result", command=self.save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo)
        edit_menu.add_command(label="Redo", command=self.redo)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)
        
    def pan_image(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        
    def mouse_wheel(self, event):
        if event.delta > 0:
            self.zoom_image(1.1)
        else:
            self.zoom_image(0.9)
        
    def zoom_image(self, factor=1.0, reset=False):
        if reset:
            self.zoom_factor = 1.0
        else:
            self.zoom_factor *= factor
            
        if self.current_image:
            self.display_image(zoom=True)
            
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image()
            self.status_label.config(text="Image loaded. Click 'Detect Barcode' to process.")
            self.add_to_history(self.current_image)
            
    def display_image(self, image=None, zoom=False):
        if image is None and self.current_image_path and not zoom:
            # Load new image
            self.current_image = Image.open(self.current_image_path)
        elif image is not None:
            # Use provided image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(image)
            else:
                self.current_image = image
                
        if self.current_image:
            # Calculate new size based on zoom factor
            new_width = int(self.current_image.width * self.zoom_factor)
            new_height = int(self.current_image.height * self.zoom_factor)
            
            # Resize image
            resized_image = self.current_image.resize((new_width, new_height), 
                                                    Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(resized_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=photo)
            self.canvas.image = photo  # Keep a reference
            
            # Update scrollregion
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
    def add_to_history(self, image):
        # Remove any redo history
        if self.current_history_index < len(self.image_history) - 1:
            self.image_history = self.image_history[:self.current_history_index + 1]
            
        self.image_history.append(image)
        self.current_history_index = len(self.image_history) - 1
        
    def undo(self):
        if self.current_history_index > 0:
            self.current_history_index -= 1
            self.current_image = self.image_history[self.current_history_index]
            self.display_image(self.current_image)
            
    def redo(self):
        if self.current_history_index < len(self.image_history) - 1:
            self.current_history_index += 1
            self.current_image = self.image_history[self.current_history_index]
            self.display_image(self.current_image)
            
    def save_result(self):
        if self.current_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), 
                          ("JPEG files", "*.jpg"),
                          ("All files", "*.*")]
            )
            if file_path:
                self.current_image.save(file_path)
                self.status_label.config(text=f"Image saved to {file_path}")
                
    def show_about(self):
        messagebox.showinfo(
            "About Barcode Detector",
            "Barcode Detector\n\n"
            "This is the final project for the image processing course for the 2024\\2025 semester.\n"
            "Done by Hebatallah AbuHarb - 220210448\n\n"
            "This application provides barcode detection using both OpenCV "
            "and low-level image processing methods.\n\n"
            "Features:\n"
            "- Multiple detection methods\n"
            "- Zoom and pan capabilities\n"
            "- Undo/Redo functionality\n"
            "- Image saving\n\n"
            "Created with ❤️ using Python By Hebatallah AbuHarb"
        )
    
    
    def detect_opencv(self):
        # Load the image
        image = cv2.imread(self.current_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        
        # Process image
        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            result = image.copy()
            cv2.drawContours(result, [box], -1, (0, 255, 0), 2)
            self.display_image(image=result)
            self.add_to_history(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))
            self.status_label.config(text="Barcode detected using OpenCV method")
        else:
            self.status_label.config(text="No barcode detected using OpenCV method")
    
    def detect_lowlevel(self):
        # Load image
        image = np.array(Image.open(self.current_image_path).convert('L'))
        
        # Apply Sobel operator
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gradient_x = convolve2d(image, sobel_x, mode='same', boundary='symm')
        gradient_y = convolve2d(image, sobel_y, mode='same', boundary='symm')
        gradient = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Process image
        kernel = np.ones((3, 3)) / 9
        blurred = convolve2d(gradient, kernel, mode='same', boundary='symm')
        binary = np.clip(blurred, 0, 255) > 128
        
        # Morphological operations
        kernel = np.ones((3, 3))
        dilated = np.maximum(binary, convolve2d(binary, kernel, mode='same', boundary='symm'))
        
        # Find contours
        labeled_image = label(dilated)
        regions = regionprops(labeled_image)
        
        if regions:
            # Get largest region
            largest_region = max(regions, key=lambda r: r.area)
            min_row, min_col, max_row, max_col = largest_region.bbox
            
            # Create RGB result image
            result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Draw bounding box
            cv2.rectangle(result_image, (min_col, min_row), (max_col, max_row), (255, 0, 0), 2)
            
            # Update display and history
            self.display_image(image=result_image)
            self.add_to_history(Image.fromarray(result_image))
            self.status_label.config(text="Barcode detected using low-level method")
        else:
            self.status_label.config(text="No barcode detected using low-level method")

    def detect_barcode(self):
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        try:
            method = self.method_var.get()
            if method == "opencv":
                self.detect_opencv()
            else:
                self.detect_lowlevel()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during detection: {str(e)}")
            self.status_label.config(text="Error during detection")

def main():
    root = tk.Tk()
    root.title("Advanced Barcode Detector")
    
    # Set theme
    style = ttk.Style()
    try:
        style.theme_use('clam')  # Use 'clam' theme for a modern look
    except:
        pass  # Fallback to default theme if 'clam' is not available
    
    # Configure default styles
    style.configure(".", font=('Helvetica', 10))
    style.configure("TButton", padding=5)
    style.configure("TLabel", padding=2)
    
    # Create and run application
    app = BarcodeDetectorGUI(root)
    
    # Center window on screen
    window_width = 1200
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Make window resizable
    root.resizable(True, True)
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()