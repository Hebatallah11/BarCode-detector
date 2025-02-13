# Barcode Detection System

A modern Python application that implements two different approaches for barcode detection, featuring a user-friendly graphical interface and comprehensive image manipulation capabilities.

![Screenshot 2025-02-08 110956](https://github.com/user-attachments/assets/77215383-c4a3-4620-bf8b-90fdd5fe51a6)


# Features

- Dual detection methods:
    - OpenCV-based detection
    - Custom low-level implementation

- Modern graphical user interface
- Image manipulation tools (zoom, pan, scroll)
- Undo/Redo functionality
- Multiple image format support
- Comprehensive error handling
- Status feedback system

# Requirements

- Python 3.x
- OpenCV 4.x
- NumPy
- SciPy
- Tkinter/ttk
- PIL/Pillow
- Scikit-image

# Usage

1. Run the application:
    python BarCodeDetector.py

2. Using the interface:
- Click "Browse Image" to select an image
- Choose detection method (OpenCV or Low-level)
- Click "Detect Barcode" to process the image
- Use zoom controls or mouse wheel to adjust view
- Use undo/redo buttons to manage changes
- Save results using File → Save Result

# Detection Methods

## OpenCV Method

Uses advanced image processing techniques including:
- Sobel edge detection
- Gradient analysis
- Morphological operations
- Contour detection

## Low-Level Method

Implements fundamental image processing operations:
- Custom convolution
- Basic edge detection
- Region analysis
- Connected component labeling

## GUI Features

- Modern, responsive interface
- Image navigation controls
- Multiple view options
- Status feedback
- Comprehensive menu system
