# SmartColorizer (DDColor iOS Implementation) (Partially Functional)

This project is an iOS implementation of the **DDColor** model, originally developed by [piddnad](https://github.com/piddnad/DDColor). The goal of this project is to bring the powerful image colorization capabilities of DDColor to iOS devices, leveraging Core ML for efficient on-device processing.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Model Source](#model-source)
4. [Features](#features)
5. [What Works](#what-works)
6. [Known Issues](#known-issues)
7. [Potential Solutions](#potential-solutions)
8. [How to Use](#how-to-use)
9. [Dependencies](#dependencies)
10. [Contributing](#contributing)
11. [Acknowledgments](#acknowledgments)

---

## Introduction
This project aims to implement the **DDColor** model on iOS for grayscale image colorization. The model, originally developed for Python-based environments, has been converted to Core ML format to enable real-time image colorization on iOS devices. This implementation uses a combination of Core ML, Vision, and UIKit to process images and generate colorized outputs.

---

## Objectives
The primary objectives of this project are:
1. **Port the DDColor model to iOS** using Core ML for efficient on-device inference.
2. **Develop a user-friendly iOS app** that allows users to upload grayscale images and view colorized results.
3. **Optimize the model** for mobile devices, ensuring fast processing times while maintaining output quality.
4. **Document the process** to enable other developers to replicate or extend the implementation.

---

## Model Source
The original DDColor model is sourced from the GitHub repository:  
[piddnad/DDColor](https://github.com/piddnad/DDColor)

The model was trained on a large dataset of images and is designed to predict plausible colorizations for grayscale images. This implementation adapts the model for iOS by converting it to Core ML format.

---

## Features
- **Grayscale-to-Color Conversion**: Convert grayscale images to colorized versions using the DDColor model.
- **Real-Time Processing**: Optimized for on-device processing with Core ML and Vision frameworks.
- **User-Friendly Interface**: Simple iOS app interface for uploading images and viewing results.

---

## What Works
- **Core ML Model Integration**: The DDColor model has been successfully converted to Core ML and integrated into the iOS app.
- **Basic Image Processing**: The app can process grayscale images and generate colorized outputs.
- **Debugging Tools**: The app includes logging and debugging tools to monitor the model’s performance and output.

---

## Known Issues
1. **Blueish Tint in Output**: The colorized images sometimes have a blueish tint due to incorrect channel order or pixel format issues.
2. **Model Accuracy**: The model occasionally produces unrealistic colorizations, especially for complex or abstract images.
3. **Performance on Older Devices**: The model may be slow on older iOS devices with limited computational resources.
4. **Input Image Constraints**: The model expects grayscale images of a specific size (256x256), and resizing may affect output quality.

---

## Potential Solutions
1. **Blueish Tint**:
   - Ensure the RGB channels are correctly ordered (`RGBA` instead of `BGRA`).
   - Normalize the LAB and XYZ values to prevent blown-out colors.
2. **Model Accuracy**:
   - Fine-tune the model using additional training data or domain-specific images.
   - Implement post-processing techniques (e.g., contrast adjustment) to improve output quality.
3. **Performance Optimization**:
   - Quantize the Core ML model to reduce its size and improve inference speed.
   - Use Metal Performance Shaders (MPS) for hardware-accelerated processing.
4. **Input Image Constraints**:
   - Implement better image resizing algorithms (e.g., Lanczos interpolation) to maintain image quality.
   - Add support for variable input sizes by modifying the Core ML model.

---

## How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/ddcolor-ios.git
   cd ddcolor-ios
   
2.  **Download and Load Model**:
    Download the unzip ml.package file and place it in root directory. The zip file can be obtained [here](https://drive.google.com/drive/folders/16W1XdZ2DYbTHlsalAdpTzh69LIgQ8B4P?usp=sharing).
    
3.  **Open the Project**:
    Open the Xcode project (DDColor.xcodeproj) and ensure all dependencies are installed.

4. **Run the App**:
   Build and run the app on a simulator or physical device.

5. **Upload an Image**:
   Use the app interface to upload a grayscale image and view the colorized result.

---

## Dependencies

**Core ML**: For model inference.

**Vision**: For image processing and analysis.

**UIKit**: For building the app interface.

**Core Image**: For additional image manipulation tasks.

---

## Contributing 

Contributions are welcome! If you’d like to contribute to this project, please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bug fix.

3. Commit your changes and push to the branch.

4. Submit a pull request with a detailed description of your changes.

---

## Acknowledgments

[**piddnad**](https://github.com/piddnad): For developing the original DDColor model.

**Apple**: For providing Core ML and Vision frameworks.

**Open Source Community**: For tools and libraries that made this project possible.

---

## Contact

For questions or feedback, please open an issue on GitHub or contact the project maintainer.
