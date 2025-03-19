import SwiftUI
import AVFoundation
import CoreML
import Vision
import Accelerate


struct ContentView: View {
    @State private var image: UIImage? = nil
    @State private var showCamera = false
    @State private var colorizedImage: UIImage? = nil
    @State private var isProcessing = false
    @State private var errorMessage: String? = nil
    @State private var isCancelled = false
    
    @State private var model: colorization_model? = nil
    @State private var isModelLoaded = false
    

    
    var body: some View {
        VStack {
            if isModelLoaded {
                // Main UI
                if let colorizedImage = colorizedImage {
                    Image(uiImage: colorizedImage)
                        .resizable()
                        .scaledToFit()
                } else if let image = image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                    
                    if isProcessing {
                        ProgressView("Processing...")
                            .padding()
                        Button("Cancel") {
                            isCancelled = true
                            isProcessing = false
                            errorMessage = "Processing cancelled."
                        }
                        .padding()
                    } else {
                        Button("Colorize Image") {
                            isProcessing = true
                            isCancelled = false
                            errorMessage = nil
                            DispatchQueue.global(qos: .userInitiated).async {
                                colorizeImage()
                            }
                        }
                        .padding()
                    }
                } else {
                    Text("Tap to capture grayscale image")
                }
                
                Button("Capture Image") {
                    showCamera = true
                }
                .padding()
                .sheet(isPresented: $showCamera) {
                    CameraView(image: $image)
                }
                
                if let errorMessage = errorMessage {
                    Text(errorMessage)
                        .foregroundColor(.red)
                        .padding()
                }
            } else {
                // Show a loading indicator while the model is being loaded
                ProgressView("Loading model...")
                    .padding()
            }
        }
        .onAppear {
            // Load the model asynchronously when the view appears
            loadModel()
        }
    }
    
    func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let model = try colorization_model(configuration: MLModelConfiguration())
                DispatchQueue.main.async {
                    self.model = model
                    self.isModelLoaded = true
                    print("Model loaded successfully")
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to load model."
                    print("Failed to load model: \(error)")
                }
            }
        }
    }
    
    func colorizeImage() {
        guard let image = image, let model = model else {
            DispatchQueue.main.async {
                isProcessing = false
                errorMessage = "Failed to load model."
            }
            return
        }

        // Resize the captured RGB image to 256x256
            let resizedImage = image.resized(to: CGSize(width: 256, height: 256))
            print("Resized image color space: \(resizedImage.cgImage?.colorSpace)")
            
            // Convert the resized RGB image to a CVPixelBuffer
            guard let buffer = resizedImage.toCVPixelBuffer() else {
                DispatchQueue.main.async {
                    isProcessing = false
                    errorMessage = "Failed to process image."
                }
                return
            }
        
        // Debug: Print the CVPixelBuffer properties
        print("CVPixelBuffer width: \(CVPixelBufferGetWidth(buffer))")
        print("CVPixelBuffer height: \(CVPixelBufferGetHeight(buffer))")
        print("CVPixelBuffer format: \(CVPixelBufferGetPixelFormatType(buffer))")
        
        do {
            // Convert the CVPixelBuffer to an MLMultiArray
            let mlArray = try MLMultiArray.fromPixelBuffer(buffer)
            
            // Debug: Print the first 10 values of the MLMultiArray input
            print("MLMultiArray input values: \(Array(0..<10).map { mlArray[$0].floatValue })")
            
            // Create the model input with the correct input name "img_rgb"
            let input = colorization_modelInput(img_rgb: mlArray)
            
            // Perform the prediction
            let output = try model.prediction(input: input)
            
            // Debug: Print the first 10 values of the AB channels
            print("Raw AB channel values: \(Array(0..<10).map { output.output_ab[$0].floatValue })")
            
            // Debug: Print the min and max values of the AB channels
            // Calculate min and max of the MLMultiArray
            var abMin: Float = .greatestFiniteMagnitude
            var abMax: Float = -.greatestFiniteMagnitude
            
            for i in 0..<output.output_ab.count {
                let value = output.output_ab[i].floatValue
                abMin = min(abMin, value)
                abMax = max(abMax, value)
            }
            
            print("AB channel range: \(abMin) to \(abMax)")
            
            if isCancelled {
                DispatchQueue.main.async {
                    isProcessing = false
                }
                return
            }
            
            // Convert the output MLMultiArray to a UIImage
            if let colorizedImage = multiArrayToUIImage(output.output_ab, grayscaleImage: image) {
                DispatchQueue.main.async {
                    self.colorizedImage = colorizedImage
                    isProcessing = false
                }
            }
        } catch {
            DispatchQueue.main.async {
                isProcessing = false
                errorMessage = "Prediction failed: \(error.localizedDescription)"
            }
        }
    }


}

// MARK: - Helper Functions

// Convert MLMultiArray to UIImage
func multiArrayToUIImage(_ multiArray: MLMultiArray, grayscaleImage: UIImage) -> UIImage? {
    let shape = multiArray.shape.map { $0.intValue }
    
    // Ensure the shape is [1, 2, 256, 256]
    guard shape.count == 4, shape[1] == 2, shape[2] == 256, shape[3] == 256 else {
        print("Invalid MLMultiArray shape")
        return nil
    }
    
    let width = shape[3]
    let height = shape[2]
    
    // Extract the AB channels from the MLMultiArray
    var abData = [Float](repeating: 0, count: width * height * 2)
    for i in 0..<abData.count {
        abData[i] = multiArray[i].floatValue
    }
    
    // Debug: Print the first 10 AB channel values
    print("AB channel values: \(abData[0..<10])")
    
    // Convert AB data to RGB image
    return convertABtoRGB(from: abData, grayscaleImage: grayscaleImage, width: width, height: height)
}



// Convert AB data to RGB image (placeholder implementation)
func convertABtoRGB(from abData: [Float], grayscaleImage: UIImage, width: Int, height: Int) -> UIImage? {
    guard abData.count == width * height * 2 else {
        print("Invalid AB data size")
        return nil
    }
    
    let pixelCount = width * height
    
    // Extract the L channel from the grayscale image
    guard let grayscaleCGImage = grayscaleImage.cgImage else {
        print("Failed to get CGImage from grayscale image")
        return nil
    }
    
    let colorSpace = CGColorSpaceCreateDeviceGray()
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width, space: colorSpace, bitmapInfo: CGImageAlphaInfo.none.rawValue)
    context?.draw(grayscaleCGImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    
    guard let lData = context?.data?.assumingMemoryBound(to: UInt8.self) else {
        print("Failed to extract L channel data")
        return nil
    }
    
    // Convert UnsafeMutablePointer<UInt8> to [Float]
    let lChannel = (0..<width * height).map { index -> Float in
        return Float(lData[index]) / 255.0 * 100.0
    }
    
    // Debug: Print the first 10 L channel values
    print("L channel values: \(lChannel[0..<10])")
    
    // Debug: Print the min and max values of the L, A, and B channels
    let lMin = lChannel.min() ?? 0
    let lMax = lChannel.max() ?? 0
    let aMin = abData.min() ?? 0
    let aMax = abData.max() ?? 0
    let bMin = abData.min() ?? 0
    let bMax = abData.max() ?? 0
    
    print("L channel range: \(lMin) to \(lMax)")
    print("A channel range: \(aMin) to \(aMax)")
    print("B channel range: \(bMin) to \(bMax)")
    
    // Normalize A and B channels to [-100, 100]
    let aNormalized = abData.enumerated().map { index, value in
        index % 2 == 0 ? max(-100, min(value, 100)) : value
    }
    let bNormalized = abData.enumerated().map { index, value in
        index % 2 == 1 ? max(-100, min(value, 100)) : value
    }
    
    // Combine L and AB channels to form LAB color space
    var labData = [Float](repeating: 0, count: pixelCount * 3)
    for i in 0..<pixelCount {
        labData[i * 3 + 0] = lChannel[i]           // L channel (normalized to [0, 100])
        labData[i * 3 + 1] = aNormalized[i * 2 + 0] // A channel
        labData[i * 3 + 2] = bNormalized[i * 2 + 1] // B channel
    }
    
    // Convert LAB to RGB
    guard let rgbData = convertLABtoRGB(labData: labData, width: width, height: height) else {
        print("Failed to convert LAB to RGB")
        return nil
    }
    
    // Debug: Print the first 10 RGB values
    print("RGB data: \(rgbData[0..<40])")
    
    // Convert BGRA to RGBA
    var rgbaData = [UInt8](repeating: 0, count: pixelCount * 4)
    for i in 0..<pixelCount {
        rgbaData[i * 4 + 0] = rgbData[i * 4 + 2] // R (swap B and R)
        rgbaData[i * 4 + 1] = rgbData[i * 4 + 1] // G
        rgbaData[i * 4 + 2] = rgbData[i * 4 + 0] // B
        rgbaData[i * 4 + 3] = rgbData[i * 4 + 3] // A
    }
    
    // Create a CGImage from the RGBA data
    let colorSpaceRGB = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue) // Use premultipliedLast for RGBA
    let data = Data(bytes: rgbaData, count: pixelCount * 4)
    let provider = CGDataProvider(data: data as CFData)!
    
    guard let cgImage = CGImage(
        width: width,
        height: height,
        bitsPerComponent: 8,
        bitsPerPixel: 32,
        bytesPerRow: width * 4,
        space: colorSpaceRGB,
        bitmapInfo: bitmapInfo,
        provider: provider,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
    ) else {
        print("Failed to create CGImage")
        return nil
    }
    
    // Debug: Print CGImage properties
    print("CGImage width: \(cgImage.width)")
    print("CGImage height: \(cgImage.height)")
    print("CGImage bitsPerComponent: \(cgImage.bitsPerComponent)")
    print("CGImage bitsPerPixel: \(cgImage.bitsPerPixel)")
    print("CGImage bytesPerRow: \(cgImage.bytesPerRow)")
    print("CGImage colorSpace: \(cgImage.colorSpace)")
    print("CGImage bitmapInfo: \(cgImage.bitmapInfo)")
    
    return UIImage(cgImage: cgImage)
}





func convertLABtoRGB(labData: [Float], width: Int, height: Int) -> [UInt8]? {
    let pixelCount = width * height
    var rgbData = [UInt8](repeating: 0, count: pixelCount * 4)
    
    for i in 0..<pixelCount {
        let l = labData[i * 3 + 0] // L channel (0 to 100)
        let a = labData[i * 3 + 1] // A channel (-100 to 100)
        let b = labData[i * 3 + 2] // B channel (-100 to 100)
        
        // Convert LAB to XYZ
        let y = (l + 16) / 116
        let x = a / 500 + y
        let z = y - b / 200

        let x3 = pow(x, 3)
        let y3 = pow(y, 3)
        let z3 = pow(z, 3)

        let xNorm = x3 > 0.008856 ? x3 : (x - 16 / 116) / 7.787
        let yNorm = y3 > 0.008856 ? y3 : (y - 16 / 116) / 7.787
        let zNorm = z3 > 0.008856 ? z3 : (z - 16 / 116) / 7.787

        // D65 reference white point
        let xFinal = xNorm * 95.047 / 95.047 // Normalize X
        let yFinal = yNorm * 100.0 / 100.0   // Normalize Y
        let zFinal = zNorm * 108.883 / 108.883 // Normalize Z
        
        // Debug: Print XYZ values
        if i < 10 { // Print for the first 10 pixels
            print("XYZ values for pixel \(i): \(xFinal), \(yFinal), \(zFinal)")
        }
        
        // Convert XYZ to RGB
        var red = xFinal * 3.2406 + yFinal * -1.5372 + zFinal * -0.4986
        var green = xFinal * -0.9689 + yFinal * 1.8758 + zFinal * 0.0415
        var blue = xFinal * 0.0557 + yFinal * -0.2040 + zFinal * 1.0570
        
        // Apply gamma correction
        red = red > 0.0031308 ? 1.055 * pow(red, 1 / 2.4) - 0.055 : 12.92 * red
        green = green > 0.0031308 ? 1.055 * pow(green, 1 / 2.4) - 0.055 : 12.92 * green
        blue = blue > 0.0031308 ? 1.055 * pow(blue, 1 / 2.4) - 0.055 : 12.92 * blue
        
        // Clamp values to [0, 1]
        red = max(0, min(1, red))
        green = max(0, min(1, green))
        blue = max(0, min(1, blue))
        
        // Debug: Print RGB values
        if i < 10 { // Print for the first 10 pixels
            print("RGB values for pixel \(i): \(red), \(green), \(blue)")
        }
        
        // Convert to 8-bit RGB
        rgbData[i * 4 + 0] = UInt8(red * 255)
        rgbData[i * 4 + 1] = UInt8(green * 255)
        rgbData[i * 4 + 2] = UInt8(blue * 255)
        rgbData[i * 4 + 3] = 255 // Alpha channel
    }
    
    return rgbData
}





// MARK: - UIImage Extension

extension UIImage {
    // Resize the image to a specific size
    func resized(to size: CGSize) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
    }
    
    // Convert UIImage to CVPixelBuffer
    func toCVPixelBuffer() -> CVPixelBuffer? {
            let attrs = [
                kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
                kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
            ] as CFDictionary
            
            var pixelBuffer: CVPixelBuffer?
            let width = Int(self.size.width)
            let height = Int(self.size.height)
            
            let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pixelBuffer)
            
            guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
                print("Failed to create CVPixelBuffer with status: \(status)")
                return nil
            }
            
            CVPixelBufferLockBaseAddress(buffer, .init(rawValue: 0))
            defer { CVPixelBufferUnlockBaseAddress(buffer, .init(rawValue: 0)) }
            
            let context = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer),
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            )
            
            guard let cgImage = self.cgImage else {
                print("Failed to get CGImage from UIImage")
                return nil
            }
            
            context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return buffer
        }
}

// MARK: - MLMultiArray Extension

extension MLMultiArray {
    // Convert CVPixelBuffer to MLMultiArray
    static func fromPixelBuffer(_ pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        // Lock the pixel buffer
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        // Get the base address of the pixel buffer
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Failed to get base address of CVPixelBuffer")
            throw NSError(domain: "com.yourapp", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to get pixel buffer base address"])
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let channels = 3 // RGB input
        
        // Create the MLMultiArray with the correct shape
        let shape = [1, NSNumber(value: channels), NSNumber(value: height), NSNumber(value: width)]
        let mlArray = try MLMultiArray(shape: shape, dataType: .float32)
        
        // Access the pixel buffer data as UInt8 (BGRA format)
        let data = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        // Populate the MLMultiArray with RGB values
        for y in 0..<height {
            for x in 0..<width {
                // Calculate the pixel index in the CVPixelBuffer
                let pixelIndex = y * width + x
                
                // Extract R, G, B channels from BGRA format
                let r = Float(data[pixelIndex * 4 + 2]) / 255.0 // Red channel
                let g = Float(data[pixelIndex * 4 + 1]) / 255.0 // Green channel
                let b = Float(data[pixelIndex * 4 + 0]) / 255.0 // Blue channel
                
                // Populate the MLMultiArray
                mlArray[y * width + x] = NSNumber(value: r)
                mlArray[width * height + y * width + x] = NSNumber(value: g)
                mlArray[2 * width * height + y * width + x] = NSNumber(value: b)
            }
        }
        
        return mlArray
    }


}

// MARK: - CameraView

struct CameraView: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }
    
    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        picker.cameraDevice = .rear
        picker.mediaTypes = ["public.image"]
        picker.cameraCaptureMode = .photo
        return picker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    
    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        var parent: CameraView
        init(_ parent: CameraView) { self.parent = parent }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage {
                print("Captured image color space: \(image.cgImage?.colorSpace)")
                parent.image = image
            }
            picker.dismiss(animated: true)
        }
    }
}

// MARK: - App Entry Point

@main
struct ColorizationPlaygroundApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
