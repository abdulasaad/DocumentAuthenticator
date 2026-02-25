# Document Authenticator


A professional digital forensics suite built to analyze document manipulation, detect copy-move forgeries, and verify offline signatures using state-of-the-art computer vision and deep learning techniques. 

The system leverages an ensemble of AI and mathematical algorithms including PyTorch Siamese networks, ORB+RANSAC homography, Error Level Analysis (ELA), robust edge detection filters, and Optical Character Recognition (OCR).

---

## 🔍 Core Features

### 1. Offline Signature Verification
Upload a reference (enrolled) signature and a queried signature. The system uses a trained PyTorch Siamese Network backbone to extract embeddings and calculate the cosine distance between the two signatures to determine authenticity.
* *Note: Requires a pre-trained `weights/siamese_best.pt` file to operate.*

### 2. Copy-Move Forgery Detection
Detects regions of a document that have been cloned and pasted elsewhere within the same image.
* Uses **ORB** (Oriented FAST and Rotated BRIEF) for high-speed feature extraction.
* Uses **RANSAC** (Random Sample Consensus) to strictly mathematically verify the geometric arrangement of cloned keypoints, rejecting false positives.

### 3. Comprehensive Document Analysis
A multi-tool workstation for analyzing a single document artifact:
* **Error Level Analysis (ELA):** Highlights regions of the image that degrade at different rates when subjected to heavy JPEG compression, exposing pasted digital elements. Features an autonomous OpenCV contour-mapping engine that dynamically draws bounding boxes around anomalies (>3.5 standard deviations from the document mean).
* **Edge Detection:** Extracts structural outlines to find hard splicing lines using Canny, Sobel, Laplacian, or Prewitt algorithms.
* **Optical Character Recognition (OCR):** Extracts raw text using EasyOCR, or switches to TrOCR for complex handwritten documents.
* **Wavelet Decomposition:** Analyzes the high-frequency domain of the document noise using PyWavelets to find invisible tampering boundaries.

---

## ⚡ Setup & Deployment

This application includes a custom zero-touch Windows management launcher designed for portable/offline use.

### Prerequisites (For developers)
If you are running the source code directly without the portable binary:
1. Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```

### End-User Execution (Windows)
To run the software, simply double-click the included `Run DocAuth.bat` file. 

The boot sequence will automatically:
1. Boot the built-in portable Python environment.
2. Silently ping this GitHub repository utilizing `git pull` to fetch the absolute latest features and bug fixes.
3. Launch the Streamlit backend server.
4. Open your default web browser to the application dashboard.

*Note: If the PC is offline or Git is not installed, the auto-updater will safely bypass and launch the local software version immediately.*
