# Document Authenticator


Welcome to the Document Authenticator! This is a simple, powerful tool designed to help you figure out if a document or signature has been digitally faked, altered, or forged.

You don't need to be a computer expert to use it. Just upload an image of your document, and the software will analyze it for you.

---

## 🔍 What can this tool do?

### 1. Check if a Signature is Fake (Signature Verification)
If you have a known, real signature from someone, you can upload it here alongside a newly signed document. The AI will compare the handwriting style, curves, and strokes to tell you if they match, or if the new one is a forgery.

### 2. Find Pasted Elements (Copy-Move Forgery)
Sometimes, a forger will copy a piece of a document (like a date, a stamp, or a signature) and paste it somewhere else on the same page. This tool scans the entire document to find exact duplicate regions and highlights them for you.

### 3. Spot Hidden Alterations (Document Analysis)
If someone takes a picture of a document, erases a number in Photoshop, and types in a new number, it might look perfect to the human eye. This tool uses three different x-ray-like methods to expose the truth:
* **Error Level Analysis (ELA):** When an image is saved in Photoshop, the edited parts save differently than the original parts. This tool creates a heatmap that makes edited areas (like a pasted stamp or changed text) glow bright red so you can easily spot them.
* **Edge Detection:** This strips away all the colors and shows you only the hard outlines of the document. This makes it easy to spot if a piece of text or a logo was "spliced" or cut out from another piece of paper.
* **Wavelet Scanner:** This looks at the hidden "background noise" of the image. If someone erased something, the background noise in that specific spot will be totally smooth, giving them away.

---

## ⚡ How to Run the Software

This software is designed to be incredibly easy to start. You don't need to install any complex coding environments.

**Just do this:**
1. Open the folder.
2. Double-click the file named `Run DocAuth.bat`.
3. Wait a few seconds. The black window will automatically check the internet for any new updates from the developer, download them, and then instantly open the tool in your web browser. 

*If you don't have internet access, don't worry! It will safely skip the update check and just open the software normally.*
