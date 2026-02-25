"""
DocAuth — Document Forgery Detection and Analysis
Streamlit multi-tab application.

Run with:
    streamlit run app.py

Tabs:
  1. Signature Verification    — Siamese network pair comparison
  2. Copy-Move Detection       — ORB+RANSAC / photoholmes
  3. Document Analysis         — ELA, edge detection, OCR, wavelet
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Authenticator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Document Authenticator")
st.caption("Powered by PyTorch Siamese networks · ORB+RANSAC · ELA · EasyOCR · PyWavelets | **By Abdullah Saad**")

# Inject CSS to prevent giant images from making the user scroll continuously 
st.markdown("""
    <style>
        [data-testid="stImage"] img {
            max-height: 500px !important;
            width: auto !important;
            object-fit: contain;
        }
    </style>
""", unsafe_allow_html=True)



# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "✍️  Signature Verification",
    "🔍  Copy-Move Detection",
    "📄  Document Analysis",
])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_upload(uploaded) -> Path:
    """Save a Streamlit UploadedFile to a local temp file and return the path."""
    suffix = Path(uploaded.name).suffix or ".png"
    # Use a local temp_uploads directory to completely avoid Windows AppData Temp locking issues
    save_dir = Path(__file__).parent / "temp_uploads"
    save_dir.mkdir(exist_ok=True)
    
    file_path = save_dir / f"upload_{hash(uploaded.name)}{suffix}"
    
    # Streamlit UploadedFile acts like a file pointer, which might be at EOF if PIL already read it
    uploaded.seek(0)
    
    with open(file_path, "wb") as f:
        f.write(uploaded.read())
        
    return file_path


def _verdict_badge(verdict: str) -> None:
    colours = {"Authentic": "🟢", "Genuine": "🟢", "Suspicious": "🟡", "Forged": "🔴"}
    icon = colours.get(verdict, "⚪")
    st.markdown(f"## {icon} {verdict}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Signature Verification
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Offline Signature Verification")
    st.markdown(
        "Upload a **reference** (enrolled) signature and a **query** (candidate) signature. "
        "The Siamese network compares their embeddings and determines if they match."
    )
    
    col_left, col_right = st.columns([1, 2], gap="large")
    
    with col_left:
        ref_file = st.file_uploader("Reference signature", type=["png", "jpg", "jpeg"], key="sig_ref")
        qry_file = st.file_uploader("Query signature", type=["png", "jpg", "jpeg"], key="sig_qry")
        weights_path = "weights/siamese_best.pt"
        
        st.write("") # Spacing
        run_sig_btn = st.button("🔎 Verify Signatures", disabled=not (ref_file and qry_file), use_container_width=True)

    with col_right:
        if ref_file or qry_file:
            c1, c2 = st.columns(2)
            with c1:
                if ref_file:
                    st.image(ref_file, caption="Reference", use_container_width=True)
            with c2:
                if qry_file:
                    st.image(qry_file, caption="Query", use_container_width=True)

        if run_sig_btn:
            ref_path = _save_upload(ref_file)
            qry_path = _save_upload(qry_file)
            weights = Path(weights_path)

            if not weights.exists():
                st.warning(
                    f"Weights file `{weights_path}` not found. "
                    "Train the model first with:\n"
                    "```\npython -m src.signature.train\n```"
                )
            else:
                with st.spinner("Running Siamese network..."):
                    from src.signature.inference import verify
                    result = verify(ref_path, qry_path, weights=weights)

                _verdict_badge(result["verdict"])
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{result['confidence']:.1%}")
                m2.metric("Cosine Distance", f"{result['distance']:.4f}")
                m3.metric("Match", "Yes ✓" if result["match"] else "No ✗")

    st.divider()
    with st.expander("ℹ️  About the model"):
        st.markdown("""
**Architecture**: Siamese Network with shared EfficientNet-B0 backbone (timm) +
projection head (Linear → BN → ReLU → Dropout → Linear).

**Training**: Contrastive loss (pytorch-metric-learning), AdamW optimiser,
CosineAnnealingLR scheduler. Default 30 epochs on CEDAR-style paired data.

**References**:
- HTCSigNet (Pattern Recognition, 2025) — Hybrid Transformer-Conv signature network
- Multi-Scale CNN-CrossViT (Complex & Intelligent Systems, 2025) — 98.85% on CEDAR
- TransOSV (Pattern Recognition, 2023) — First ViT-based writer-independent verification
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Copy-Move Detection
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Copy-Move Forgery Detection")
    st.markdown(
        "Upload a document image. The detector identifies regions that have been "
        "copied and pasted within the same image using ORB keypoint matching and "
        "RANSAC geometric verification."
    )

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        img_file = st.file_uploader("Document image", type=["png", "jpg", "jpeg", "tiff"], key="cm_img")
        
        st.markdown("#### Tuning")
        ransac_thresh = st.slider("RANSAC Threshold (Stricter matching)", 1.0, 10.0, 5.0, 0.5,
                                  help="Lower values require identical keypoints to match geometric space more strictly.")
        min_matches = st.slider("Minimum Duplicate Features", 5, 50, 10, 1,
                                help="Minimum number of identical keypoints inside a region to be considered a forgery.")
        
        st.write("") # Spacing
        run_cm_btn = st.button("🔎 Detect Copy-Move", type="primary", use_container_width=True, disabled=not img_file)

    with col_right:
        if img_file and run_cm_btn:
            img_pil = Image.open(img_file).convert("RGB")
            img_path = _save_upload(img_file)
            with st.spinner("Running copy-move detector..."):
                from src.copy_move.detector import detect_copy_move
                from src.copy_move.visualizer import overlay_heatmap, annotate_regions

                result = detect_copy_move(
                    img_path, 
                    min_match_count=min_matches, 
                    ransac_threshold=ransac_thresh
                )

            _verdict_badge(result["verdict"])
            m1, m2 = st.columns(2)
            m1.metric("Forgery Score", f"{result['score']:.1%}")
            m2.metric("Detection Method", result["method"])

            st.markdown("#### Detection Results")
            c1, c2 = st.columns(2)
            with c1:
                mask = result["mask"]
                if mask.any():
                    overlay = overlay_heatmap(np.array(img_pil), mask, alpha=0.4)
                    st.image(overlay, caption="Heatmap overlay", use_container_width=True)
                else:
                    st.info("No significant copy-move regions detected.")

            with c2:
                if result["heatmap"] is not None:
                    st.image(result["heatmap"], caption="Photoholmes heatmap", use_container_width=True)
                elif mask.any():
                    annotated = annotate_regions(np.array(img_pil), mask)
                    st.image(annotated, caption="Annotated regions", use_container_width=True)

    st.divider()
    with st.expander("ℹ️  About the detector"):
        st.markdown("""
**Primary**: [PhotoHolmes](https://github.com/photoholmes/photoholmes) (Splicebuster) when installed.

**Fallback**: ORB feature extraction → BFMatcher → RANSAC homography estimation.
Inlier ratio determines the forgery confidence score.

**References**:
- CMFDFormer (arXiv 2311.13263, 2023): MiT transformer backbone for CMFD
- PhotoHolmes (arXiv 2412.14969, Springer 2025): unified forensics library
- MVSS-Net++ (T-PAMI): multi-view multi-scale supervision
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Document Analysis (ELA + Edge + OCR + Wavelet)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Document Analysis")
    st.markdown("Upload a document to run Error Level Analysis, edge detection, OCR, and wavelet decomposition.")

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        doc_file = st.file_uploader("Document image", type=["png", "jpg", "jpeg", "tiff", "bmp"], key="doc_img")
        
        st.markdown("#### Analysis Setup")
        analysis_options = st.multiselect(
            "Select analyses to run",
            ["Error Level Analysis (ELA)", "Edge Detection", "OCR", "Wavelet Decomposition"],
            default=["Error Level Analysis (ELA)", "Edge Detection"],
            label_visibility="collapsed"
        )

        st.markdown("#### Tooling Parameters")
        st.markdown("**ELA**")
        ela_quality = st.slider("JPEG Resave Quality", 50, 100, 95, 1)
        ela_scale = st.slider("Error Multiplier", 1, 50, 15, 1)
        apply_heatmap = st.checkbox("Apply Jet Heatmap to ELA", value=True)
        mark_suspicious = st.checkbox("Highlight Suspicious Regions", value=True, help="Draws a box around areas with unusually high ELA intensity.")
        
        st.markdown("**Edge Detection**")
        pre_blur = st.slider("Pre-processing Gaussian Blur", 0, 15, 3, 1)
        detector_map = {
            "Canny (Best for sharp edges)": "canny",
            "Sobel (Best for gradients)": "sobel",
            "Laplacian (Best for fine details)": "laplacian",
            "Prewitt X (Horizontal edges)": "prewitt_x",
            "Prewitt Y (Vertical edges)": "prewitt_y"
        }
        detector_display = st.selectbox("Detection Algorithm", list(detector_map.keys()), key="edge_det")
        detector = detector_map[detector_display]
        
        st.markdown("**OCR Extraction**")
        handwritten = st.toggle("Handwritten text mode (uses TrOCR)", value=False, key="ocr_hw")
        
        st.write("") # Spacing
        run_doc_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True, disabled=not doc_file)

    with col_right:
        if doc_file and run_doc_btn:
            doc_pil = Image.open(doc_file).convert("RGB")
            doc_path = _save_upload(doc_file)

            # ── ELA ───────────────────────────────────────────────────────────
            if "Error Level Analysis (ELA)" in analysis_options:
                st.subheader("Error Level Analysis")
                with st.spinner("Generating ELA map..."):
                    from src.analysis.ela import generate_ela, ela_score
                    ela_img = generate_ela(doc_pil, quality=ela_quality, scale=ela_scale)
                    score = ela_score(ela_img)
                    
                    display_img = ela_img
                    if apply_heatmap or mark_suspicious:
                        import cv2
                        import numpy as np
                        ela_cv = np.array(ela_img.convert("L"))
                        
                        if apply_heatmap:
                            display_cv = cv2.applyColorMap(ela_cv, cv2.COLORMAP_JET)
                        else:
                            display_cv = cv2.cvtColor(np.array(ela_img), cv2.COLOR_RGB2BGR)

                        if mark_suspicious:
                            # 1. Baseline thresholding (Top ~2% of brightness)
                            mean_val = np.mean(ela_cv)
                            std_val = np.std(ela_cv)
                            thresh_val = max(50, min(mean_val + 3.5 * std_val, 240))
                            
                            _, mask = cv2.threshold(ela_cv, thresh_val, 255, cv2.THRESH_BINARY)
                            
                            # 2. Aggressive morphological closing to merge text into blobs, followed by opening to remove noise
                            kernel_size = max(3, int(ela_cv.shape[0] * 0.005))
                            kernel = np.ones((kernel_size, kernel_size), np.uint8)
                            
                            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
                            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                            
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            thickness = max(2, int(display_cv.shape[0] * 0.005))
                            box_color = (0, 0, 255) if apply_heatmap else (255, 0, 0) # Red in BGR 
                            
                            # Dynamic area threshold based on image size (e.g. at least 0.1% of image area)
                            min_area = (ela_cv.shape[0] * ela_cv.shape[1]) * 0.001 
                            
                            for cnt in contours:
                                area = cv2.contourArea(cnt)
                                x, y, w, h = cv2.boundingRect(cnt)
                                
                                # Filter 1: Must be a substantial blob, not speckle noise
                                if area > min_area:
                                    # Filter 2: Ignore long, extremely thin lines (like table borders)
                                    aspect_ratio = float(w) / h
                                    if 0.1 < aspect_ratio < 10.0:
                                        # Filter 3: Density check - box should actually contain anomalous pixels, not just be a huge empty box
                                        roi = mask[y:y+h, x:x+w]
                                        density = cv2.countNonZero(roi) / (w * h)
                                        if density > 0.3:
                                            cv2.rectangle(display_cv, (x, y), (x+w, y+h), box_color, thickness)
                        
                        display_img = Image.fromarray(cv2.cvtColor(display_cv, cv2.COLOR_BGR2RGB))

                c1, c2 = st.columns(2)
                with c1:
                    st.image(doc_pil, caption="Original", use_container_width=True)
                with c2:
                    st.image(display_img, caption="ELA Map", use_container_width=True)

                # The score is heavily diluted by white space. 
                # A single forged stamp pushes a pristine document's global score up past ~0.003
                verdict = "Forged" if score > 0.012 else ("Suspicious" if score > 0.003 else "Authentic")
                _verdict_badge(verdict)
                st.metric("ELA Intensity Score", f"{score:.4f}")
                st.caption(
                    "**How to read:** Bright areas or colored halos in the ELA map indicate regions with different compression levels, "
                    "often caused by digital pasting or text alteration. A completely uniform texture indicates a single, unedited file."
                )

            # ── Edge Detection ─────────────────────────────────────────────────
            if "Edge Detection" in analysis_options:
                st.subheader("Edge Detection")
                with st.spinner("Running edge detection..."):
                    from src.analysis.edge_detection import detect_all
                    
                    # Apply optional blur directly to PIL image before detection
                    process_img = doc_pil
                    if pre_blur > 0:
                        import cv2
                        import numpy as np
                        # Ensure kernel is odd
                        k = pre_blur if pre_blur % 2 == 1 else pre_blur + 1
                        cv_img = cv2.cvtColor(np.array(doc_pil), cv2.COLOR_RGB2BGR)
                        blurred = cv2.GaussianBlur(cv_img, (k, k), 0)
                        process_img = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
                        
                    edges = detect_all(process_img)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(doc_pil, caption="Original", use_container_width=True)
                with c2:
                    st.image(edges[detector], caption=f"{detector.capitalize()} edges", use_container_width=True)

            # ── OCR ───────────────────────────────────────────────────────────
            if "OCR" in analysis_options:
                st.subheader("Optical Character Recognition")
                with st.spinner("Extracting text..."):
                    from src.analysis.ocr import extract_text
                    ocr_result = extract_text(doc_path, handwritten=handwritten)

                st.text_area("Extracted text", ocr_result["full_text"], height=200)
                m1, m2 = st.columns(2)
                m1.metric("Avg. Confidence", f"{ocr_result['avg_confidence']:.1%}")
                m2.metric("Engine", ocr_result["engine"])

                if ocr_result["words"]:
                    with st.expander("Word-level results"):
                        import pandas as pd
                        df = pd.DataFrame([
                            {"Text": w["text"], "Confidence": f"{w['confidence']:.1%}"}
                            for w in ocr_result["words"]
                        ])
                        st.dataframe(df, use_container_width=True)

            # ── Wavelet ────────────────────────────────────────────────────────
            if "Wavelet Decomposition" in analysis_options:
                st.subheader("Wavelet Decomposition")
                col_w, col_l = st.columns(2)
                wavelet = col_w.selectbox("Wavelet", ["haar", "db1", "db4", "sym4"], key="wav_name")
                level = col_l.slider("Decomposition level", 1, 6, 3, key="wav_level")

                with st.spinner("Running wavelet decomposition..."):
                    from src.analysis.wavelet import decompose
                    wav_result = decompose(doc_pil, wavelet=wavelet, level=level)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(doc_pil, caption="Original", use_container_width=True)
                with c2:
                    st.image(wav_result["heatmap"], caption=f"{wavelet} detail heatmap (level {level})", use_container_width=True)

    st.divider()
    with st.expander("ℹ️  ELA quality setting"):
        quality = st.slider(
            "JPEG re-compression quality", min_value=70, max_value=99, value=95,
            help="Lower quality amplifies differences in manipulated regions.",
            key="ela_quality",
        )
