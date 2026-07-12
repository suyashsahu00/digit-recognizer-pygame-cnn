# ✍️ Handwritten Digit Recognizer (CNN + Live Drawing Canvas)

A deep learning project that trains a **Convolutional Neural Network on MNIST** and lets you draw a digit and get an instant prediction — available in two interfaces: a desktop Pygame app and a browser-based Streamlit app.

> **Scope:** Recognizes single handwritten **digits (0–9)** drawn one at a time. This is a live MNIST classifier, not a full OCR engine.

🌐 **Live Demo:** [digit-recognizer-pygame-cnn.streamlit.app](https://digit-recognizer-pygame-cnn.streamlit.app/)

---

## 🎥 Demo

![demo](assets/demo-screenshot.png)

---

## 🧠 How It Works

### 1. Model Training (`MNIST.ipynb`)

- **Dataset:** `keras.datasets.mnist` — 60,000 training images + 10,000 test images, 28×28 grayscale
- **Preprocessing:** Normalized to `[0,1]`, channel axis added, labels one-hot encoded
- **Architecture:**

  | Layer        | Details                                               |
  | ------------ | ----------------------------------------------------- |
  | Conv2D       | 32 filters, 3×3 kernel, ReLU, input (28, 28, 1)      |
  | MaxPooling2D | 2×2                                                   |
  | Conv2D       | 64 filters, 3×3 kernel, ReLU                          |
  | MaxPooling2D | 2×2                                                   |
  | Flatten      | —                                                     |
  | Dropout      | 0.25                                                  |
  | Dense        | 10 units, Softmax                                     |

- **Optimizer:** Adam · **Loss:** Categorical cross-entropy
- **Callbacks:** `EarlyStopping` (patience=4, monitors val_accuracy) + `ModelCheckpoint` (saves best as `bestmodel.h5`)
- **Result:** **99.13% test accuracy**, 0.0513 test loss

---

### 2. Pygame Desktop App (`app.py`)

- Opens a **640×480** black drawing canvas
- Draw a digit with your mouse (left-click and drag)
- On **mouse release**, the app:
  - Computes a bounding box around the stroke
  - Pads the crop to a **square** (preserving aspect ratio)
  - Adds a border margin, resizes to **28×28**, normalizes
  - Feeds into the CNN → shows the predicted word label (e.g. "Seven") in red, next to your drawing
- Press **`N`** to clear the canvas and draw the next digit

### 3. Streamlit Browser App (`streamlit_app.py`)

- Wide-layout browser interface, no installation of Pygame required
- **900×300** freedraw canvas (white strokes, black background)
- On each draw, the app:
  - Crops to the bounding box of drawn pixels (15px padding)
  - Pads the crop to a **square** before resizing to 28×28
  - Normalizes and feeds into the CNN
  - Shows the **predicted digit (large) and confidence %** in a side panel
- Click **🗑️ Clear Canvas** before drawing the next digit — the model only works correctly on one digit at a time
- Model is cached with `@st.cache_resource` (loads once per session)

---

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| Python 3.11 | Runtime |
| TensorFlow / Keras | Model training & inference |
| NumPy | Array manipulation |
| OpenCV (`cv2`) | Image resizing & preprocessing |
| Pygame | Desktop drawing canvas |
| Streamlit | Browser-based web app |
| streamlit-drawable-canvas | Interactive drawing widget |

---

## 📂 Project Structure

```
digit-recognizer-pygame-cnn/
├── MNIST.ipynb               # Notebook: data prep, CNN architecture, training
├── app.py                    # Desktop app (Pygame): draw + real-time prediction
├── streamlit_app.py          # Browser app (Streamlit): draw + real-time prediction
├── bestmodel.h5              # Saved trained CNN weights (~440 KB)
├── requirements.txt          # Local install (includes pygame)
├── requirements-cloud.txt    # Streamlit Cloud install (excludes pygame)
├── .python-version           # Pins Python 3.11 for Streamlit Cloud
├── assets/
│   └── demo-screenshot.png   # Demo image shown in README
└── LICENSE                   # MIT
```

---

## ⚙️ Installation & Usage

### Local Setup

```bash
# Clone the repo
git clone https://github.com/suyashsahu00/digit-recognizer-pygame-cnn.git
cd digit-recognizer-pygame-cnn

# Install all dependencies (includes pygame for the desktop app)
pip install -r requirements.txt
```

### Option A — Desktop App (Pygame)

```bash
python app.py
```

**Controls:**
- **Left-click + drag** to draw a digit
- **Release mouse** → prediction appears next to your drawing
- **`N` key** → clear the canvas for the next digit
- Close the window to quit

### Option B — Browser App (Streamlit)

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

**Controls:**
- Draw one digit on the black canvas
- Prediction and confidence % appear instantly in the right panel
- Click **🗑️ Clear Canvas** before drawing the next digit

> **Important:** This is a single-digit classifier. Always clear between digits — drawing multiple digits at once will produce incorrect predictions.

### Streamlit Cloud Deployment

This app is deployed at [digit-recognizer-pygame-cnn.streamlit.app](https://digit-recognizer-pygame-cnn.streamlit.app/).

- Uses `requirements-cloud.txt` (excludes `pygame`, which isn't available on headless cloud environments)
- Uses `.python-version` to pin **Python 3.11** (TensorFlow doesn't yet support Python 3.13+)

---

## 📊 Results

| Metric           | Value  |
| ---------------- | ------ |
| Test Accuracy    | 99.13% |
| Test Loss        | 0.0513 |
| Total Parameters | 34,826 |
| Model Size       | ~440 KB |

---

## 📄 License

MIT — see [LICENSE](LICENSE).
