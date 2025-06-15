> VahanVision:Automatic Indian Numberplate Detection


---
This project uses a **YOLOv8 model** combined with **EasyOCR** to detect vehicles and number plates from **video input**, extract the license text, determine the **Indian state** based on the plate code, and save annotated results.

---

### 📂 Features

* 🔍 Detect vehicles and number plates using YOLOv8
* 🧠 Use EasyOCR with character-correction logic to read plates accurately
* 🧼 Ignore non-relevant text (e.g. "IND", car brand names)
* ⚙️ Enhanced preprocessing for **low-light and blurry images**
* 📌 Match Indian number plate format and extract **state info**
* 📄 Store results in **CSV**
* 🖼️ Save annotated output **video**
* 📟 Print output in the **terminal**

---

### 🛠️ Setup Instructions

1. **Clone the repo / download the code**

2. **Install dependencies**

```bash
pip install ultralytics easyocr opencv-python pandas
```

3. **Prepare your files**

* ✅ Trained YOLOv8 model:
  Place your `license_plate_detector.pt` model in a known path.

* ✅ Input video:
  Update `video_path` in the script to point to your `.mp4` or `.avi` file.

---

### 🧾 Directory Structure

```
project_root/
├── input_video.mp4
├── license_plate_detector.pt
├── output_video.avi         # (Generated)
├── number_plate_results.csv # (Generated)
└── run_plate_recognition.py
```

---

### 🚀 How to Run

Edit the paths in the script as needed and run:

```bash
python run_plate_recognition.py
```

---

### 📌 Output

* Annotated **video** with bounding boxes and plate text

* **CSV file** with:

  * Frame Number
  * Plate Text
  * Vehicle & Plate Box Coordinates
  * Indian State

* **Console Logs**

```bash
🎞️ Frame 0
   ➤ Plate: MH12AB1234
   ➤ State: Maharashtra
```

---

### 🧠 Logic Used

* **Regex validation** for Indian format: `^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$`
* **OCR cleaning** with substitution logic:

  * `'0' ↔ 'O'`, `'1' ↔ 'I'`, `'5' ↔ 'S'`, etc.
* **Noise filtering**: Skips OCR words like "IND", "TATA", "HONDA"
* **Preprocessing**: CLAHE + Gaussian Blur + Sharpening

---

### 🔁 Customization Tips

* 🎥 To process every **N seconds**, modify `frame_interval`
* 🌍 Add more `state_codes` for new RTOs if needed
* 🛠 To run on a **webcam**, use `cv2.VideoCapture(0)`

---

### 🧪 Sample Results

| Frame | Plate Text | State       |
| ----- | ---------- | ----------- |
| 0     | MH12AB1234 | Maharashtra |
| 30    | KA03MN0987 | Karnataka   |

---

### 📧 Contact

Maintained by [Atharva Chinchane](mailto:atharvachinchane10@gmail.com)
