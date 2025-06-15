# import os
# import cv2
# import re
# import pandas as pd
# import easyocr
# import numpy as np
# from ultralytics import YOLO

# # Load your trained YOLOv8 model
# model = YOLO(r"A:\ml\inpr\ml model\license_plate_detector.pt")  # Replace with your model path

# # Initialize OCR reader
# reader = easyocr.Reader(['en'])

# # Define directories
# input_dir = r"A:\ml\inpr\test"
# output_dir = r"A:\ml\inpr\test\output"
# os.makedirs(output_dir, exist_ok=True)

# # Indian State Codes
# state_codes = {
#     'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam', 'BR': 'Bihar',
#     'CH': 'Chandigarh', 'CT': 'Chhattisgarh', 'DL': 'Delhi', 'GA': 'Goa', 'GJ': 'Gujarat',
#     'HR': 'Haryana', 'HP': 'Himachal Pradesh', 'JH': 'Jharkhand', 'JK': 'Jammu and Kashmir',
#     'KA': 'Karnataka', 'KL': 'Kerala', 'MP': 'Madhya Pradesh', 'MH': 'Maharashtra',
#     'MN': 'Manipur', 'ML': 'Meghalaya', 'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha',
#     'PB': 'Punjab', 'PY': 'Puducherry', 'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu',
#     'TS': 'Telangana', 'TR': 'Tripura', 'UP': 'Uttar Pradesh', 'UK': 'Uttarakhand',
#     'WB': 'West Bengal'
# }

# # Common OCR confusions
# similar_chars = {
#     '0': 'O', 'O': '0', '1': 'I', 'I': '1',
#     '8': 'B', 'B': '8', '5': 'S', 'S': '5',
#     '2': 'Z', 'Z': '2', '6': 'G', 'G': '6'
# }

# # Words to ignore from OCR output
# ignore_words = ['IND', 'INDIA', 'BHARAT']

# # Function to correct OCR character confusion
# def correct_ocr_text(text):
#     return ''.join([similar_chars.get(char.upper(), char.upper()) for char in text])

# # Validate Indian number plate format
# def is_valid_plate(text):
#     text = text.replace(" ", "").replace("-", "")
#     return re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$', text)

# # Enhance image for better OCR results
# def enhance_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     eq = cv2.equalizeHist(gray)
#     filtered = cv2.bilateralFilter(eq, 11, 17, 17)
#     return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

# # Store results
# results_list = []

# # Loop over all images
# for filename in os.listdir(input_dir):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         img_path = os.path.join(input_dir, filename)
#         image = cv2.imread(img_path)
#         enhanced = enhance_image(image.copy())

#         detections = model(enhanced)[0]

#         plate_text, state_name = "", "Unknown"
#         vehicle_box, plate_box = None, None

#         for box in detections.boxes:
#             cls = int(box.cls[0])
#             label = model.names[cls].lower()
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             if label == "numberplate":
#                 plate_box = [x1, y1, x2, y2]
#                 crop = enhanced[y1:y2, x1:x2]
#                 ocr_results = reader.readtext(crop)

#                 for result in ocr_results:
#                     raw = result[1]
#                     if any(word in raw.upper() for word in ignore_words):
#                         continue
#                     cleaned = re.sub(r'[^A-Z0-9]', '', raw.upper())
#                     corrected = correct_ocr_text(cleaned)

#                     if is_valid_plate(corrected):
#                         plate_text = corrected
#                         state_name = state_codes.get(corrected[:2], "Unknown")
#                         break
#                     elif len(corrected) >= 6:
#                         plate_text = corrected  # fallback

#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.8, (0, 255, 0), 2)

#             elif label == "vehicle":
#                 vehicle_box = [x1, y1, x2, y2]
#                 cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

#         # Save annotated image
#         output_path = os.path.join(output_dir, filename)
#         cv2.imwrite(output_path, image)

#         results_list.append({
#             "Image_Name": filename,
#             "Vehicle_Box": vehicle_box,
#             "Plate_Box": plate_box,
#             "Plate_Text": plate_text,
#             "State": state_name
#         })

#         # Print results in terminal
#         print(f"\n🖼️ Image: {filename}")
#         print(f"   ➤ Plate: {plate_text if plate_text else 'Not detected'}")
#         print(f"   ➤ State: {state_name}")

# # Export CSV
# df = pd.DataFrame(results_list)
# df.to_csv("number_plate_results.csv", index=False)

# print("\n✅ All images processed!")
# print("📁 Annotated images saved in:", output_dir)
# print("📄 Results saved to: number_plate_results.csv")


import os
import cv2
import re
import pandas as pd
import easyocr
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"A:\ml\inpr\ml model\license_plate_detector.pt")

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Input/Output
video_path = r"A:\ml\inpr\input_video.mp4"
output_video_path = r"A:\ml\inpr\output_video.avi"
csv_path = r"A:\ml\inpr\number_plate_results.csv"

# Indian State Codes
state_codes = {
    'AP': 'Andhra Pradesh', 'AR': 'Arunachal Pradesh', 'AS': 'Assam', 'BR': 'Bihar',
    'CH': 'Chandigarh', 'CT': 'Chhattisgarh', 'DL': 'Delhi', 'GA': 'Goa', 'GJ': 'Gujarat',
    'HR': 'Haryana', 'HP': 'Himachal Pradesh', 'JH': 'Jharkhand', 'JK': 'Jammu and Kashmir',
    'KA': 'Karnataka', 'KL': 'Kerala', 'MP': 'Madhya Pradesh', 'MH': 'Maharashtra',
    'MN': 'Manipur', 'ML': 'Meghalaya', 'MZ': 'Mizoram', 'NL': 'Nagaland', 'OD': 'Odisha',
    'PB': 'Punjab', 'PY': 'Puducherry', 'RJ': 'Rajasthan', 'SK': 'Sikkim', 'TN': 'Tamil Nadu',
    'TS': 'Telangana', 'TR': 'Tripura', 'UP': 'Uttar Pradesh', 'UK': 'Uttarakhand',
    'WB': 'West Bengal'
}

# Similar OCR Characters
similar_chars = {
    '0': 'O', 'O': '0', '1': 'I', 'I': '1',
    '8': 'B', 'B': '8', '5': 'S', 'S': '5',
    '2': 'Z', 'Z': '2', '6': 'G', 'G': '6'
}

ignore_words = ['IND', 'INDIA', 'BHARAT']

# OCR correction
def correct_ocr_text(text):
    return ''.join([similar_chars.get(char.upper(), char.upper()) for char in text])

# Validate Indian Number Plate
def is_valid_plate(text):
    text = text.replace(" ", "").replace("-", "")
    return re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$', text)

# Enhance image for low light and blur
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# Prepare video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # Process 1 frame per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

results = []
frame_idx = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        enhanced = enhance_image(frame.copy())
        detections = model(enhanced)[0]
        plate_text, state_name = "", "Unknown"
        vehicle_box, plate_box = None, None

        for box in detections.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "numberplate":
                plate_box = [x1, y1, x2, y2]
                crop = enhanced[y1:y2, x1:x2]
                ocr_results = reader.readtext(crop)

                for result in ocr_results:
                    raw = result[1]
                    if any(word in raw.upper() for word in ignore_words):
                        continue
                    cleaned = re.sub(r'[^A-Z0-9]', '', raw.upper())
                    corrected = correct_ocr_text(cleaned)

                    if is_valid_plate(corrected):
                        plate_text = corrected
                        state_name = state_codes.get(corrected[:2], "Unknown")
                        break
                    elif len(corrected) >= 6:
                        plate_text = corrected

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

            elif label == "vehicle":
                vehicle_box = [x1, y1, x2, y2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Store Results
        results.append({
            "Frame": frame_count,
            "Vehicle_Box": vehicle_box,
            "Plate_Box": plate_box,
            "Plate_Text": plate_text,
            "State": state_name
        })

        # Show results in terminal
        print(f"\n🎞️ Frame {frame_count}")
        print(f"   ➤ Plate: {plate_text if plate_text else 'Not detected'}")
        print(f"   ➤ State: {state_name}")

        frame_count += 1

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)

print("\n✅ Video processing completed!")
print("📄 Results saved to:", csv_path)
print("🎥 Annotated video saved to:", output_video_path)
