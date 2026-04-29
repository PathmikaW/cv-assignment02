import cv2
import numpy as np
import os

os.makedirs('output/q2', exist_ok=True)

# Camera parameters
f_mm = 8.0
pixel_mm = 2.2e-3   # 2.2 um converted to mm
do_mm = 720.0       # object distance (lens to earring plane)

# Lens formula: 1/f = 1/do + 1/di
di_mm = (f_mm * do_mm) / (do_mm - f_mm)

# Lateral magnification m = di / do
# One pixel on the sensor represents (pixel_mm / m) mm in object space
m = di_mm / do_mm
scale_mm_per_px = pixel_mm / m

print(f"f={f_mm} mm,  do={do_mm} mm,  di={di_mm:.4f} mm")
print(f"Magnification m = {m:.6f}")
print(f"Scale = {scale_mm_per_px:.4f} mm/px")

# Detect earrings via thresholding on the white background
img = cv2.imread('a2_images/earrings.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # left to right

annotated = img.copy()
results = []

print("\nEarring measurements:")
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    # Use the average of width and height as the outer diameter in pixels
    diameter_px = (w + h) / 2.0
    diameter_mm = diameter_px * scale_mm_per_px
    results.append((diameter_px, diameter_mm))

    cx, cy = x + w // 2, y + h // 2
    label = f"Earring {i+1}: {diameter_mm:.1f} mm"
    print(f"  {label}  ({diameter_px:.0f} px diameter)")

    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 220), 2)
    cv2.putText(annotated, label, (x, y + h + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 180), 2)

cv2.imwrite('output/q2/q2_earring_size.png', annotated)
print("Saved: output/q2/q2_earring_size.png")
