import sys
import os
import cv2
import numpy as np


def main(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Image not found: {path}")

    os.makedirs("outputs", exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #Thresholding (Otsu)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Edge detection (Canny)
    edges = cv2.Canny(blur, 100, 200)

    #Segmentation με χρώμα (HSV) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 120, 120])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    #Καθαρισμός μάσκας
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()
    for c in contours:
        if cv2.contourArea(c) > 500:  
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, "target", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    cv2.imwrite("outputs/1_gray_blur.png", blur)
    cv2.imwrite("outputs/2_threshold.png", thresh)
    cv2.imwrite("outputs/3_edges.png", edges)
    cv2.imwrite("outputs/4_color_mask.png", mask)
    cv2.imwrite("outputs/5_detection.png", result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python pipeline.py images/simadoura.jpg")
    main(sys.argv[1])
