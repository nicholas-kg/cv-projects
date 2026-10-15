# CV Basics
A beginner project exploring classical image processing with OpenCV in Python. It detects a colored object (e.g. an orange buoy) using a step-by-step pipeline.
## Overview
 
`pipeline.py` runs the following steps on an input image and saves the result of each one:
 
1. **Filtering**: grayscale conversion and Gaussian blur to reduce noise
2. **Thresholding**: Otsu's method to separate foreground from background
3. **Edge detection**: Canny edge detector
4. **Segmentation**: HSV color masking, cleaned up with morphological opening/closing
5. **Detection**: contour extraction and bounding boxes around the detected object
 ## Execution
 
```bash
python pipeline.py images/simadoura.jpg
```

Results are saved in the `outputs/` folder.


