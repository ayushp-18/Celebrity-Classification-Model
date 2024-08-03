# Celebrity Classification Model

This project implements a celebrity classification model using OpenCV and Haar Cascade classifiers for facial and eye detection. The model is designed to recognize and classify images of celebrities.

## Table of Contents
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Tech Stack

- **Python**: The main programming language used for the implementation.
- **Jupyter Notebook**: For interactive code development and visualization.
- **OpenCV**: For image processing and computer vision tasks.
- **Haar Cascades**: For face and eye detection.

## Libraries Used

- `opencv-python`: OpenCV library for image processing.
- `numpy`: For numerical operations.
- `matplotlib`: For plotting and visualization.
- `PIL`: Python Imaging Library for image manipulation.

## Installation

To run this project, you need to have Python installed along with several libraries. You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

Make sure you have the following files in your project directory:
- `Celebrity classification model.ipynb`
- `haarcascade_eye.xml`
- `haarcascade_frontalface_default.xml`

##Dataset

the training and the test images can be found on this drive - https://drive.google.com/drive/folders/1h78wXFg6o6ORo7anw7yeIsyQrIUVwAEN?usp=sharing

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ayushp-18/Celebrity-Classification-Model.git
    cd Celebrity-Classification-Model
    ```

2. **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook:**

    Open `Celebrity classification model.ipynb` using Jupyter Notebook or Jupyter Lab and run all the cells to see the model in action.

## Examples

Below are some examples of how to use the model:

1. **Loading the Haar Cascades:**

    ```python
    import cv2

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    ```

2. **Detecting Faces and Eyes:**

    ```python
    def detect_faces_and_eyes(image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    detect_faces_and_eyes('path_to_image.jpg')
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
