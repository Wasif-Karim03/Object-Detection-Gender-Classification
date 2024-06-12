# Object-Detection-Gender-Classification

Here's a sample `README.md` file for your project:

```markdown
# Real-Time Object Detection and Gender Classification

This project uses YOLOv5 for real-time object detection and a pre-trained gender classification model to identify the gender of detected faces using a laptop's camera.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Model Details](#model-details)
- [Credits](#credits)

## Installation

1. **Clone the repository**:

    ```sh
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the required packages**:

    ```sh
    pip3 install torch torchvision torchaudio
    pip3 install opencv-python opencv-python-headless
    pip3 install numpy certifi
    ```

3. **Download the pre-trained gender classification models**:

    - [deploy_gender.prototxt](https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net/deploy_gender.prototxt)
    - [gender_net.caffemodel](https://www.dropbox.com/s/0a6kehyvzb27e9d/gender_net.caffemodel?dl=1)

    Place these files in the project directory.

## Usage

Run the script to start real-time object detection and gender classification using your laptop's camera:

```sh
python3 main.py
```

Press `q` to exit the application.

## Files

- `main.py`: The main script that performs object detection and gender classification.
- `deploy_gender.prototxt`: The deploy file for the gender classification model.
- `gender_net.caffemodel`: The pre-trained gender classification model.

## Model Details

### YOLOv5

- **Model**: `yolov5x`
- **Framework**: PyTorch
- **Usage**: General object detection
- **Source**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)

### Gender Classification Model

- **Model**: Gender classification using Caffe framework
- **Deploy Prototxt**: `deploy_gender.prototxt`
- **Caffe Model**: `gender_net.caffemodel`
- **Usage**: Gender classification of detected faces
- **Source**: [Adience Gender and Age Classification](https://github.com/GilLevi/AgeGenderDeepLearning)

## Credits

- **Ultralytics**: For the YOLOv5 model
- **Gil Levi and Tal Hassner**: For the gender classification model
- **OpenCV**: For the DNN module and face detection
- **PyTorch**: For the deep learning framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

### Instructions

1. **Clone the repository**: Make sure to replace `https://github.com/your-username/your-repo-name.git` with the actual URL of your GitHub repository.
2. **Model Downloads**: Include the links for downloading the `deploy_gender.prototxt` and `gender_net.caffemodel` files in the `Model Details` section.
3. **Credits**: Ensure all sources and contributors are properly credited.

This README file provides a clear overview of your project, including installation instructions, usage, and details about the models used. It will help others understand and use your project effectively.
