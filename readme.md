# AI Project - Image Recognition


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)

---

## Introduction

 This project uses the **Parseg model**, which was trained on a Vietnamese dataset. The Parseg model can predict only characters from image input; it is in phase 2 of OCR. Therefore, the input image is the result of a model that can find word positions.

## Features

- **High-accuracy image classification**
- **Multi-object recognition** within a single image
- **User-friendly interface** for easy image upload and processing
- **RESTful API** for integration with other applications

## Technologies Used

- **Programming Language**: Python 3.10+
- **Frameworks**: Torch
- **Libraries**: OpenCV, NumPy

## Installation

### System Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/username/image-recognition-project.git
   cd image-recognition-project

2. **Create and activate a virtual environment**


    ```bash
    python -m venv env
    source env/bin/activate  

3. **Install package**

    ```bash
    pip install -r requirements.txt

### Inference on gooogle colab

Try [run this scoure on google colab](demo_parseg_VietNamese.ipynb)




