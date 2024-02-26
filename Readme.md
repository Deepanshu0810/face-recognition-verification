## Face Recognition and Verification

### Stages of the Application
1. Face Detection using MTCNN
2. Calculate Face Embedding using FaceNet
3. Face Verification using cosine distance

### User Guide
#### Pre Requisites
    - Python 3.9 or above
    - pip
    - git bash

#### Environement Setup

1. Clone the repository
    ```bash
    git clone https://github.com/Deepanshu0810/face-recognition-verification.git
    cd face-recognition-verification
    ```
2. Create a virtual environment
    ```bash
    python -m venv env
    env/Scripts/activate
    ```
3. Install the requirements
    ```bash
    pip install -r requirements.txt
    ```

#### Run the Application
1. Run the application
    ```bash
    python app.py
    ```
2. Open the browser and go to the following link
    ```bash
    http://localhost:5000
    ```

#### Add New Faces
1. Go to `new_data` folder
2. Create a new folder with the name of the person
3. Add the images of the person in the folder
4. Run the following command
    ```bash
    cd face-recognition-verification
    python train.py
    ```
5. Restart the application
