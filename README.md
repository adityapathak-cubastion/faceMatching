# Face Matching

This project houses two implementations of recognizing human faces, particularly those present in the publicly available <a href = "https://www.kaggle.com/datasets/jessicali9530/lfw-dataset">'LFW' dataset</a> (more than 13,000 images; faces recognized with ```facenet-pytorch```) and <a href = "https://www.kaggle.com/datasets/nagasai524/indian-actor-faces-for-face-recognition">'Indian Actor Faces' dataset</a> (more than 5,000 images; faces recognized with ```face_recognition```).

## Technologies Used
- Developed using VS Code, Jupyter Notebook and Google Colab
- Python and it's libraries - Flask, Pytorch, face_recognition, pickle, faiss, Pillow, opencv and more
- ```match_face``` and ```recognise_actor``` APIs tested with Postman
<br>

Potential improvements that I want to implement:
- Implementing a Vector DB, like ```Milvus``` for practical applications like managing attendance
- Using libraries like <a href = "https://github.com/serengil/deepface">```deepface```</a>
- Handling multiple persons in an image
- Adding more information for better face matching!

## Acknowledgements
Thanks to Akshay sir, Abhishek sir, Balesh sir, and Mayank for the constant guidance and support.<br>
Thanks to Cubastion Consulting Pvt. Ltd. for a productive and supportive environment that fosters learning.

<img src = "https://github.com/adityapathak-cubastion/faceMatching/blob/main/actorRecognition-face_recognition/generating_encodings.png">
