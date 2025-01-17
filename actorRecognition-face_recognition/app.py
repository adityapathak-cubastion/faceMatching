from flask import Flask, request, jsonify
from PIL import Image
import pickle, numpy as np, os, face_recognition

app = Flask(__name__)

with open("C:/Users/AdityaPathak/Desktop/faceMatching_Local/FaceMatching_V0.1/faiss_index.pkl", "rb") as f:
    index, known_face_names = pickle.load(f)

@app.route('/recognise_actor', methods = ['POST'])
def recognise_actor():
    if 'file' not in request.files:
        return jsonify({"Error": "No file provided!"}), 400
 
    file = request.files['file']
    if file.filename == '':
        return jsonify({"Error": "No file selected!"}), 400
    
    temp_path = 'tempImage.jpg'
    file.save(temp_path)
 
    try:
        test_image = face_recognition.load_image_file(temp_path)
        
        test_face_locations = face_recognition.face_locations(test_image)
        test_face_encodings = face_recognition.face_encodings(test_image, test_face_locations)
        
        test_face_encodings = np.array(test_face_encodings, dtype = "float32")
        distances, indices = index.search(test_face_encodings, 1)
        
        if distances[0][0] < 0.55:
            name = known_face_names[indices[0][0]]
 
        return jsonify({"Closest Match": name, "Distance": float(distances[0][0])})
    
    except Exception as e:
        return jsonify({"Error": str(e)}), 500
    
    finally:
        os.remove(temp_path) # Cleaning up temp file
 
if __name__ == '__main__':
    app.run(debug = True)