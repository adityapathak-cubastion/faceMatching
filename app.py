from flask import Flask, request, jsonify
from PIL import Image
import pickle, torch, numpy as np, os
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from werkzeug.serving import WSGIRequestHandler
WSGIRequestHandler.protocol_version = "HTTP/1.1"

with open("C:/Users/AdityaPathak/Desktop/faceMatching/faiss_model.pkl", "rb") as f:
    index, file_paths = pickle.load(f)

mtcnn = MTCNN(image_size = 160, margin = 0, min_face_size = 40)
facenet = InceptionResnetV1(pretrained = "vggface2").eval()

def preprocessImage(imgPath):
    img = Image.open(imgPath).convert('RGB')
    face = mtcnn(img)
    if face is None:
        raise ValueError(f"No face detected in {imgPath}!")
    return face.unsqueeze(0)

def generateEmbedding(imgPath):
    faceTensor = preprocessImage(imgPath)
    with torch.no_grad():
        embedding = facenet(faceTensor).cpu().numpy()
    return embedding

@app.route('/match_face', methods = ['POST'])
def match_face():
    if 'file' not in request.files:
        return jsonify({"Error": "No file provided!"}), 400
 
    file = request.files['file']
    if file.filename == '':
        return jsonify({"Error": "No file selected!"}), 400
    
    temp_path = 'tempImage.jpg'
    file.save(temp_path)
 
    try:
        input_embedding = generateEmbedding(temp_path).flatten() # Generating embedding for input image
        input_embedding = np.expand_dims(input_embedding, axis = 0)
 
        distances, indices = index.search(input_embedding, k = 1) # Searching for closest match in FAISS index
        closest_match_path = file_paths[indices[0][0]]
        distance = float(distances[0][0])
 
        return jsonify({"Closest Match": closest_match_path, "Distance": distance})
    
    except Exception as e:
        return jsonify({"Error": str(e)}), 500
    
    finally:
        os.remove(temp_path) # Cleaning up temp file
 
if __name__ == '__main__':
    app.run(debug = True)