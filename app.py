from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

app = Flask(__name__)

model = load_model("leaf_model.h5")

with open("class_names.json", "r") as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for name, index in class_indices.items():
    if 0 <= index < len(class_indices):
        class_names[index] = name

# Fallback if ordering is not preserved correctly
class_names = [name for name in class_names if name is not None]

disease_info = {

    "Pepper__bell___Bacterial_spot": {
        "tanaman": "Paprika",
        "penyakit": "Bacterial Spot",
        "solusi": "Gunakan bibit sehat dan semprot bakterisida."
    },

    "Pepper__bell___healthy": {
        "tanaman": "Paprika",
        "penyakit": "Sehat",
        "solusi": "Tidak diperlukan penanganan khusus."
    },

    "Potato___Early_blight": {
        "tanaman": "Kentang",
        "penyakit": "Early Blight",
        "solusi": "Gunakan fungisida dan buang daun yang terinfeksi."
    },

    "Potato___Late_blight": {
        "tanaman": "Kentang",
        "penyakit": "Late Blight",
        "solusi": "Semprot fungisida dan hindari kelembaban berlebih."
    },

    "Potato___healthy": {
        "tanaman": "Kentang",
        "penyakit": "Sehat",
        "solusi": "Pertahankan perawatan rutin."
    },

    "Tomato_Bacterial_spot": {
        "tanaman": "Tomat",
        "penyakit": "Bacterial Spot",
        "solusi": "Gunakan bakterisida dan buang bagian yang terinfeksi."
    },

    "Tomato_Early_blight": {
        "tanaman": "Tomat",
        "penyakit": "Early Blight",
        "solusi": "Gunakan fungisida dan lakukan rotasi tanaman."
    },

    "Tomato_Late_blight": {
        "tanaman": "Tomat",
        "penyakit": "Late Blight",
        "solusi": "Semprot fungisida dan kurangi kelembaban."
    },

    "Tomato_Leaf_Mold": {
        "tanaman": "Tomat",
        "penyakit": "Leaf Mold",
        "solusi": "Perbaiki sirkulasi udara dan gunakan fungisida."
    },

    "Tomato_Septoria_leaf_spot": {
        "tanaman": "Tomat",
        "penyakit": "Septoria Leaf Spot",
        "solusi": "Buang daun terinfeksi dan gunakan fungisida."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "tanaman": "Tomat",
        "penyakit": "Spider Mites",
        "solusi": "Gunakan insektisida atau predator alami."
    },

    "Tomato__Target_Spot": {
        "tanaman": "Tomat",
        "penyakit": "Target Spot",
        "solusi": "Gunakan fungisida dan jaga kebersihan lahan."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "tanaman": "Tomat",
        "penyakit": "Yellow Leaf Curl Virus",
        "solusi": "Kendalikan kutu kebul dan gunakan bibit sehat."
    },

    "Tomato__Tomato_mosaic_virus": {
        "tanaman": "Tomat",
        "penyakit": "Tomato Mosaic Virus",
        "solusi": "Cabut tanaman terinfeksi dan sterilkan alat."
    },

    "Tomato_healthy": {
        "tanaman": "Tomat",
        "penyakit": "Sehat",
        "solusi": "Tidak diperlukan penanganan khusus."
    }
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    upload_folder = "static"
    filepath = os.path.join(upload_folder, file.filename)

    file.save(filepath)

    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    confidence = float(np.max(prediction)) * 100

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    print("=" * 50)
    print("Prediksi :", predicted_class)
    print("Confidence :", round(confidence, 2))
    print("=" * 50)

    if os.path.exists(filepath):
        os.remove(filepath)

    result = disease_info[predicted_class]

    return jsonify({
        "status": "success",
        "tanaman": result["tanaman"],
        "penyakit": result["penyakit"],
        "solusi": result["solusi"],
        "confidence": round(confidence, 2)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)