from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("model/dog_emotion_model.keras")

classes = ['angry', 'happy', 'relaxed', 'sad']

translations = {
    "happy": [
        "Human, I demand belly rubs immediately!",
        "Let's play fetch!",
        "Today is the best day ever!",
        "Take me on a walk",
        "Give me a kissy",
        "I wuv you!"
    ],

    "sad": [
        "Nobody played with me today.",
        "Please spend some time with me.",
        "Life is difficult.",
        "I'm just a doggy"
    ],

    "angry": [
        "That toy belongs to me.",
        "Touch my snack and face consequences.",
        "I am not amused."
    ],

    "relaxed": [
        "Life is good.",
        "Wake me after my nap.",
        "Everything is peaceful."
    ]
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(224,224)
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    emotion = classes[np.argmax(prediction)]

    confidence = round(
        np.max(prediction) * 100,
        2
    )

    translation = random.choice(
        translations[emotion]
    )

    paw_score = random.randint(80,100)

    treat_probability = random.randint(60,100)

    return render_template(
        "result.html",
        emotion=emotion.capitalize(),
        confidence=confidence,
        translation=translation,
        image_path=filepath,
        paw_score=paw_score,
        treat_probability=treat_probability
    )


if __name__ == "__main__":
    app.run(debug=True)