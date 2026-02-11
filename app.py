from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model("skin_model.keras")
print("Model loaded successfully!")

classes = ["acne", "blackheads", "normal", "redness", "whiteheads"]

advice = {
    "acne": {
        "reason": "Clogged pores due to excess oil, dead skin cells, and bacteria.",
        "do": [
            "Use salicylic acid face wash",
            "Wash face twice daily",
            "Use non-comedogenic products"
        ],
        "avoid": [
            "Touching or squeezing pimples",
            "Oily creams",
            "Sleeping with makeup"
        ],
        "meds": [
            "Salicylic acid (1–2%)",
            "Benzoyl peroxide (2.5–5%)"
        ],
        "doctor": "If acne becomes painful, spreads, or does not improve in 3 weeks."
    },

    "whiteheads": {
        "reason": "Blocked pores caused by oil and dead skin buildup.",
        "do": [
            "Use mild exfoliation",
            "Apply salicylic acid",
            "Keep skin clean"
        ],
        "avoid": [
            "Heavy moisturizers",
            "Touching the area",
            "Harsh scrubs"
        ],
        "meds": [
            "Salicylic acid cleanser",
            "Retinoid creams (mild)"
        ],
        "doctor": "If the area becomes swollen or painful."
    },

    "blackheads": {
        "reason": "Open clogged pores exposed to air causing oxidation.",
        "do": [
            "Use pore-cleansing mask",
            "Gentle exfoliation",
            "Use oil-free products"
        ],
        "avoid": [
            "Thick oily creams",
            "Overwashing skin"
        ],
        "meds": [
            "Salicylic acid",
            "Clay masks"
        ],
        "doctor": "If blackheads turn into inflamed acne."
    },

    "redness": {
        "reason": "Skin irritation due to allergies, sensitivity, or harsh products.",
        "do": [
            "Use soothing moisturizer",
            "Apply sunscreen",
            "Use gentle cleanser"
        ],
        "avoid": [
            "Hot water on face",
            "Harsh chemicals",
            "Scrubbing skin"
        ],
        "meds": [
            "Aloe vera gel",
            "Niacinamide serum"
        ],
        "doctor": "If redness is painful, spreading, or persistent."
    },

    "normal": {
        "reason": "Healthy balanced skin condition.",
        "do": [
            "Maintain daily skincare routine",
            "Use sunscreen",
            "Stay hydrated"
        ],
        "avoid": [
            "Skipping cleansing",
            "Using harsh products"
        ],
        "meds": [
            "Light moisturizer",
            "Sunscreen (SPF 30+)"
        ],
        "doctor": "No doctor needed unless irritation appears."
    }
}
advice = {
    "acne": {
        "reason": "Clogged pores due to excess oil, dead skin cells, and bacteria.",
        "do": [
            "Use salicylic acid face wash",
            "Wash face twice daily",
            "Use non-comedogenic products"
        ],
        "avoid": [
            "Touching or squeezing pimples",
            "Oily creams",
            "Sleeping with makeup"
        ],
        "meds": [
            "Salicylic acid (1–2%)",
            "Benzoyl peroxide (2.5–5%)"
        ],
        "doctor": "If acne becomes painful, spreads, or does not improve in 3 weeks."
    },

    "whiteheads": {
        "reason": "Blocked pores caused by oil and dead skin buildup.",
        "do": [
            "Use mild exfoliation",
            "Apply salicylic acid",
            "Keep skin clean"
        ],
        "avoid": [
            "Heavy moisturizers",
            "Touching the area",
            "Harsh scrubs"
        ],
        "meds": [
            "Salicylic acid cleanser",
            "Retinoid creams (mild)"
        ],
        "doctor": "If the area becomes swollen or painful."
    },

    "blackheads": {
        "reason": "Open clogged pores exposed to air causing oxidation.",
        "do": [
            "Use pore-cleansing mask",
            "Gentle exfoliation",
            "Use oil-free products"
        ],
        "avoid": [
            "Thick oily creams",
            "Overwashing skin"
        ],
        "meds": [
            "Salicylic acid",
            "Clay masks"
        ],
        "doctor": "If blackheads turn into inflamed acne."
    },

    "redness": {
        "reason": "Skin irritation due to allergies, sensitivity, or harsh products.",
        "do": [
            "Use soothing moisturizer",
            "Apply sunscreen",
            "Use gentle cleanser"
        ],
        "avoid": [
            "Hot water on face",
            "Harsh chemicals",
            "Scrubbing skin"
        ],
        "meds": [
            "Aloe vera gel",
            "Niacinamide serum"
        ],
        "doctor": "If redness is painful, spreading, or persistent."
    },

    "normal": {
        "reason": "Healthy balanced skin condition.",
        "do": [
            "Maintain daily skincare routine",
            "Use sunscreen",
            "Stay hydrated"
        ],
        "avoid": [
            "Skipping cleansing",
            "Using harsh products"
        ],
        "meds": [
            "Light moisturizer",
            "Sunscreen (SPF 30+)"
        ],
        "doctor": "No doctor needed unless irritation appears."
    }
}


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    class_index = np.argmax(prediction)

    label = classes[class_index]
    confidence = round(prediction[class_index] * 100, 2)

    # create dictionary of all class confidences
    confidences = {
        classes[i]: round(prediction[i] * 100, 2)
        for i in range(len(classes))
    }

    return label, confidence, confidences


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    tip = None
    confidences = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        path = "static/upload.jpg"
        file.save(path)

        image_path = path

        result, confidence, confidences = predict_image(path)
        info = advice[result]

    return render_template(
    "index.html",
    result=result,
    confidence=confidence,
    confidences=confidences,
    image_path=image_path,
    info=info
)

    



if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
