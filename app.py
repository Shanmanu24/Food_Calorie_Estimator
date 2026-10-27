import re
import torch
from flask import Flask, request, render_template, jsonify
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from calories import CALORIES

app = Flask(__name__)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

DISHES = list(CALORIES.keys())


def clean(name):
    return re.sub(r"\s*\(.*?\)", "", name)   # "idli (2 pcs)" -> "idli"


PROMPTS = ["a photo of " + clean(d) + ", a food dish" for d in DISHES]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify(error="No image uploaded"), 400

    image = Image.open(file.stream).convert("RGB")
    inputs = processor(text=PROMPTS, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)[0]

    top = probs.topk(3)
    results = [
        {"dish": DISHES[i], "confidence": round(float(p) * 100, 1), "calories": CALORIES[DISHES[i]]}
        for p, i in zip(top.values, top.indices)
    ]
    return jsonify(results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
