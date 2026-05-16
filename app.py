from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import os

app = Flask(__name__)

# Load YOLO model
try:
    yolo_model = YOLO("best2_small.pt")
except:
    print("Warning: best2_small.pt not found.")
    yolo_model = None

# Global variables for transformer (lazy loaded)
tokenizer = None
translator_model = None
device = None

def load_translator():
    """Lazy load transformer model on first use"""
    global tokenizer, translator_model, device
    
    if translator_model is None:
        print("Loading translator model... (this may take a few minutes)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = M2M100Tokenizer.from_pretrained("mattiadc/hiero-transformer")
        translator_model = M2M100ForConditionalGeneration.from_pretrained("mattiadc/hiero-transformer")
        translator_model = translator_model.to(device)
        tokenizer.src_lang = "ar"
        print("Translator model loaded!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate_api():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Empty filename"
        }), 400

    # Create static folder if missing
    upload_dir = app.static_folder
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded image
    upload_path = os.path.join(upload_dir, file.filename)
    file.save(upload_path)
    image_url = url_for('static', filename=file.filename)

    # Run YOLO detection
    if yolo_model is None:
        return jsonify({
            "success": False,
            "error": "YOLO model not loaded. Please ensure best2.pt is in the project root."
        }), 500
    
    results = yolo_model(upload_path)

    detected_codes = []
    boxes = []

    for r in results:
        for box in r.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            code = yolo_model.names[cls_id]

            detected_codes.append(code)

            boxes.append({
                "code": code,
                "confidence": round(conf * 100)
            })

    # Translate using transformer model
    if detected_codes:
        load_translator()  # Load model on first use
        text = " ".join(detected_codes)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = translator_model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id("en"),
                num_beams=5,
                max_length=128,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        translation = "No glyphs detected"

    return jsonify({
        "success": True,
        "translation": translation,
        "codes": detected_codes,
        "glyph_count": len(detected_codes),
        "rows": [detected_codes],
        "image_url": image_url,
        "overlay_b64": None,
        "boxes": boxes
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
