import joblib
from flask import Flask, request, jsonify
app = Flask(__name__)
model = joblib.load("model.joblib")
@app.route("/", methods=["GET"])
def home(): return "Phase 3 API is running!"
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json['data']
    prediction = model.predict(data).tolist()
    return jsonify({"prediction": prediction})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)