from flask import Flask, render_template, request, url_for
import joblib

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    data = vectorizer.transform([review])
    prediction = model.predict(data)[0]
    sentiment = "Positif 😊" if prediction == 1 else "Negatif 😠"

    # Gambar yang sudah kamu simpan sebelumnya di folder static
    distribusi_path = url_for('static', filename='distribusi_prediksi.png')
    roc_path = url_for('static', filename='roc_curve.png')
    cm_path = url_for('static', filename='confusion_matrix.png')

    return render_template(
        "index.html",
        result=sentiment,
        review=review,
        distribusi=distribusi_path,
        roc=roc_path,
        cm=cm_path
    )

if __name__ == "__main__":
    app.run(debug=True)
