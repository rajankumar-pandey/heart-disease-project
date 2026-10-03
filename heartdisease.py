from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
	"Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
	"Sex_M", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
	"RestingECG_Normal", "RestingECG_ST", "ExerciseAngina_Y",
	"ST_Slope_Flat", "ST_Slope_Up",
]
INPUT_FIELDS = [
	"Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
	"RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]
CHOICES = {
	"Sex": {"M", "F"},
	"ChestPainType": {"ATA", "NAP", "TA", "ASY"},
	"RestingECG": {"Normal", "ST", "LVH"},
	"ExerciseAngina": {"Y", "N"},
	"ST_Slope": {"Up", "Flat", "Down"},
}


def load_pickle(filename):
	with (BASE_DIR / filename).open("rb") as file:
		return joblib.load(file)


def model_row(values):
	return [
		float(values["Age"]),
		float(values["RestingBP"]),
		float(values["Cholesterol"]),
		float(values["FastingBS"]),
		float(values["MaxHR"]),
		float(values["Oldpeak"]),
		int(values["Sex"] == "M"),
		int(values["ChestPainType"] == "ATA"),
		int(values["ChestPainType"] == "NAP"),
		int(values["ChestPainType"] == "TA"),
		int(values["RestingECG"] == "Normal"),
		int(values["RestingECG"] == "ST"),
		int(values["ExerciseAngina"] == "Y"),
		int(values["ST_Slope"] == "Flat"),
		int(values["ST_Slope"] == "Up"),
	]


app = Flask(__name__)
model = None
scaler = None
load_error = None

try:
	saved_features = load_pickle("columns.pkl")
	if list(saved_features) != FEATURES:
		raise ValueError("columns.pkl does not match the expected feature order")
	model = load_pickle("LogisticRegression_heart.pkl")
	scaler = load_pickle("scaler.pkl")
except (ImportError, ModuleNotFoundError, OSError, ValueError) as error:
	load_error = str(error)


@app.get("/")
def index():
	return render_template("index.html")


@app.get("/health")
def health():
	if load_error or model is None or scaler is None:
		return jsonify(status="error", error=load_error or "Model is unavailable"), 503
	return jsonify(status="ok", message="Prediction service is ready")


@app.post("/predict")
def predict():
	if load_error or model is None or scaler is None:
		return jsonify(error=f"The prediction model is unavailable: {load_error}"), 503

	values = request.get_json(silent=True) or {}
	missing = [field for field in INPUT_FIELDS if field not in values]
	if missing:
		return jsonify(error=f"Missing fields: {', '.join(missing)}"), 400
	invalid_choices = [
		f"{field} must be one of {', '.join(sorted(choices))}"
		for field, choices in CHOICES.items()
		if values[field] not in choices
	]
	if invalid_choices:
		return jsonify(error="; ".join(invalid_choices)), 400

	try:
		row = np.array([model_row(values)])
		scaled_row = scaler.transform(row)
		prediction = int(model.predict(scaled_row)[0])
		probability = float(model.predict_proba(scaled_row)[0][1] * 100)
	except (KeyError, TypeError, ValueError) as error:
		return jsonify(error=f"Please enter valid numeric values: {error}"), 400
	except Exception as error:
		return jsonify(error=f"Prediction failed: {error}"), 500

	message = (
		"The model detected a higher heart disease risk signal."
		if prediction
		else "The model detected a lower heart disease risk signal."
	)
	return jsonify(
		message=message,
		probability=probability,
		prediction=prediction,
		report={field: values[field] for field in INPUT_FIELDS},
	)


if __name__ == "__main__":
	app.run(debug=False)
