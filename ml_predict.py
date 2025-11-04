import joblib
from pathlib import Path

# Path to your trained ML model
MODEL_PATH = Path("models/legal_classifier.pkl").resolve()

def predict_with_ml(scenario_text: str):
    """
    Predicts a legal judgment using the trained ML model.

    Args:
        scenario_text (str): Case description or scenario text.

    Returns:
        dict: {
            "predicted_decision": str,
            "confidence": float
        }
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"❌ Model not found at {MODEL_PATH}. Train it first using train_model.py."
        )

    # Load the trained ML pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")

    # Predict the outcome
    try:
        pred = pipeline.predict([scenario_text])[0]
        probs = pipeline.predict_proba([scenario_text])[0]
        confidence = round(float(max(probs)) * 100, 2)
    except Exception as e:
        raise RuntimeError(f"❌ Prediction failed: {e}")

    return {
        "predicted_decision": str(pred),
        "confidence": confidence
    }


if __name__ == "__main__":
    # Test run
    scenario = "A citizen was detained without trial for 6 months under preventive detention laws."
    try:
        result = predict_with_ml(scenario)
        print(f"✅ Prediction: {result['predicted_decision']} (Confidence: {result['confidence']}%)")
    except Exception as e:
        print(e)
