import os
import json
import re
import joblib
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from data_utils import get_past_cases, get_law_provisions  # Your existing helpers

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama3-70b-8192")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# -----------------------
# ML Model Path
# -----------------------
MODEL_PATH = Path("models/legal_classifier.pkl").resolve()

def predict_with_ml(scenario_text: str):
    """
    Predict the likely judgment using the trained ML model.
    Returns:
        dict: {"predicted_decision": str, "confidence": float}
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}. Train it first.")

    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"❌ Error loading model: {e}")

    try:
        pred = pipeline.predict([scenario_text])[0]
        probs = pipeline.predict_proba([scenario_text])[0]
        confidence = round(float(max(probs)) * 100, 2)
    except Exception as e:
        raise RuntimeError(f"❌ Error during prediction: {e}")

    return {"predicted_decision": str(pred), "confidence": confidence}

# -----------------------
# Groq Prompt & Parsing
# -----------------------
def build_prompt(scenario, ml_result, num_cases=5, num_laws=5):
    """Build prompt for Groq API using scenario, ML output, past cases, and laws."""
    cases = get_past_cases(num_cases)
    laws = get_law_provisions(num_laws)

    case_text = "\n".join(
        [f"- Case: {c['case_title']} | Issues: {c['issues']} | Decision: {c['decision']}"
         for c in cases if c.get('case_title')]
    )
    law_text = "\n".join(
        [f"- Article {l['article']}: {l['title']} — {l['description']}"
         for l in laws if l.get('article')]
    )

    prompt = f"""
You are a legal analyst. You have access to:
1. ML model prediction
2. Past cases
3. Indian legal provisions

ML Model Output:
- Predicted Decision: {ml_result['predicted_decision']}
- Confidence: {ml_result['confidence']}%

Past Cases:
{case_text}

Relevant Indian Laws:
{law_text}

Scenario:
{scenario}

TASK:
Write a judgment prediction in STRICT JSON format:
{{
  "summary": "<Brief summary of the scenario>",
  "prediction": "<Final judgment outcome in legal language>",
  "confidence": <integer between 0 and 100>,
  "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"]
}}
"""
    return prompt.strip()

def parse_response(text):
    """Safely parse JSON output from LLM."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {
        "summary": "Unable to parse response.",
        "prediction": text[:200],
        "confidence": 0,
        "reasons": []
    }

# -----------------------
# Hybrid ML + Groq Prediction
# -----------------------
def predict_judgment(scenario):
    """
    Hybrid prediction:
    1. Get ML output
    2. Build prompt for Groq API
    3. Generate human-like reasoning dynamically
    4. Return structured JSON
    """
    # Step 1: ML Prediction
    ml_result = predict_with_ml(scenario)

    # Step 2: Build prompt
    prompt = build_prompt(scenario, ml_result)

    # Step 3: Call Groq API
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        raw_output = response.choices[0].message.content
        llm_result = parse_response(raw_output)
    except Exception as e:
        # Fallback if Groq API fails
        llm_result = {
            "summary": f"Based on the scenario: '{scenario[:100]}...', the predicted judgment is '{ml_result['predicted_decision']}'.",
            "prediction": ml_result['predicted_decision'],
            "confidence": ml_result['confidence'],
            "reasons": [
                f"The ML model predicted '{ml_result['predicted_decision']}' with confidence {ml_result['confidence']}%.",
                "Reasoning could not be generated via Groq API."
            ]
        }

    # Step 4: Ensure confidence exists
    if not llm_result.get("confidence"):
        llm_result["confidence"] = ml_result["confidence"]

    return llm_result

# -----------------------
# Example Usage
# -----------------------
if __name__ == "__main__":
    scenario_text = "A citizen was detained without trial for 6 months under preventive detention laws."
    result = predict_judgment(scenario_text)
    print(json.dumps(result, indent=2))
