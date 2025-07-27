"""
API wrapper for the Tennis Match Prediction Model.

This Flask application exposes two endpoints:

1. `/health` – simple health‑check endpoint that reports whether the
   underlying machine learning model and encoders were loaded correctly.
2. `/predict` – accepts a JSON payload containing two player names and
   an optional surface type and returns the predicted winner along with
   win probabilities for each player and model metadata.

The model artefacts (trained ensemble, one‑hot encoder and feature
column order) are expected to live in the `models/` directory. They
are loaded once at application startup to avoid the overhead of
reloading them for every request.

The default feature values used here mirror those found in the
Streamlit interface of the original project. Where historical data is
missing (e.g. Elo ratings, recent form or surface win rates),
reasonable defaults are assumed. You can improve predictions by
connecting this API to a database of real player statistics and
calculating the inputs dynamically.

Note: the API deliberately does not depend on Streamlit. When deploying
to a platform like Railway the requirements should include Flask and
Gunicorn, as specified in `requirements.txt` and `Procfile`.
"""

import os
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Attempt to load the model, encoder and feature order on startup. Any
# exceptions are logged and exposed through the `/health` endpoint.
MODEL_PATH = os.path.join("models", "tennis_model.pkl")
ENCODER_PATH = os.path.join("models", "encoder.pkl")
FEATURES_PATH = os.path.join("models", "feature_columns.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
  
    MODEL_LOAD_ERROR = None
except Exception as e:
    model = None
    encoder = None
    feature_columns = None
    MODEL_LOAD_ERROR = str(e)


@app.route("/health", methods=["GET"])
def health_check() -> Dict[str, Any]:
    """Simple health check endpoint.

    Returns a JSON object reporting whether the model artefacts have been
    loaded successfully. If there was an exception during model load
    its message is returned under `error`.
    """
    return jsonify(
        {
            "status": "healthy" if MODEL_LOAD_ERROR is None else "degraded",
            "model_loaded": model is not None,
            "error": MODEL_LOAD_ERROR,
        }
    )


@app.route("/predict", methods=["POST"])
def predict_match() -> Dict[str, Any]:
    """Predicts the outcome of a tennis match.

    The expected JSON payload has the following keys:

        {
          "player1": "Novak Djokovic",
          "player2": "Rafael Nadal",
          "surface": "Clay"
        }

    All keys except `player1` and `player2` are optional. If
    `surface` is omitted the prediction defaults to Hard courts. The
    players' names are echoed back in the response but do not affect
    the numeric features because the public API does not have access to
    historical statistics. To improve accuracy integrate real
    statistics for Elo ratings, recent form, surface win rates etc.

    On success the JSON response contains the predicted winner, the
    probability of each player winning and some metadata about the
    model. On failure (e.g. invalid input or internal error) a 400 or
    500 status code is returned with an `error` field describing the
    problem.
    """
    if model is None:
        return jsonify({"error": "Model not loaded", "detail": MODEL_LOAD_ERROR}), 500

    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {e}"}), 400

    if not data or "player1" not in data or "player2" not in data:
        return jsonify({"error": "Missing required fields 'player1' or 'player2'"}), 400

    player1 = data["player1"]
    player2 = data["player2"]
    surface = data.get("surface", "Hard")

    try:
        features = create_feature_vector(surface)
        # Model expects a 2D array
        proba = model.predict_proba([features])[0]
 
       # In this binary classification the first probability
        # corresponds to the "loser" class and the second to the
        # "winner" class as defined during training. We treat
        # `player1` as the candidate winner.
        player1_win_prob = float(proba[1])
        player2_win_prob = float(proba[0])
        predicted_winner = player1 if player1_win_prob >= player2_win_prob else player2
        confidence = max(player1_win_prob, player2_win_prob)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    response = {
        "player1": player1,
        "player2": player2,
        "surface": surface,
        "predictions": {
            "winner": predicted_winner,
            "player1_win_probability": round(player1_win_prob * 100, 2),
            "player2_win_probability": round(player2_win_prob * 100, 2),
            "confidence": round(confidence * 100, 2),
        },
        "model_info": {
            "type": "Ensemble (RandomForest + GradientBoosting + XGBoost)",
            "accuracy": "87%+",  # based on README information
            "features_used": len(feature_columns) if feature_columns else None,
        },
    }
    return jsonify(response)



def create_feature_vector(surface: str) -> np.ndarray:
    """Constructs a feature vector for prediction.

    The underlying model was trained with 15 numeric features

    (representing elo ratings, ages, heights, recent form, surface win
    rates, head‑to‑head count, tournament wins and days since last match
    for each player) plus one‑hot encoded surface columns. Since this
    API does not have access to historical stats it uses sensible
    defaults for all numeric inputs. These defaults can be tuned to
    reflect professional level players but will not capture individual
    strengths or weaknesses.

    Parameters
    ----------
    surface : str
        The court surface. Must be one of 'Hard', 'Clay', 'Grass' or
        'Carpet'. The value is case‑insensitive.

    Returns
    -------
    numpy.ndarray
        A one‑dimensional array containing the numeric features followed
        by encoded surface features. The order matches the model's
        `feature_columns` so that `model.predict_proba` receives
        correctly aligned inputs.
    """
    # Normalise surface name
    surface_norm = surface.capitalize() if surface else "Hard"
    # Define default numeric values. These values mirror those used in
    # the Streamlit UI when no historical data is available.
    defaults: Dict[str, float] = {
        "winner_elo": 1500.0,
        "loser_elo": 1500.0,
        "winner_age": 25.0,
        "winner_ht": 180.0,
        "loser_age": 25.0,
        "loser_ht": 180.0,
        "winner_recent_form": 1500.0,
        "loser_recent_form": 1500.0,
        "winner_surface_win_rate": 0.6,
        "loser_surface_win_rate": 0.6,
        "h2h_matches": 0.0,
        "winner_tourney_wins": 0.0,
        "loser_tourney_wins": 0.0,
        "winner_days_since_last_match": 7.0,
        "loser_days_since_last_match": 7.0,
    }

    # Create a DataFrame with one row for numeric values and the surface
    numeric_df = pd.DataFrame({k: [v] for k, v in defaults.items()})
    surface_df = pd.DataFrame({"surface": [surface_norm]})

    # One‑hot encode surface. The encoder returns a numpy array with one
    # column per category. If the provided surface is unseen, the
    # encoder will produce an all‑zero vector (most encoders have
    # handle_unknown='ignore').
    if encoder is None:
        raise RuntimeError("Encoder not loaded")
    encoded_surface = encoder.transform(surface_df)[0]
    # Assemble full feature vector in the order expected by the model.
    # We build a mapping from feature name to value then iterate over
    # `feature_columns` to preserve order.
    feature_map: Dict[str, float] = {**defaults}
    # Determine the names of encoded surface columns using the encoder
    try:
        surface_feature_names = list(encoder.get_feature_names_out(["surface"]))
    except AttributeError:
        # Older versions of scikit‑learn store categories_ instead
        categories = getattr(encoder, "categories_", [[]])[0]
        surface_feature_names = [f"surface_{c}" for c in categories]
    # Map encoded values to their feature names
    for name, value in zip(surface_feature_names, encoded_surface):
        feature_map[name] = float(value)

    # If feature_columns is available use it to order the vector; otherwise
    # fall back to a sorted order of keys to ensure reproducibility.
    ordered_columns = feature_columns if feature_columns is not None else sorted(feature_map.keys())
    vector = [feature_map.get(col, 0.0) for col in ordered_columns]
    return np.array(vector, dtype=float)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
