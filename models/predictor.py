import xgboost as xgb
import os
import json

class LightweightPredictor:
    def __init__(self, model_path, scaler_params_path=None):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        self.scaler_mean = None
        self.scaler_scale = None
        
        if scaler_params_path and os.path.exists(scaler_params_path):
            with open(scaler_params_path, 'r') as f:
                params = json.load(f)
                self.scaler_mean = params.get('mean')
                self.scaler_scale = params.get('scale')

    def predict(self, features):
        # Apply scaling if parameters are available
        if self.scaler_mean is not None and self.scaler_scale is not None:
            features = [(f - m) / s for f, m, s in zip(features, self.scaler_mean, self.scaler_scale)]
        
        # XGBoost prediction
        dmatrix = xgb.DMatrix([features])
        preds = self.model.predict(dmatrix)
        
        # Return 1 for phishing, 0 for safe (threshold 0.5)
        prediction = 1 if preds[0] > 0.5 else 0
        probability = float(preds[0])
        
        # If it's a multi-class model or something else, this might need adjustment
        # based on probabilities. For binary classifier (phishing vs safe):
        return prediction, max(probability, 1 - probability)
