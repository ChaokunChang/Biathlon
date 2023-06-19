import numpy as np
import logging

from apxinfer.core.utils import XIPFeatureVec, XIPPredEstimation
from apxinfer.core.feature import fvec_random_sample
from apxinfer.core.model import XIPModel

logging.basicConfig(level=logging.INFO)


class PredictionEstimatorHelper:
    def xip_estimate_error(pred_value: float, preds: np.ndarray, min_conf: float):
        pred_error = np.quantile(np.abs(preds - pred_value), min_conf)
        return pred_error

    def xip_estimate_conf(pred_value: float, preds: np.ndarray, max_error: float):
        pred_conf = np.mean(np.abs(preds - pred_value) <= max_error)
        return pred_conf

    def xip_estimate_conf_relative(pred_value: float, preds: np.ndarray,
                                   max_relative_error: float):
        epsilon = np.finfo(np.float64).eps
        mape = np.abs(preds - pred_value) / np.maximum(np.abs(pred_value), epsilon)
        pred_conf = np.mean(mape <= max_relative_error)
        return pred_conf

    def xip_estimate(pred_type: str, preds: np.ndarray,
                     constraint_type: str,
                     constraint_value: float) -> XIPPredEstimation:
        if pred_type == 'classifier':
            pred_value = np.argmax(np.bincount(preds))
        elif pred_type == 'regressor':
            pred_value = np.mean(preds)
        else:
            raise ValueError(f"Unsupported model type: {pred_type}")
        if constraint_type == 'error':
            pred_error = constraint_value
            pred_conf = PredictionEstimatorHelper.xip_estimate_conf(pred_value, preds, constraint_value)
        elif constraint_type == 'conf':
            pred_error = PredictionEstimatorHelper.xip_estimate_error(pred_value, preds, constraint_value)
            pred_conf = constraint_value
        elif constraint_type == 'relative_error':
            pred_error = np.abs(constraint_value * pred_value)
            pred_conf = PredictionEstimatorHelper.xip_estimate_conf_relative(pred_value, preds, constraint_value)
        return XIPPredEstimation(pred_value=pred_value, pred_error=pred_error, pred_conf=pred_conf, fvec=None)


class XIPPredictionEstimator:
    def __init__(self, constraint_type: str, constraint_value: float) -> None:
        self.constraint_type = constraint_type
        self.constraint_value = constraint_value
        self.logger = logging.getLogger('XIPPredictionEstimator')

    def estimate(self, model: XIPModel, fvec: XIPFeatureVec) -> XIPPredEstimation:
        raise NotImplementedError


class MCPredictionEstimator(XIPPredictionEstimator):
    def __init__(self, constraint_type: str, constraint_value: float,
                 seed: int, n_samples: int = 1000) -> None:
        super().__init__(constraint_type, constraint_value)
        self.seed = seed
        self.n_samples = n_samples

    def estimate(self, model: XIPModel, fvec: XIPFeatureVec) -> XIPPredEstimation:
        preds = model.predict(fvec_random_sample(fvec, self.n_samples, self.seed))
        pred_type = model.model_type
        return PredictionEstimatorHelper.xip_estimate(pred_type, preds, self.constraint_type, self.constraint_value)
