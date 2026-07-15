# FastAPI Application for Airbnb Models Serving=
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import traceback
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from prediction_pipeline import (
    compute_amenities_count,
    make_prediction,
)

project_root = Path(__file__).parent.parent
models_dir = project_root / "models"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Airbnb Prediction API",
    description="Production-ready API for serving trained Airbnb prediction models",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"FastAPI app initialized. Models dir: {models_dir}")

class ListingInput(BaseModel):
    bedrooms: int = Field(..., ge=0, le=50)
    beds: int = Field(..., ge=0, le=50)
    baths: int = Field(..., ge=0, le=50)
    guests: int = Field(..., ge=0, le=100)
    amenities_count: Optional[int] = Field(None, ge=0)
    photos_count: int = Field(..., ge=0)
    superhost: int = Field(..., ge=0, le=1)
    num_reviews: int = Field(..., ge=0)
    avg_rating: float = Field(..., ge=1, le=5)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    distance_from_city_center: Optional[float] = Field(None, ge=0)
    cancellation_policy: Union[str, float, int] = Field(..., description="Cancellation policy label or encoded numeric value")
    min_nights: int = Field(..., ge=0, le=1000)
    cleaning_fee: float = Field(..., ge=0, le=10000)
    extra_guest_fee: float = Field(..., ge=0, le=1000)
    registration: int = Field(..., ge=0, le=1)
    professional_management: int = Field(..., ge=0, le=1)
    listing_type: str = Field("other", description="Listing type (mapped to training categories)")
    room_type: str = Field("private_room", description="Room type: hotel_room, private_room, shared_room")
    city_name: Optional[str] = Field(None, description="Deprecated for inference; coordinates are used for georesolution")
    city_population: Optional[float] = Field(None, ge=0, description="Optional city population; auto-resolved from geoencoding when available")
    amenities: Optional[Union[str, List[str]]] = Field(
        None,
        description="Amenities as comma-separated string or list of strings",
    )
    ttm_blocked_days: Optional[float] = Field(0.0, ge=0)
    ttm_total_days: Optional[float] = Field(365.0, ge=1)

class PredictionResponse(BaseModel):
    success: bool
    model: str
    target: str
    prediction: float
    prediction_formatted: str
    timestamp: str
    request_id: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    models_directory: str
    timestamp: str

class BatchListingInput(BaseModel):
    listings: List[ListingInput]
    model: str

class BatchPredictionResponse(BaseModel):
    success: bool
    model: str
    total_listings: int
    successful_predictions: int
    failed_predictions: int
    predictions: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    processing_time_seconds: float
    timestamp: str

class CompetitiveAnalysisResponse(BaseModel):
    success: bool
    cluster_id: int
    cluster_size: int
    market_position: str
    competitiveness_score: float
    price_vs_cluster: float
    predicted_price: float
    cluster_median_price: float
    percentile_ranks: Dict[str, float]
    timestamp: str

class ModelCache:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.resolved_model_files = {}
        self.model_file_mtimes = {}
        self.feature_lists = {}
        self.model_registry = None

    def _load_features_from_txt(self, file_path: Path) -> List[str]:
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        features: List[str] = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '. ' in line:
                    _, feature = line.split('. ', 1)
                    features.append(feature.strip())
        if not features:
            raise ValueError(f"No features found in {file_path}")
        return features

    def load_feature_lists(self):
        if self.feature_lists:
            return self.feature_lists

        mapping = {
            'model1_price': self.models_dir / 'model1_price_features.txt',
            'model2_revenue': self.models_dir / 'model2_revenue_features.txt',
            'best_ttm_avg_rate': self.models_dir / 'best_ttm_avg_rate_features.txt',
            'best_ttm_revenue': self.models_dir / 'best_ttm_revenue_features.txt',
        }
        for model_key, path in mapping.items():
            if path.exists():
                self.feature_lists[model_key] = self._load_features_from_txt(path)

        logger.info(f"Loaded feature lists: {list(self.feature_lists.keys())}")
        return self.feature_lists

    def load_model_registry(self):
        if self.model_registry is None:
            registry_path = self.models_dir / 'model_registry.json'
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
            else:
                self.model_registry = {}
        return self.model_registry

    def _resolve_model_path(self, model_name: str) -> Path:
        """Resolve a requested model key to an existing model artifact path.
        Tries both .json (XGBoost native) and .pkl (joblib pickle) formats."""
        candidate_files: List[str] = [f"{model_name}.json", f"{model_name}.pkl"]

        if model_name == 'model1_xgb_price':
            registry = self.load_model_registry()
            models = registry.get('models', {}) if isinstance(registry, dict) else {}
            registry_file = models.get('model1_price', {}).get('model_file')
            if registry_file:
                candidate_files.insert(0, registry_file)
            candidate_files.extend(['model1_best_price.pkl', 'model1_best_price.json'])
        elif model_name == 'model2_xgb_revenue':
            registry = self.load_model_registry()
            models = registry.get('models', {}) if isinstance(registry, dict) else {}
            registry_file = models.get('model2_revenue', {}).get('model_file')
            if registry_file:
                candidate_files.insert(0, registry_file)
        elif model_name == 'best_ttm_avg_rate_model':
            candidate_files.insert(0, 'best_ttm_avg_rate_model.json')
            candidate_files.insert(1, 'best_ttm_avg_rate_model.pkl')
        elif model_name == 'best_ttm_revenue_model':
            candidate_files.insert(0, 'best_ttm_revenue_model.pkl')
            candidate_files.insert(1, 'best_ttm_revenue_model.json')

        seen = set()
        for file_name in candidate_files:
            if not file_name or file_name in seen:
                continue
            seen.add(file_name)
            model_path = self.models_dir / file_name
            if model_path.exists():
                return model_path

        raise FileNotFoundError(
            f"Model file not found for '{model_name}'. Tried: {candidate_files}"
        )
    
    def load_model(self, model_name: str):
        """Load model from cache, supporting both XGBoost JSON and joblib pickle formats.
        
        Handles:
        - XGBoost JSON (.json): Native XGBoost format (version-compatible)
        - Joblib pickle (.pkl): Serialized sklearn/ensemble models
        """
        model_file = self._resolve_model_path(model_name)
        model_mtime = model_file.stat().st_mtime
        cached_file = self.resolved_model_files.get(model_name)
        cached_mtime = self.model_file_mtimes.get(model_name)

        if model_name not in self.models or cached_file != model_file.name or cached_mtime != model_mtime:
            logger.info(f"DEBUG: Loading/reloading model from: {model_file} (suffix: {model_file.suffix})")

            try:
                if model_file.suffix == '.json':
                    from xgboost import Booster
                    logger.info(f"DEBUG: Loading as XGBoost Booster (JSON)")
                    booster = Booster()
                    booster.load_model(str(model_file))
                    self.models[model_name] = booster
                    logger.info(f"✓ Loaded XGBoost model (JSON format): {model_name} from {model_file.name}")
                else:
                    logger.info(f"DEBUG: Loading as joblib pickle")
                    self.models[model_name] = joblib.load(model_file)
                    logger.info(f"✓ Loaded joblib model (pickle format): {model_name} from {model_file.name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name} from {model_file}: {e}")
                raise

            self.resolved_model_files[model_name] = model_file.name
            self.model_file_mtimes[model_name] = model_mtime
        
        return self.models[model_name]

    def get_loaded_model_filename(self, model_name: str) -> Optional[str]:
        return self.resolved_model_files.get(model_name)
    
    def get_features(self, model_key: str):
        feature_lists = self.load_feature_lists()
        if model_key in feature_lists:
            return feature_lists[model_key]
        raise ValueError(f"Features not found for model: {model_key}")

    def get_target_transform(self, model_key: str, model_name: Optional[str] = None) -> str:
        if model_name:
            loaded_file = self.get_loaded_model_filename(model_name)
            if loaded_file in {'model1_best_price.pkl'}:
                return 'none'

        registry = self.load_model_registry()
        models = registry.get('models', {}) if isinstance(registry, dict) else {}
        if model_key == 'model1_price':
            model_cfg = models.get('model1_price', {})
            return model_cfg.get('target_transform') or model_cfg.get('metrics', {}).get('target_transform', 'log1p')
        if model_key == 'model2_revenue':
            model_cfg = models.get('model2_revenue', {})
            return model_cfg.get('target_transform') or model_cfg.get('metrics', {}).get('target_transform', 'log1p')
        if model_key == 'best_ttm_avg_rate':
            model_cfg = models.get('best_ttm_avg_rate', {})
            return model_cfg.get('target_transform') or model_cfg.get('metrics', {}).get('target_transform', 'log1p')
        if model_key == 'best_ttm_revenue':
            model_cfg = models.get('best_ttm_revenue', {})
            return model_cfg.get('target_transform') or model_cfg.get('metrics', {}).get('target_transform', 'log1p')
        return 'none'
    
    def verify_all_models_loaded(self):
        return {
            'model1_xgb_price': (
                (self.models_dir / 'model1_xgb_price.pkl').exists()
                or (self.models_dir / 'model1_best_price.pkl').exists()
            ),
            'model2_xgb_revenue': (self.models_dir / 'model2_xgb_revenue.pkl').exists(),
            'model1_price_features': (self.models_dir / 'model1_price_features.txt').exists(),
            'model2_revenue_features': (self.models_dir / 'model2_revenue_features.txt').exists(),
            'model_registry': (self.models_dir / 'model_registry.json').exists(),
            'best_ttm_avg_rate_model': (
                (self.models_dir / 'best_ttm_avg_rate_model.json').exists()
                or (self.models_dir / 'best_ttm_avg_rate_model.pkl').exists()
            ),
            'best_ttm_avg_rate_features': (self.models_dir / 'best_ttm_avg_rate_features.txt').exists(),
            'best_ttm_revenue_model': (
                (self.models_dir / 'best_ttm_revenue_model.json').exists()
                or (self.models_dir / 'best_ttm_revenue_model.pkl').exists()
            ),
            'best_ttm_revenue_features': (self.models_dir / 'best_ttm_revenue_features.txt').exists(),
        }

cache = ModelCache(models_dir)

def _compute_amenities_count(amenities: Optional[Union[str, List[str]]], fallback_count: Optional[int]) -> int:
    return compute_amenities_count(amenities, fallback_count)

@lru_cache(maxsize=1)
def _load_competitive_reference():
    """
    Load competitive reference data: clustered listings and KMeans model.
    
    Returns:
        Tuple of (ref_df: DataFrame with clustered listings, kmeans: KMeans model)
    """
    try:
        kmeans_path = models_dir / "competitive_geo_kmeans.pkl"
        kmeans = joblib.load(kmeans_path)
        
        sample_path = models_dir / "competitive_clustered_listings_sample.csv"
        ref_df = pd.read_csv(sample_path)
        
        if 'amenities_count' not in ref_df.columns and 'total_amenities' in ref_df.columns:
            ref_df['amenities_count'] = ref_df['total_amenities']
        elif 'amenities_count' not in ref_df.columns:
            ref_df['amenities_count'] = 0
        
        logger.info(f"Loaded competitive reference: {len(ref_df)} listings, {len(kmeans.cluster_centers_)} clusters")
        return ref_df, kmeans
        
    except FileNotFoundError as e:
        logger.error(f"Competitive reference files not found: {e}")
        raise ValueError(f"Competitive analysis data unavailable: {e}")
    except Exception as e:
        logger.error(f"Failed to load competitive reference: {e}")
        raise ValueError(f"Failed to load competitive analysis data: {e}")

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": f"Invalid input: {str(exc)}", "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    try:
        models_status = cache.verify_all_models_loaded()
        all_ready = all(models_status.values())
        
        return HealthCheckResponse(
            status="healthy" if all_ready else "degraded",
            models_loaded=models_status,
            models_directory=str(models_dir),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            models_loaded={},
            models_directory=str(models_dir),
            timestamp=datetime.now().isoformat()
        )

@app.post("/api/predict/price", response_model=PredictionResponse)
async def predict_price(listing: ListingInput) -> PredictionResponse:
    try:
        request_id = f"price_{datetime.now().timestamp()}"
        prediction = make_prediction(
            listing_dict=listing.dict(),
            model_name="best_ttm_avg_rate_model",
            feature_key="best_ttm_avg_rate",
            cache=cache,
            request_id=request_id
        )

        return PredictionResponse(
            success=True,
            model="Best Trained Model",
            target="ttm_avg_rate",
            prediction=prediction,
            prediction_formatted=f"${prediction:.2f}/night",
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Price prediction error: {e}")
        raise ValueError(f"Price prediction failed: {str(e)}")

@app.post("/api/predict/revenue", response_model=PredictionResponse)
async def predict_revenue(listing: ListingInput) -> PredictionResponse:
    try:
        request_id = f"revenue_{datetime.now().timestamp()}"
        prediction = make_prediction(
            listing_dict=listing.dict(),
            model_name="best_ttm_revenue_model",
            feature_key="best_ttm_revenue",
            cache=cache,
            request_id=request_id
        )
        
        return PredictionResponse(
            success=True,
            model="Best Revenue Model",
            target="ttm_revenue",
            prediction=prediction,
            prediction_formatted=f"${prediction:.2f}/year",
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Revenue prediction error: {e}")
        raise ValueError(f"Revenue prediction failed: {str(e)}")

@app.post("/api/predict/ttm-avg-rate", response_model=PredictionResponse)
async def predict_ttm_avg_rate(listing: ListingInput) -> PredictionResponse:
    """Predict TTM average rate using best trained model from notebook."""
    try:
        request_id = f"ttm_avg_rate_{datetime.now().timestamp()}"
        prediction = make_prediction(
            listing_dict=listing.dict(),
            model_name="best_ttm_avg_rate_model",
            feature_key="best_ttm_avg_rate",
            cache=cache,
            request_id=request_id
        )
        
        return PredictionResponse(
            success=True,
            model="Best Trained Model",
            target="ttm_avg_rate",
            prediction=prediction,
            prediction_formatted=f"${prediction:.2f}",
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"TTM average rate prediction error: {e}")
        raise ValueError(f"TTM average rate prediction failed: {str(e)}")

@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchListingInput) -> BatchPredictionResponse:
    """Batch prediction endpoint for multiple listings."""
    from time import time
    start_time = time()
    
    try:
        model_map = {
            'price': ('model1_xgb_price', 'model1_price', 'ttm_avg_rate'),
            'revenue': ('best_ttm_revenue_model', 'best_ttm_revenue', 'ttm_revenue'),
            'ttm_avg_rate': ('best_ttm_avg_rate_model', 'best_ttm_avg_rate', 'ttm_avg_rate'),
        }
        
        if request.model not in model_map:
            raise ValueError(f"Unknown model: {request.model}")
        
        model_name, feature_key, target = model_map[request.model]
        
        predictions = []
        errors = []
        successful = 0
        
        for idx, listing in enumerate(request.listings):
            try:
                request_id = f"batch_{request.model}_{datetime.now().timestamp()}_{idx}"
                
                pred = make_prediction(
                    listing_dict=listing.dict(),
                    model_name=model_name,
                    feature_key=feature_key,
                    cache=cache,
                    request_id=request_id
                )
                
                if request.model == 'price':
                    pred_formatted = f"${pred:.2f}/night"
                elif request.model == 'revenue':
                    pred_formatted = f"${pred:.2f}/month"
                else:
                    pred_formatted = f"${pred:.2f}"
                
                predictions.append({
                    "listing_index": idx, 
                    "prediction": pred, 
                    "prediction_formatted": pred_formatted, 
                    "success": True
                })
                successful += 1
                
            except Exception as e:
                logger.warning(f"Batch prediction error for listing {idx}: {e}")
                errors.append({"listing_index": idx, "error": str(e)})
        
        processing_time = time() - start_time
        
        return BatchPredictionResponse(
            success=len(errors) == 0,
            model=request.model,
            total_listings=len(request.listings),
            successful_predictions=successful,
            failed_predictions=len(errors),
            predictions=predictions,
            errors=errors,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise ValueError(f"Batch prediction failed: {str(e)}")


@app.post("/api/analyze/competitive", response_model=CompetitiveAnalysisResponse)
async def analyze_competitive_position(listing: ListingInput) -> CompetitiveAnalysisResponse:
    try:
        request_id = f"competitive_{datetime.now().timestamp()}"
        predicted_price = make_prediction(
            listing_dict=listing.dict(),
            model_name="best_ttm_avg_rate_model",
            feature_key="best_ttm_avg_rate",
            cache=cache,
            request_id=request_id
        )

        ref_df, cluster_model = _load_competitive_reference()
        cluster_id = int(cluster_model.predict([[float(listing.latitude), float(listing.longitude)]])[0])
        cluster_df = ref_df[ref_df["cluster_id"] == cluster_id]
        if cluster_df.empty:
            cluster_df = ref_df

        cluster_size = int(len(cluster_df))
        cluster_median_price = float(cluster_df["ttm_avg_rate_business"].median())
        cluster_median_rating = float(cluster_df["avg_rating"].median())
        cluster_median_reviews = float(cluster_df["num_reviews"].median())
        cluster_median_amenities = float(cluster_df["amenities_count"].median())

        price_component = 100.0 * (cluster_median_price - predicted_price) / (cluster_median_price + 1e-6)
        rating_component = 20.0 * (float(listing.avg_rating) - cluster_median_rating)
        reviews_component = 10.0 * (float(listing.num_reviews) - cluster_median_reviews) / (cluster_median_reviews + 1e-6)
        amenities_component = 10.0 * (
            _compute_amenities_count(listing.amenities, listing.amenities_count) - cluster_median_amenities
        ) / (cluster_median_amenities + 1e-6)

        competitiveness_score = float(np.clip(price_component + rating_component + reviews_component + amenities_component, -100, 100))
        price_vs_cluster = float(predicted_price / (cluster_median_price + 1e-6))

        if competitiveness_score >= 30:
            market_position = "Strong"
        elif competitiveness_score >= 5:
            market_position = "Good"
        elif competitiveness_score >= -20:
            market_position = "Fair"
        else:
            market_position = "Weak"

        price_percentile = float((cluster_df["ttm_avg_rate_business"] <= predicted_price).mean())
        rating_percentile = float((cluster_df["avg_rating"] <= float(listing.avg_rating)).mean())
        reviews_percentile = float((cluster_df["num_reviews"] <= float(listing.num_reviews)).mean())

        return CompetitiveAnalysisResponse(
            success=True,
            cluster_id=cluster_id,
            cluster_size=cluster_size,
            market_position=market_position,
            competitiveness_score=competitiveness_score,
            price_vs_cluster=price_vs_cluster,
            predicted_price=predicted_price,
            cluster_median_price=cluster_median_price,
            percentile_ranks={
                "price_percentile": price_percentile,
                "rating_percentile": rating_percentile,
                "reviews_percentile": reviews_percentile,
            },
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Competitive analysis error: {e}")
        raise ValueError(f"Competitive analysis failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
