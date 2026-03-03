from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_wine_quality


app = FastAPI()


class WineData(BaseModel):
    """
    Pydantic BaseModel representing wine chemical measurements.
    
    Attributes:
        alcohol (float): Alcohol content
        malic_acid (float): Malic acid content
        ash (float): Ash content
        alcalinity_of_ash (float): Alcalinity of ash
        magnesium (float): Magnesium content
        total_phenols (float): Total phenols
        flavanoids (float): Flavanoids content
        nonflavanoid_phenols (float): Nonflavanoid phenols
        proanthocyanins (float): Proanthocyanins content
        color_intensity (float): Color intensity
        hue (float): Hue
        od280_od315 (float): OD280/OD315 of diluted wines
        proline (float): Proline content
    """
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float


class WineResponse(BaseModel):
    response: int


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Simple health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    """
    Predict the wine class based on provided chemical features.
    
    This endpoint accepts wine chemical measurements and returns the predicted wine class.
    
    Args:
        wine_features (WineData): A WineData object containing:
            - All 13 chemical measurements
    
    Returns:
        WineResponse: A response object containing:
            - response (int): The predicted wine class (0, 1, or 2)
    
    Raises:
        HTTPException: Returns a 500 status code with error details if prediction fails.
    
    Example:
        POST /predict
        {
            "alcohol": 13.2,
            "malic_acid": 2.3,
            "ash": 2.4,
            "alcalinity_of_ash": 18.5,
            "magnesium": 110.0,
            "total_phenols": 2.8,
            "flavanoids": 3.1,
            "nonflavanoid_phenols": 0.28,
            "proanthocyanins": 2.0,
            "color_intensity": 5.6,
            "hue": 1.05,
            "od280_od315": 3.2,
            "proline": 1100.0
        }
        Response:
        {
            "response": 0
        }
    """
    try:
        features = [[
            wine_features.alcohol,
            wine_features.malic_acid,
            wine_features.ash,
            wine_features.alcalinity_of_ash,
            wine_features.magnesium,
            wine_features.total_phenols,
            wine_features.flavanoids,
            wine_features.nonflavanoid_phenols,
            wine_features.proanthocyanins,
            wine_features.color_intensity,
            wine_features.hue,
            wine_features.od280_od315,
            wine_features.proline
        ]]
        
        prediction = predict_wine_quality(features)
        return WineResponse(response=int(prediction[0]))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))