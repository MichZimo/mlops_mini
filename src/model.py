from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def create_model():
    return Pipeline(
        [
            ('Reg', LinearRegression())
        ]
    )