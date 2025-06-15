import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_model():
    """Create a logistic regression model within a pipeline."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])
    return pipeline


def train_model(X, y):
    """Train model on provided features and labels."""
    model = build_model()
    model.fit(X, y)
    return model


def save_model(model, path):
    """Persist trained model to the given path."""
    joblib.dump(model, path)


def load_model(path):
    """Load a saved model from disk."""
    return joblib.load(path)
