from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def train_ridge_model(X_train, y_train, alpha=1.0):
    """
    Trains a Ridge Regression model using provided training data.

    Parameters:
        X_train (DataFrame): Training feature matrix
        y_train (Series): Training target variable
        alpha (float): Regularization strength (default = 1.0)

    Returns:
        model (Ridge): Fitted Ridge regression model
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def predict_ridge(model, X_test):
    """
    Generates predictions using the trained Ridge model.

    Parameters:
        model (Ridge): Fitted Ridge model
        X_test (DataFrame): Test features

    Returns:
        np.array: Predicted target values
    """
    return model.predict(X_test)

def evaluate_ridge(y_true, y_pred):
    """
    Evaluates Ridge regression predictions using MSE and R-squared.

    Parameters:
        y_true (Series): Actual target values
        y_pred (np.array): Predicted target values

    Returns:
        dict: Evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'mse': mse,
        'r2': r2
    }
