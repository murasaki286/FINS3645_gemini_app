import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def expanding_window_forecast(X, y, dates, alpha=1.0, initial_train_size=30, step_size=1, verbose=False):
    """
    Performs expanding window forecasting using Ridge Regression.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        dates (pd.Series): Corresponding dates
        alpha (float): Regularization strength for Ridge
        initial_train_size (int): Number of samples to start training
        step_size (int): Step size for expansion (default=1)
        verbose (bool): Whether to print progress

    Returns:
        pd.DataFrame: forecast results with date, predicted_return, actual_return, r2, mse
    """
    predictions = []
    actuals = []
    r2_scores = []
    mse_scores = []
    used_dates = []

    n = len(y)

    for i in range(initial_train_size, n, step_size):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[i:i + 1]
        y_test = y.iloc[i:i + 1]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)[0]
        predictions.append(y_pred)
        actuals.append(y_test.iloc[0])
        used_dates.append(dates.iloc[i])

        y_train_pred = model.predict(X_train)
        r2 = r2_score(y_train, y_train_pred)
        mse = mean_squared_error(y_train, y_train_pred)

        r2_scores.append(r2)
        mse_scores.append(mse)

        if verbose:
            print(f"Step {i}: Date={dates.iloc[i].date()}, Pred={y_pred:.5f}, Actual={y_test.iloc[0]:.5f}, R2={r2:.4f}")

    result_df = pd.DataFrame({
        "date": used_dates,
        "predicted_return": predictions,
        "actual_return": actuals,
        "r2": r2_scores,
        "mse": mse_scores
    })

    return result_df
