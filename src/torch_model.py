import torch
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


class SurrogateModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def train_surrogate_model(
    df,
    feature_columns,
    target_column,
    epochs=1000,
    learning_rate=0.01,
    test_size=0.2,
    random_state=42
):

    # ------------------------------------------------
    # 1. Prepare data
    # ------------------------------------------------

    data = df[
        feature_columns + [target_column]
    ].dropna()

    X = data[
        feature_columns
    ].values.astype(
        np.float32
    )

    y = data[
        target_column
    ].values.reshape(
        -1,
        1
    ).astype(
        np.float32
    )


    # ------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


    # ------------------------------------------------
    # 3. Standardization
    # ------------------------------------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(
        X_train
    )

    X_test_scaled = x_scaler.transform(
        X_test
    )

    y_train_scaled = y_scaler.fit_transform(
        y_train
    )

    y_test_scaled = y_scaler.transform(
        y_test
    )


    # ------------------------------------------------
    # 4. Convert to PyTorch tensors
    # ------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32
    )


    # ------------------------------------------------
    # 5. Create model
    # ------------------------------------------------

    model = SurrogateModel(
        input_size=len(feature_columns)
    )


    # ------------------------------------------------
    # 6. Loss and optimizer
    # ------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # ------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------

    losses = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        predictions = model(
            X_train_tensor
        )

        loss = criterion(
            predictions,
            y_train_tensor
        )

        loss.backward()

        optimizer.step()

        losses.append(
            loss.item()
        )


    # ------------------------------------------------
    # 8. Predict on train set
    # ------------------------------------------------

    model.eval()

    with torch.no_grad():

        train_pred_scaled = model(
            X_train_tensor
        ).numpy()

        test_pred_scaled = model(
            X_test_tensor
        ).numpy()


    train_pred = y_scaler.inverse_transform(
        train_pred_scaled
    ).flatten()

    test_pred = y_scaler.inverse_transform(
        test_pred_scaled
    ).flatten()


    y_train_actual = y_train.flatten()
    y_test_actual = y_test.flatten()


    # ------------------------------------------------
    # 9. Metrics
    # ------------------------------------------------

    train_r2 = r2_score(
        y_train_actual,
        train_pred
    )

    train_mse = mean_squared_error(
        y_train_actual,
        train_pred
    )


    test_r2 = r2_score(
        y_test_actual,
        test_pred
    )

    test_mse = mean_squared_error(
        y_test_actual,
        test_pred
    )


    # ------------------------------------------------
    # 10. Training residuals
    # ------------------------------------------------

    train_residuals = (
        y_train_actual
        - train_pred
    )


    # ------------------------------------------------
    # 11. Feature ranges for OOD detection
    # ------------------------------------------------

    feature_ranges = {}

    for i, feature in enumerate(
        feature_columns
    ):

        feature_ranges[feature] = {
            "min": float(
                X_train[:, i].min()
            ),
            "max": float(
                X_train[:, i].max()
            )
        }


    # ------------------------------------------------
    # 12. Return everything needed by the app
    # ------------------------------------------------

    return {
        "model": model,

        "x_scaler": x_scaler,
        "y_scaler": y_scaler,

        "feature_columns": feature_columns,
        "target_column": target_column,

        "losses": losses,

        "train_actual": y_train_actual,
        "train_predicted": train_pred,

        "test_actual": y_test_actual,
        "test_predicted": test_pred,

        "train_r2": train_r2,
        "train_mse": train_mse,

        "test_r2": test_r2,
        "test_mse": test_mse,

        "training_residuals": train_residuals,

        "feature_ranges": feature_ranges
    }


def predict_surrogate(
    trained_result,
    input_values
):

    model = trained_result[
        "model"
    ]

    x_scaler = trained_result[
        "x_scaler"
    ]

    y_scaler = trained_result[
        "y_scaler"
    ]


    # ------------------------------------------------
    # Prepare input
    # ------------------------------------------------

    X = np.array(
        [input_values],
        dtype=np.float32
    )


    X_scaled = x_scaler.transform(
        X
    )


    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )


    # ------------------------------------------------
    # Prediction
    # ------------------------------------------------

    model.eval()

    with torch.no_grad():

        prediction_scaled = model(
            X_tensor
        ).numpy()


    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )


    return float(
        prediction[0][0]
    )


def check_out_of_distribution(
    trained_result,
    input_values
):

    feature_columns = trained_result[
        "feature_columns"
    ]

    feature_ranges = trained_result[
        "feature_ranges"
    ]

    out_of_range = []


    for feature, value in zip(
        feature_columns,
        input_values
    ):

        minimum = feature_ranges[
            feature
        ]["min"]

        maximum = feature_ranges[
            feature
        ]["max"]


        if (
            value < minimum
            or value > maximum
        ):

            out_of_range.append(
                {
                    "feature": feature,
                    "value": value,
                    "min": minimum,
                    "max": maximum
                }
            )


    return out_of_range