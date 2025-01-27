import os
import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv
from collections import Counter
from typing import List, Tuple
import time

load_dotenv()


class LottoPredictor:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'db': os.getenv('DB_NAME'),
            'cursorclass': pymysql.cursors.DictCursor
        }
        self.number_range = range(1, 46)
        self.models = []

    def get_data(self) -> pd.DataFrame:
        with pymysql.connect(**self.db_config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT `1`,`2`,`3`,`4`,`5`,`6` FROM lotto ORDER BY count")
                results = cursor.fetchall()
                df = pd.DataFrame(results)
                # 문자열을 정수로 변환
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df.iloc[:-1].values  # All rows except last
        y = df.iloc[1:].values  # All rows except first
        return X, y

    def train_models(self):
        df = self.get_data()
        X, y = self.prepare_features(df)
        random_seed = int(time.time())  # 현재 시간을 seed로

        self.models = []
        for i in range(6):
            model = RandomForestRegressor(n_estimators=100, random_state=random_seed)
            model.fit(X, y[:, i])
            self.models.append(model)

        self.models = []
        for i in range(6):  # Train model for each position
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y[:, i])
            self.models.append(model)

    def get_frequency_weights(self) -> dict:
        df = self.get_data()
        all_numbers = df.values.flatten()
        frequency = Counter(all_numbers)
        total = sum(frequency.values())
        return {num: count / total for num, count in frequency.items()}

    def predict_next_numbers(self) -> List[int]:
        df = self.get_data()
        last_draw = df.iloc[-1].values.reshape(1, -1)

        # seed 값 범위 조정 (0 ~ 2^32-1)
        random_seed = int(time.time() * 1000) % (2 ** 32 - 1)
        np.random.seed(random_seed)

        predictions = []
        for model in self.models:
            pred = model.predict(last_draw)[0]
            noise = np.random.normal(0, 2)
            pred = round(pred + noise)
            predictions.append(max(1, min(45, pred)))

        while len(set(predictions)) < 6:
            new_num = np.random.randint(1, 46)
            if new_num not in predictions:
                predictions[predictions.count(min(predictions))] = new_num

        return sorted(list(set(predictions)))[:6]

        # Ensure no duplicates and numbers are within range
        final_numbers = []
        for pred in predictions:
            if pred not in final_numbers and 1 <= pred <= 45:
                final_numbers.append(pred)
            else:
                # Find next best number based on weights
                remaining = [n for n in self.number_range
                             if n not in final_numbers]
                next_num = max(remaining, key=lambda x: weights.get(x, 0))
                final_numbers.append(next_num)

        return sorted(final_numbers)

    def get_historical_accuracy(self) -> float:
        df = self.get_data()
        X, y = self.prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train models on training data
        test_models = []
        for i in range(6):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train[:, i])
            test_models.append(model)

        # Make predictions on test data
        correct_predictions = 0
        total_numbers = len(X_test) * 6

        for i, test_draw in enumerate(X_test):
            predictions = []
            for model in test_models:
                pred = round(model.predict(test_draw.reshape(1, -1))[0])
                predictions.append(pred)

            actual = y_test[i]
            correct_predictions += sum(1 for p, a in zip(predictions, actual)
                                       if abs(p - a) <= 2)

        return correct_predictions / total_numbers


def main():
    predictor = LottoPredictor()
    predictor.train_models()

    predicted_numbers = predictor.predict_next_numbers()
    accuracy = predictor.get_historical_accuracy()

    print(f"\nPredicted numbers for next draw: {predicted_numbers}")
    print(f"Historical prediction accuracy: {accuracy:.2%}")
    print("\nNote: This is for educational purposes only. Lottery outcomes are random.")


if __name__ == "__main__":
    main()