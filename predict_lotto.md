# Lottery Number Prediction Algorithm Documentation

## Algorithm Overview
The prediction system combines machine learning, statistical analysis, and randomization to generate varied lottery number predictions.

## Core Components

### 1. Random Forest Regression
- Trains 6 separate models for each number position
- Uses previous winning numbers to predict next numbers
- Adds random noise to predictions for variability

### 2. Randomization Process
```python
# Uses current timestamp for random seed
random_seed = int(time.time() * 1000) % (2**32 - 1)
```

### 3. Prediction Process
1. Get last drawing numbers
2. Feed into trained models
3. Add Gaussian noise to predictions
4. Handle duplicate numbers with random replacements
5. Ensure numbers are within valid range (1-45)

## Model Training
```python
def train_models(self):
    # Uses timestamp-based random seed
    # Trains separate model for each position
    # Includes random noise in predictions
```

## Prediction Enhancement
- Adds Gaussian noise (mean=0, std=2) to predictions
- Replaces duplicates with random numbers
- Maintains number range constraints

## Technical Requirements
```bash
Required packages:
- scikit-learn
- pandas
- numpy
- pymysql
- python-dotenv
```

## Usage Example
```python
predictor = LottoPredictor()
predictor.train_models()
predicted_numbers = predictor.predict_next_numbers()  # Different results each run
```

## Limitations
1. Past patterns don't guarantee future results
2. Lottery numbers are random by design
3. Model accuracy is for educational purposes only

## Note
This project is for educational and research purposes only. Lottery outcomes are random and cannot be reliably predicted.

## Author
Jeong Taek Bang
