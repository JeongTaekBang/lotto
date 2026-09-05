"""XGBoost feature mode test"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasources.sqlite_source import SQLiteDataSource
from training.trainer import UnifiedTrainer
from models.factory import ModelFactory
import numpy as np

def test_model(feature_mode, dim):
    datasource = SQLiteDataSource()
    trainer = UnifiedTrainer(datasource, seq_length=20, feature_mode=feature_mode)
    X, y = trainer.prepare_sequences()

    # Last 100 rounds
    X_eval, y_eval = X[-100:], y[-100:]

    model = ModelFactory.create('xgboost', {'input_dim': dim})
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'xgboost.pkl')
    model.load(model_path)

    hits_dist = {i: 0 for i in range(7)}
    for i in range(len(X_eval)):
        sample = X_eval[i:i+1]
        pred = model.predict_proba(sample)
        top6_pred = set(pred.top_k(6))
        actual = set(np.where(y_eval[i] > 0.5)[0] + 1)
        hits = len(top6_pred & actual)
        hits_dist[hits] += 1

    total = len(X_eval)
    avg_hits = sum(k * v for k, v in hits_dist.items()) / total
    hit_3plus = sum(hits_dist[k] for k in range(3, 7)) / total * 100

    print(f'Feature mode: {feature_mode} ({dim}dim)')
    print(f'Average hits: {avg_hits:.2f}')
    print(f'3+ hit rate: {hit_3plus:.1f}%')
    print(f'Distribution: {hits_dist}')
    print()

if __name__ == '__main__':
    test_model('basic', 45)
