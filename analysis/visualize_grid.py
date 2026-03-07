"""CNN Grid 7x7 시각화 - 회차별 패턴 + 예측 확률 히트맵"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from datasources.mysql_source import MySQLDataSource
from models.factory import ModelFactory
from training.trainer import UnifiedTrainer

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

GRID_LABELS = np.arange(1, 50).reshape(7, 7)  # 1~49 (46~49는 빈칸)
VALID_MASK = GRID_LABELS <= 45


def numbers_to_grid(numbers):
    """당첨번호 → 7x7 바이너리 그리드"""
    grid = np.zeros((7, 7), dtype=float)
    for n in numbers:
        idx = n - 1
        r, c = divmod(idx, 7)
        grid[r, c] = 1.0
    return grid


def draw_grid(ax, grid, title, cmap='Blues', show_numbers=True, vmin=0, vmax=1,
              highlight=None):
    """7x7 그리드를 ax에 그리기"""
    display = np.where(VALID_MASK, grid, np.nan)
    ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    for r in range(7):
        for c in range(7):
            num = GRID_LABELS[r, c]
            if num > 45:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       fill=True, color='#f0f0f0', zorder=2))
                continue
            if show_numbers:
                val = grid[r, c]
                color = 'white' if val > 0.6 * vmax else 'black'
                fontweight = 'bold' if (highlight and num in highlight) else 'normal'
                fontsize = 9 if val < 0.01 else 10
                ax.text(c, r, str(num), ha='center', va='center',
                        fontsize=fontsize, color=color, fontweight=fontweight)
            if highlight and num in highlight:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       fill=False, edgecolor='red',
                                       linewidth=2.5, zorder=3))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, pad=8)


def main():
    # --- 데이터 로드 ---
    ds = MySQLDataSource()
    records = ds.load()
    last_round = records[-1].round_num

    os.makedirs('analysis/output', exist_ok=True)

    # =========================================================
    # 1. 최근 12회차 당첨번호 7x7 그리드
    # =========================================================
    recent = records[-12:]
    fig, axes = plt.subplots(3, 4, figsize=(16, 13))
    fig.suptitle('최근 12회차 당첨번호 (7x7 Grid)', fontsize=16, fontweight='bold', y=0.98)

    for i, rec in enumerate(recent):
        ax = axes[i // 4][i % 4]
        grid = numbers_to_grid(rec.numbers)
        draw_grid(ax, grid, f'{rec.round_num}회  {rec.numbers}',
                  cmap='Blues', highlight=set(rec.numbers))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = 'analysis/output/grid_recent_rounds.png'
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[1/4] 최근 12회차 그리드 저장: {path1}')

    # =========================================================
    # 2. 누적 빈도 히트맵 (전체 + 최근 50회)
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle('번호별 출현 빈도 히트맵', fontsize=16, fontweight='bold', y=1.02)

    # 전체 빈도
    freq_all = np.zeros(45)
    for rec in records:
        for n in rec.numbers:
            freq_all[n - 1] += 1
    freq_all_pct = freq_all / len(records) * 100

    grid_all = np.zeros((7, 7))
    for n in range(1, 46):
        r, c = divmod(n - 1, 7)
        grid_all[r, c] = freq_all_pct[n - 1]

    draw_grid(axes[0], grid_all, f'전체 ({len(records)}회차) 출현율 %',
              cmap='YlOrRd', vmin=0, vmax=freq_all_pct.max())
    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                norm=mcolors.Normalize(0, freq_all_pct.max()))
    plt.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.04, label='출현율 %')

    # 최근 50회 빈도
    recent50 = records[-50:]
    freq_50 = np.zeros(45)
    for rec in recent50:
        for n in rec.numbers:
            freq_50[n - 1] += 1
    freq_50_pct = freq_50 / 50 * 100

    grid_50 = np.zeros((7, 7))
    for n in range(1, 46):
        r, c = divmod(n - 1, 7)
        grid_50[r, c] = freq_50_pct[n - 1]

    draw_grid(axes[1], grid_50, f'최근 50회차 출현율 %',
              cmap='YlOrRd', vmin=0, vmax=freq_50_pct.max())
    sm2 = plt.cm.ScalarMappable(cmap='YlOrRd',
                                 norm=mcolors.Normalize(0, freq_50_pct.max()))
    plt.colorbar(sm2, ax=axes[1], fraction=0.046, pad=0.04, label='출현율 %')

    plt.tight_layout()
    path2 = 'analysis/output/grid_frequency_heatmap.png'
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[2/4] 빈도 히트맵 저장: {path2}')

    # =========================================================
    # 3. CNN Grid 모델 예측 확률 히트맵
    # =========================================================
    trainer = UnifiedTrainer(ds, seq_length=20, feature_mode='extended')
    latest_seq = trainer.get_latest_sequence()

    model = ModelFactory.create('cnn_grid', {
        'input_dim': trainer.feature_dim, 'seq_length': 20
    })
    model.load('saved_models/cnn_grid.pt')
    proba = model.predict_proba(latest_seq)
    probs = proba.probabilities

    # 예측 확률 7x7 그리드
    grid_prob = np.zeros((7, 7))
    for n in range(1, 46):
        r, c = divmod(n - 1, 7)
        grid_prob[r, c] = probs[n - 1]

    top6 = set(proba.top_k(6))
    top12 = set(proba.top_k(12))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(f'CNN Grid 모델 - {last_round + 1}회차 예측 확률',
                 fontsize=16, fontweight='bold', y=1.02)

    # 확률 히트맵
    draw_grid(axes[0], grid_prob, '예측 확률 분포',
              cmap='RdYlGn', vmin=probs.min(), vmax=probs.max(),
              highlight=top6)
    sm3 = plt.cm.ScalarMappable(cmap='RdYlGn',
                                 norm=mcolors.Normalize(probs.min(), probs.max()))
    plt.colorbar(sm3, ax=axes[0], fraction=0.046, pad=0.04, label='확률')

    # Top 12 강조 (확률 높은 번호만 표시)
    grid_top = np.zeros((7, 7))
    for n in top12:
        r, c = divmod(n - 1, 7)
        grid_top[r, c] = probs[n - 1]

    draw_grid(axes[1], grid_top, 'Top 12 번호 (빨간 테두리 = Top 6)',
              cmap='Greens', vmin=0, vmax=probs.max(),
              highlight=top6)
    sm4 = plt.cm.ScalarMappable(cmap='Greens',
                                 norm=mcolors.Normalize(0, probs.max()))
    plt.colorbar(sm4, ax=axes[1], fraction=0.046, pad=0.04, label='확률')

    plt.tight_layout()
    path3 = 'analysis/output/grid_prediction_heatmap.png'
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[3/4] 예측 확률 히트맵 저장: {path3}')

    # =========================================================
    # 4. 종합 대시보드 (5세트 예측 + 확률맵)
    # =========================================================
    from core.selector import CandidateSelector
    selector = CandidateSelector(preset='balanced')
    predictions = selector.select(proba, num_sets=5, model_name='cnn_grid')

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(f'{last_round + 1}회차 CNN Grid 예측 대시보드',
                 fontsize=16, fontweight='bold', y=0.99)

    # 상단 왼쪽: 확률 히트맵
    draw_grid(axes[0][0], grid_prob, '예측 확률 분포',
              cmap='RdYlGn', vmin=probs.min(), vmax=probs.max(),
              highlight=top6)

    # 5세트 예측
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for i, (pred, pos) in enumerate(zip(predictions, positions)):
        ax = axes[pos[0]][pos[1]]
        grid = numbers_to_grid(pred.numbers)
        score = pred.metadata.get('score', 0)
        draw_grid(ax, grid,
                  f'{i+1}세트: {pred.numbers}\n점수: {score:.2f}',
                  cmap='Blues', highlight=set(pred.numbers))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path4 = 'analysis/output/grid_dashboard.png'
    fig.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[4/4] 종합 대시보드 저장: {path4}')

    print(f'\n모든 이미지 저장 완료: analysis/output/')
    print(f'  - {path1}')
    print(f'  - {path2}')
    print(f'  - {path3}')
    print(f'  - {path4}')


if __name__ == '__main__':
    main()
