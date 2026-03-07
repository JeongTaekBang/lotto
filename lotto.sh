#!/bin/bash
# Lotto AI - Multi-Model Prediction System (macOS)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/main_new.py"
PYTHON="python"

# conda 자동 감지 및 활성화
activate_conda() {
    for CONDA_SH in \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "/opt/anaconda3/etc/profile.d/conda.sh" \
        "/opt/homebrew/anaconda3/etc/profile.d/conda.sh" \
        "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"; do
        if [ -f "$CONDA_SH" ]; then
            source "$CONDA_SH"
            conda activate lotto 2>/dev/null
            return 0
        fi
    done
    return 1
}

activate_conda
if ! command -v python &>/dev/null; then
    echo "[ERROR] Python not found. Install Anaconda and create 'lotto' environment."
    exit 1
fi

show_menu() {
    clear
    echo "=================================================="
    echo "      Lotto AI - Multi-Model Prediction System"
    echo "=================================================="
    echo ""
    echo "  [1] Predict - Single Model"
    echo "  [2] Predict - Ensemble (Auto-Weight)"
    echo "  [3] Train - Single Model"
    echo "  [4] Train - All Models"
    echo "  [5] Compare Models"
    echo "  [6] Backtest"
    echo "  [7] Crawl Data"
    echo "  [8] Statistics"
    echo "  [9] CNN Grid Visual (Predict + Dashboard)"
    echo "  [0] Exit"
    echo ""
    echo "=================================================="
}

select_model() {
    echo ""
    echo "Available Models:"
    echo "  [1] gru           (Sequence)"
    echo "  [2] transformer   (Attention)"
    echo "  [3] random_forest (Ensemble)"
    echo "  [4] markov        (Statistical)"
    echo "  [5] lstm          (Sequence)"
    echo "  [6] xgboost       (Boosting)"
    echo "  [7] cnn_grid      (Spatial CNN)"
    echo ""
    read -p "Select model [1-7]: " model_choice
    case "$model_choice" in
        1) MODEL="gru" ;;
        2) MODEL="transformer" ;;
        3) MODEL="random_forest" ;;
        4) MODEL="markov" ;;
        5) MODEL="lstm" ;;
        6) MODEL="xgboost" ;;
        7) MODEL="cnn_grid" ;;
        *) MODEL="gru" ;;
    esac
}

select_mode() {
    echo "Selection Mode (MMR diversity):"
    echo "  [1] safe        - Conservative, high diversity"
    echo "  [2] balanced    - Balanced quality/diversity"
    echo "  [3] aggressive  - Score-focused, bold picks"
    echo "  [0] skip        - Legacy mode (no MMR)"
    echo ""
    read -p "Select mode [0-3, default 2]: " mode_choice
    case "$mode_choice" in
        1) MODE_ARG="--mode=safe" ;;
        3) MODE_ARG="--mode=aggressive" ;;
        0) MODE_ARG="" ;;
        *) MODE_ARG="--mode=balanced" ;;
    esac
}

while true; do
    show_menu
    read -p "Select: " choice

    case "$choice" in
    1)  # Predict - Single Model
        clear
        echo "=================================================="
        echo "       Predict - Single Model"
        echo "=================================================="
        select_model
        echo ""
        read -p "Number of sets (default 5): " sets
        sets=${sets:-5}
        echo ""
        select_mode
        echo ""
        echo "[Predicting with $MODEL...]"
        echo ""
        if [ -z "$MODE_ARG" ]; then
            "$PYTHON" "$SCRIPT" predict --model="$MODEL" --sets="$sets"
        else
            "$PYTHON" "$SCRIPT" predict --model="$MODEL" --sets="$sets" $MODE_ARG
        fi
        echo ""
        read -p "Press Enter to continue..."
        ;;

    2)  # Predict - Ensemble
        clear
        echo "=================================================="
        echo "       Predict - Ensemble (Auto-Weight)"
        echo "=================================================="
        echo ""
        echo "Uses models (XGBoost excluded) with Extended features."
        echo ""
        read -p "Number of sets (default 5): " sets
        sets=${sets:-5}
        echo ""
        select_mode
        echo ""
        if [ -z "$MODE_ARG" ]; then
            read -p "Apply filters? (Pattern, Statistical, Frequency) [Y/N, default Y]: " filter_choice
            FILTER="--filter"
            if [[ "$filter_choice" =~ ^[Nn]$ ]]; then FILTER=""; fi
        else
            FILTER=""
        fi
        echo ""
        echo "[Ensemble Prediction with Auto-Weight...]"
        echo ""
        "$PYTHON" "$SCRIPT" predict --ensemble --auto-weight --sets="$sets" $FILTER $MODE_ARG
        echo ""
        read -p "Press Enter to continue..."
        ;;

    3)  # Train - Single Model
        clear
        echo "=================================================="
        echo "       Train - Single Model"
        echo "=================================================="
        select_model
        echo ""
        read -p "Epochs (default 100): " epochs
        epochs=${epochs:-100}
        echo ""
        echo "[Training $MODEL for $epochs epochs...]"
        echo ""
        "$PYTHON" "$SCRIPT" train --model="$MODEL" --epochs="$epochs"
        echo ""
        read -p "Press Enter to continue..."
        ;;

    4)  # Train - All Models
        clear
        echo "=================================================="
        echo "       Train - All Models (XGBoost excluded)"
        echo "=================================================="
        echo ""
        echo "This will train 6 models with Extended features:"
        echo "  gru, transformer, random_forest, markov, lstm, cnn_grid"
        echo ""
        read -p "Epochs (default 100): " epochs
        epochs=${epochs:-100}
        echo ""
        read -p "Continue? [Y/N]: " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then continue; fi
        echo ""
        echo "[Training 6 models for $epochs epochs...]"
        echo ""
        "$PYTHON" "$SCRIPT" train --model=all --epochs="$epochs"
        echo ""
        read -p "Press Enter to continue..."
        ;;

    5)  # Compare Models
        clear
        echo "=================================================="
        echo "       Compare Models"
        echo "=================================================="
        echo ""
        read -p "Number of rounds (default 100): " rounds
        rounds=${rounds:-100}
        echo ""
        echo "[Comparing all models on last $rounds rounds...]"
        echo ""
        "$PYTHON" "$SCRIPT" compare --rounds="$rounds"
        echo ""
        read -p "Press Enter to continue..."
        ;;

    6)  # Backtest
        clear
        echo "=================================================="
        echo "       Backtest"
        echo "=================================================="
        echo ""
        echo "Evaluate which model?"
        echo "  [1] Single model"
        echo "  [2] All models"
        echo ""
        read -p "Select [1-2]: " eval_choice
        echo ""
        read -p "Number of rounds (default 100): " rounds
        rounds=${rounds:-100}

        if [ "$eval_choice" = "1" ]; then
            select_model
            echo ""
            echo "[Backtesting $MODEL on last $rounds rounds...]"
            echo ""
            "$PYTHON" "$SCRIPT" evaluate --model="$MODEL" --rounds="$rounds"
        else
            echo ""
            echo "[Backtesting all models on last $rounds rounds...]"
            echo ""
            "$PYTHON" "$SCRIPT" evaluate --model=all --rounds="$rounds"
        fi
        echo ""
        read -p "Press Enter to continue..."
        ;;

    7)  # Crawl Data
        clear
        echo "=================================================="
        echo "       Crawl Data"
        echo "=================================================="
        echo ""
        echo "[Crawling latest lottery data...]"
        echo ""
        "$PYTHON" "$SCRIPT" crawl
        echo ""
        read -p "Press Enter to continue..."
        ;;

    8)  # Statistics
        clear
        echo "=================================================="
        echo "       Statistics Analysis"
        echo "=================================================="
        echo ""
        "$PYTHON" "$SCRIPT" analyze
        echo ""
        read -p "Press Enter to continue..."
        ;;

    9)  # CNN Grid Visual
        clear
        echo "=================================================="
        echo "       CNN Grid Visual (Predict + Dashboard)"
        echo "=================================================="
        echo ""
        echo "Generates 4 images in analysis/output/:"
        echo "  - Recent rounds grid"
        echo "  - Frequency heatmap"
        echo "  - Prediction probability heatmap"
        echo "  - Dashboard (probability + 5 sets)"
        echo ""
        "$PYTHON" "$SCRIPT_DIR/analysis/visualize_grid.py"
        echo ""
        read -p "Press Enter to continue..."
        ;;

    0)  echo ""
        echo "Exiting..."
        exit 0
        ;;

    *)  echo ""
        echo "[!] Invalid input. Enter 0-9."
        sleep 2
        ;;
    esac
done
