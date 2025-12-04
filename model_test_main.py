# model_test_main.py

from model_test_cnn import run as run_cnn
from model_test_rnn import run as run_rnn
from model_test_lstm import run as run_lstm
from model_test_gradient_boosting import run as run_lgbm


MODEL_DIR = "model_test_models"


def main():
    metrics = []
    metrics.append(run_cnn(MODEL_DIR))
    metrics.append(run_rnn(MODEL_DIR))
    metrics.append(run_lstm(MODEL_DIR))
    metrics.append(run_lgbm(MODEL_DIR))

    print("\n=== Model comparison (test set) ===")
    for m in metrics:
        print(
            f"{m['model']:10s}  "
            f"acc={m['accuracy']:.3f}  "
            f"prec={m['precision']:.3f}  "
            f"rec={m['recall']:.3f}  "
            f"f1={m['f1']:.3f}  "
            f"path={m['model_path']}"
        )


if __name__ == "__main__":
    main()
