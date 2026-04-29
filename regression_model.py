"""
Modelo de Regressão Linear
==========================
Aplicação Python para treinar e avaliar um modelo de regressão linear
usando o dataset forestfires.

Uso:
    python regression_model.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Carregamento e exploração do dataset
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "forestfires.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_dataset(path: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um arquivo CSV."""
    df = pd.read_csv(path)
    print("=" * 60)
    print("DATASET CARREGADO")
    print("=" * 60)
    print(f"Registros : {len(df)}")
    print(f"Colunas   : {list(df.columns)}")
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nEstatísticas descritivas:")
    print(df.describe().round(2))
    print()
    return df


# ---------------------------------------------------------------------------
# 2. Pré-processamento
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    Separa features e alvo, divide em treino/teste e normaliza as features.

    Retorna:
        X_train_sc, X_test_sc : arrays normalizados
        y_train, y_test       : séries do alvo
        encoded_feature_names : lista com os nomes das features após codificação
        scaler                : objeto StandardScaler ajustado
    """
    if target not in df.columns:
        raise ValueError(
            f"Coluna alvo '{target}' não encontrada. Colunas disponíveis: {list(df.columns)}"
        )

    feature_names = [c for c in df.columns if c != target]
    features_df = df[feature_names]
    categorical_columns = features_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    X = pd.get_dummies(features_df, columns=categorical_columns, dtype=float, drop_first=True)
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"Divisão treino/teste : {len(X_train)} / {len(X_test)} amostras")
    print()

    encoded_feature_names = list(X.columns)
    return X_train_sc, X_test_sc, y_train, y_test, encoded_feature_names, scaler


# ---------------------------------------------------------------------------
# 3. Treinamento
# ---------------------------------------------------------------------------

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Treina um modelo de regressão linear."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")
    print()
    return model


# ---------------------------------------------------------------------------
# 4. Avaliação
# ---------------------------------------------------------------------------

def evaluate_model(
    model: LinearRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    encoded_feature_names: list,
) -> np.ndarray:
    """Avalia o modelo e imprime as métricas."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("=" * 60)
    print("MÉTRICAS DE AVALIAÇÃO")
    print("=" * 60)
    print(f"MAE  (Erro Absoluto Médio)      : {mae:>12,.2f}")
    print(f"MSE  (Erro Quadrático Médio)    : {mse:>12,.2f}")
    print(f"RMSE (Raiz do MSE)              : {rmse:>12,.2f}")
    print(f"R²   (Coeficiente de Determinação): {r2:.4f}")
    print()

    print("=" * 60)
    print("COEFICIENTES DO MODELO")
    print("=" * 60)
    print(f"{'Feature':<25} {'Coeficiente':>15}")
    print("-" * 42)
    for name, coef in zip(encoded_feature_names, model.coef_):
        print(f"{name:<25} {coef:>15,.2f}")
    print(f"{'Intercepto':<25} {model.intercept_:>15,.2f}")
    print()

    return y_pred


# ---------------------------------------------------------------------------
# 5. Visualização
# ---------------------------------------------------------------------------

def plot_results(y_test: np.ndarray, y_pred: np.ndarray, output_dir: str) -> None:
    """Gera e salva gráficos de avaliação do modelo."""
    os.makedirs(output_dir, exist_ok=True)

    # --- Gráfico 1: Real vs Predito ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="white", linewidth=0.5)
    lims = [
        min(y_test.min(), y_pred.min()) * 0.95,
        max(y_test.max(), y_pred.max()) * 1.05,
    ]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Predição perfeita")
    axes[0].set_xlabel("Área real queimada (ha)")
    axes[0].set_ylabel("Área predita (ha)")
    axes[0].set_title("Área Real vs Área Predita")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Gráfico 2: Distribuição dos Resíduos ---
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Resíduo = 0")
    axes[1].set_xlabel("Resíduo (ha)")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title("Distribuição dos Resíduos")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Avaliação do Modelo de Regressão Linear - Forest Fires", fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "avaliacao_modelo.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em: {output_path}")


# ---------------------------------------------------------------------------
# 6. Pipeline principal
# ---------------------------------------------------------------------------

def main():
    df = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test, encoded_feature_names, scaler = preprocess(
        df, target="area"
    )

    model = train_model(X_train, y_train)

    y_pred = evaluate_model(model, X_test, y_test, encoded_feature_names)

    plot_results(y_test, y_pred, OUTPUT_DIR)

    print("Execução concluída.")


if __name__ == "__main__":
    main()
