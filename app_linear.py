import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# === Função para carregar os dados salvos ===
@st.cache_data
def carregar_dados():
    df = pd.read_csv("celtics_2025_26.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")

    # Renomear colunas para português e facilitar a leitura
    df = df.rename(columns={
        "GAME_DATE": "Data do Jogo",
        "MATCHUP": "Confronto",
        "WL": "Vitória/Derrota",
        "PTS": "Pontos",
        "REB": "Rebotes",
        "AST": "Assistências",
        "FGM": "Arremessos Convertidos",
        "FGA": "Arremessos Tentados",
        "FG3M": "Cestas de 3 Convertidas",
        "FG3A": "Cestas de 3 Tentativas",
        "FTM": "Lances Livres Convertidos",
        "FTA": "Lances Livres Tentados",
        "TOV": "Erros (Turnovers)"
    })

    return df

# === Interface Streamlit ===
st.title("Regressão Linear - Boston Celtics (Temporada 2025-26)")
st.write("Explore relações entre estatísticas do time e faça previsões com Regressão Linear Múltipla.")

# Carregar os dados
df = carregar_dados()

st.subheader("Visualização dos Dados")
st.dataframe(df.head())

# Seleção das variáveis
vars_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
vars_para_modelo = [v for v in vars_numericas if v not in ["SEASON_ID", "TEAM_ID", "GAME_ID"]]

st.markdown("### Selecione as Variáveis para o Modelo")
y_col = st.selectbox("Selecione a variável dependente (Y):", vars_numericas)
x_cols = st.multiselect("Selecione as variáveis independentes (X):", [v for v in vars_numericas if v != y_col])

if len(x_cols) == 0:
    st.warning("Selecione ao menos uma variável independente.")
    st.stop()

# === Treinamento do modelo ===
X = df[x_cols]
y = df[y_col]

modelo = LinearRegression()
modelo.fit(X, y)

# === Resultados ===
st.subheader("Resultados da Regressão Linear")

# Equação da regressão
st.write("**Equação da Regressão:**")
eq = f"{y_col} = {modelo.intercept_:.2f} + " + " + ".join([f"{coef:.2f}×{col}" for coef, col in zip(modelo.coef_, x_cols)])
st.latex(eq)

# Métricas
y_pred = modelo.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

col1, col2 = st.columns(2)
col1.metric("Erro Quadrático Médio (MSE)", f"{mse:.2f}")
col2.metric("Coeficiente de Determinação (R²)", f"{r2:.3f}")

# Coeficientes
st.write("### Coeficientes das Variáveis")
coef_df = pd.DataFrame({
    "Variável": x_cols,
    "Coeficiente (Impacto)": modelo.coef_
})
st.dataframe(coef_df)

# === Gráficos ===
st.subheader("Visualizações")

# Gráfico 1: Dispersão com linha de regressão (para 1 variável X)
if len(x_cols) == 1:
    fig, ax = plt.subplots()
    sns.regplot(x=X[x_cols[0]], y=y, ci=95, ax=ax, color="green", line_kws={"color": "red"})
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(y_col)
    ax.set_title("Dispersão com Linha de Regressão e Intervalo de Confiança")
    st.pyplot(fig)
else:
    st.info("ℹO gráfico de dispersão é exibido apenas quando há uma única variável independente.")

# Gráfico 2: Valores Reais vs Previstos
fig2, ax2 = plt.subplots()
sns.scatterplot(x=y, y=y_pred, ax=ax2)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax2.set_xlabel("Valor Real")
ax2.set_ylabel("Valor Previsto")
ax2.set_title("Comparação: Valores Reais vs Previstos")
st.pyplot(fig2)

# Gráfico 3: Tendência temporal (Predito x Real)
fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(df["Data do Jogo"], y, label="Real", marker="o")
ax3.plot(df["Data do Jogo"], y_pred, label="Previsto", marker="x")
ax3.fill_between(df["Data do Jogo"], y_pred - np.std(y_pred), y_pred + np.std(y_pred), color="gray", alpha=0.2)
ax3.legend()
ax3.set_title("Tendência Temporal com Intervalo de Confiança")
ax3.set_xlabel("Data do Jogo")
ax3.set_ylabel(y_col)
plt.xticks(rotation=45)  # ROTACIONA AS DATAS
plt.tight_layout()
st.pyplot(fig3)
