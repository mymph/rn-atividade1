import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nba_api.stats.endpoints import teamgamelog
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# === Função paSra buscar os dados do Celtics ===
@st.cache_data
def carregar_dados():
    gamelog = teamgamelog.TeamGameLog(team_id=1610612738, season='2024-25')
    df = gamelog.get_data_frames()[0]
    # Limpeza e organização básica
    df = df[['GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG3M', 'FG3A', 'TOV']]
    df['GAME_DATE'] = pdF.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    return df

# === App Streamlit ===
st.title("Regressão Linear - Boston Celtics (Temporada Atual)")
st.write("Selecione as variáveis para construir e visualizar o modelo de regressão linear múltipla.")

df = carregar_dados()
st.subheader("Visualização dos dados")
st.dataframe(df.head())

# Seleção de variáveis
vars_disponiveis = df.select_dtypes(include=[np.number]).columns.tolist()
y_col = st.selectbox("Selecione a variável dependente (Y):", vars_disponiveis)
x_cols = st.multiselect("Selecione as variáveis independentes (X):", [v for v in vars_disponiveis if v != y_col])

if len(x_cols) == 0:
    st.warning("Selecione ao menos uma variável independente.")
    st.stop()

# Treinamento
X = df[x_cols]
y = df[y_col]

model = LinearRegression()
model.fit(X, y)

# Resultados
st.subheader("Resultados da Regressão Linear")

st.write("**Equação:**")
eq = f"{y_col} = {model.intercept_:.2f} + " + " + ".join([f"{coef:.2f}×{col}" for coef, col in zip(model.coef_, x_cols)])
st.latex(eq)

# Métricas
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.metric("Erro Quadrático Médio (MSE)", round(mse, 2))
st.metric("Coeficiente de Determinação (R²)", round(r2, 3))

# Coeficientes
coef_df = pd.DataFrame({
    "Variável": x_cols,
    "Coeficiente": model.coef_
})
st.dataframe(coef_df)

# Gráficos
st.subheader("Gráficos")

# Dispersão + Linha de Regressão (para o caso de 1 variável)
if len(x_cols) == 1:
    fig, ax = plt.subplots()
    sns.regplot(x=X[x_cols[0]], y=y, ci=95, ax=ax)
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(y_col)
    ax.set_title("Dispersão com Linha de Regressão e Intervalo de Confiança")
    st.pyplot(fig)
else:
    st.info("O gráfico de dispersão é exibido apenas quando há uma variável X.")

# Previsão vs Realidade
fig2, ax2 = plt.subplots()
sns.scatterplot(x=y, y=y_pred, ax=ax2)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax2.set_xlabel("Valor Real")
ax2.set_ylabel("Valor Previsto")
ax2.set_title("Valores Reais vs Previstos")
st.pyplot(fig2)

# Tendência temporal (Predito x Real)
fig3, ax3 = plt.subplots()
ax3.plot(df['GAME_DATE'], y, label='Real', marker='o')
ax3.plot(df['GAME_DATE'], y_pred, label='Previsto', marker='x')
ax3.fill_between(df['GAME_DATE'], y_pred - np.std(y_pred), y_pred + np.std(y_pred), color='gray', alpha=0.2)
ax3.legend()
ax3.set_title("Tendência Temporal com Intervalo de Confiança")
ax3.set_xlabel("Data do Jogo")
ax3.set_ylabel(y_col)
st.pyplot(fig3)
