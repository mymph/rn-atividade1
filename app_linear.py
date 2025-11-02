import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# === CONFIGURAÇÃO DA PÁGINA ===
st.set_page_config(
    page_title="Celtics Stats Analyzer",
    page_icon="🏀",
    layout="wide"
)

# === CSS PERSONALIZADO ===
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #007A33;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .celtics-green {
        background-color: #007A33;
        color: white;
        padding: 10px;
        border-radius: 10px;
    }
    .stats-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #007A33;
        margin: 10px 0px;
    }
    .stButton>button {
        background-color: #007A33;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #005A25;
        color: white;
    }
    .spacing-large {
        margin-bottom: 3rem;
    }
    .spacing-medium {
        margin-bottom: 2rem;
    }
    .spacing-small {
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# === Função para carregar os dados salvos ===
@st.cache_data
def carregar_dados():
    df = pd.read_csv("celtics_2024_25.csv")
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
        "FG_PCT": "Percentual de Arremesso",
        "FG3M": "Cestas de 3 Convertidas",
        "FG3A": "Cestas de 3 Tentativas",
        "FG3_PCT": "Percentual de 3 Pontos",
        "FTM": "Lances Livres Convertidos",
        "FTA": "Lances Livres Tentados",
        "FT_PCT": "Percentual de Lances Livres",
        "OREB": "Rebotes Ofensivos",
        "DREB": "Rebotes Defensivos",
        "STL": "Roubos de Bola",
        "BLK": "Tocos",
        "TOV": "Erros (Turnovers)",
        "PF": "Faltas",
        "PLUS_MINUS": "+/-"
    })

    return df

# === HEADER PERSONALIZADO ===
st.markdown('<h1 class="main-header">🏀 Celtics Stats Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<div class="celtics-green"><h3 style="margin:0; text-align:center;">Análise de Desempenho - Temporada 2024/25</h3></div>', unsafe_allow_html=True)

# === INTRODUÇÃO ===
with st.container():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <p style='font-size: 1.2rem;'>Explore relações entre estatísticas do Boston Celtics e faça previsões usando Regressão Linear.</p>
        </div>
        """, unsafe_allow_html=True)

# Carregar os dados
df = carregar_dados()

# === SIDEBAR PARA SELEÇÃO DE VARIÁVEIS ===
with st.sidebar:
    st.markdown("### ☘️ Configurações do Modelo")
    st.markdown("---")
    
    # Filtro de jogos por data
    st.markdown("**Filtro por Data**")
    min_date = df["Data do Jogo"].min()
    max_date = df["Data do Jogo"].max()
    date_range = st.date_input(
        "Selecione o período:",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtrar dados por data
    if len(date_range) == 2:
        mask = (df["Data do Jogo"] >= pd.to_datetime(date_range[0])) & (df["Data do Jogo"] <= pd.to_datetime(date_range[1]))
        df = df[mask]

# === SEÇÃO DE DADOS ===
st.markdown("---")
st.markdown("### ☘️ Visualização dos Dados")

with st.expander("Clique para ver os dados da temporada", expanded=False):
    col1, col2 = st.columns([3,1])
    
    with col1:
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.markdown("#### Estatísticas Gerais")
        st.metric("🍀 Total de Jogos", len(df))
        st.metric("🏆 Vitórias", len(df[df["Vitória/Derrota"] == "W"]))
        st.metric("💔 Derrotas", len(df[df["Vitória/Derrota"] == "L"]))
        st.metric("💚 Pontos por Jogo", f"{df['Pontos'].mean():.1f}")

# === SELEÇÃO DE VARIÁVEIS ===
st.markdown("---")
st.markdown("### ☘️ Configuração do Modelo de Regressão")

# Definir variáveis que fazem sentido para o modelo (removendo IDs e colunas não numéricas)
vars_nao_permitidas = ["SEASON_ID", "TEAM_ID", "GAME_ID", "Data do Jogo", "Confronto", "Vitória/Derrota"]
vars_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
vars_permitidas = [v for v in vars_numericas if v not in vars_nao_permitidas]

# Organizar as variáveis por categoria
var_categories = {
    "Pontuação": ["Pontos", "Arremessos Convertidos", "Arremessos Tentados", "Percentual de Arremesso"],
    "3 Pontos": ["Cestas de 3 Convertidas", "Cestas de 3 Tentativas", "Percentual de 3 Pontos"],
    "Lances Livres": ["Lances Livres Convertidos", "Lances Livres Tentados", "Percentual de Lances Livres"],
    "Rebotes": ["Rebotes", "Rebotes Ofensivos", "Rebotes Defensivos"],
    "Outras Estatísticas": ["Assistências", "Roubos de Bola", "Tocos", "Erros (Turnovers)", "Faltas", "+/-"]
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ☘ Variável Dependente (Y)")
    st.markdown("*O que você quer prever?*")
    y_col = st.selectbox(
        "Selecione a variável alvo:",
        vars_permitidas,
        key="y_var"
    )

with col2:
    st.markdown("#### ☘ Variáveis Independentes (X)")
    st.markdown("*Quais estatísticas influenciam a previsão?*")
    
    # Seleção por categorias
    selected_vars = []
    for category, variables in var_categories.items():
        # Mostrar apenas variáveis que existem no dataframe
        available_vars = [v for v in variables if v in vars_permitidas]
        if available_vars:
            with st.expander(f"{category}", expanded=False):
                for var in available_vars:
                    if st.checkbox(var, key=f"check_{var}"):
                        selected_vars.append(var)

# Usar as variáveis selecionadas
x_cols = selected_vars

if len(x_cols) == 0:
    st.warning("⚠️ Selecione ao menos uma variável independente para continuar.")
    st.info("💡 **Dica:** Tente selecionar variáveis como 'Arremessos Convertidos', 'Cestas de 3 Tentativas' ou 'Rebotes' para prever 'Pontos'")
    st.stop()

# === TREINAMENTO DO MODELO ===
X = df[x_cols]
y = df[y_col]

modelo = LinearRegression()
modelo.fit(X, y)

# === RESULTADOS ===
st.markdown("---")
st.markdown("### ☘️ Resultados da Regressão Linear")

# Métricas em cards
y_pred = modelo.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mse)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="stats-card">
        <h4 style="margin:0; color: #007A33;">R² Score</h4>
        <h2 style="margin:0; color: #007A33;">{r2:.3f}</h2>
        <p style="margin:0; font-size: 0.8rem;">Quanto mais próximo de 1, melhor</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stats-card">
        <h4 style="margin:0; color: #007A33;">RMSE</h4>
        <h2 style="margin:0; color: #007A33;">{rmse:.2f}</h2>
        <p style="margin:0; font-size: 0.8rem;">Raiz do Erro Quadrático Médio</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stats-card">
        <h4 style="margin:0; color: #007A33;">MSE</h4>
        <h2 style="margin:0; color: #007A33;">{mse:.2f}</h2>
        <p style="margin:0; font-size: 0.8rem;">Erro Quadrático Médio</p>
    </div>
    """, unsafe_allow_html=True)

# MAIS ESPAÇO aqui
st.markdown('<div class="spacing-medium"></div>', unsafe_allow_html=True)

# Equação da regressão
st.markdown("#### Equação da Regressão")
eq_parts = [f"{modelo.intercept_:.2f}"]
for coef, col in zip(modelo.coef_, x_cols):
    eq_parts.append(f"{coef:+.2f} × {col}")  # ESPAÇO adicionado

eq = f"{y_col} = " + " ".join(eq_parts)
st.code(eq, language="latex")

# MAIS ESPAÇO aqui
st.markdown('<div class="spacing-medium"></div>', unsafe_allow_html=True)

# Coeficientes
st.markdown("#### Impacto das Variáveis")
coef_df = pd.DataFrame({
    "Variável": x_cols,
    "Coeficiente": modelo.coef_,
    "Impacto Absoluto": np.abs(modelo.coef_)
}).sort_values("Impacto Absoluto", ascending=False)

coef_df["Influência"] = coef_df["Coeficiente"].apply(
    lambda x: "🟢 Positiva" if x > 0 else "🔴 Negativa" if x < 0 else "⚪ Neutra"
)

# CORREÇÃO: Formatação correta dos coeficientes
coef_df_display = coef_df[["Variável", "Coeficiente", "Influência"]].copy()
coef_df_display["Coeficiente"] = coef_df_display["Coeficiente"].apply(lambda x: f"{x:.4f}")

st.dataframe(
    coef_df_display,
    use_container_width=True,
    hide_index=True
)

# === GRÁFICOS ===
st.markdown("---")
st.markdown("### ☘️ Visualizações")

tab1, tab2, tab3 = st.tabs(["Dispersão", "Reais vs Previstos", "Tendência Temporal"])

with tab1:
    if len(x_cols) == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(x=X[x_cols[0]], y=y, ci=95, ax=ax, 
                   scatter_kws={'alpha':0.6, 'color':'#007A33'}, 
                   line_kws={'color':'#BA9653', 'linewidth':2})
        ax.set_xlabel(x_cols[0])
        ax.set_ylabel(y_col)
        ax.set_title(f"Relação entre {x_cols[0]} e {y_col}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    else:
        st.info("✗ O gráfico de dispersão é exibido apenas quando há uma única variável independente.")
        
        # Mostrar matriz de correlação para múltiplas variáveis
        st.markdown("#### 🔗 Matriz de Correlação")
        corr_data = df[x_cols + [y_col]].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_data, annot=True, cmap="RdYlGn", center=0, ax=ax_corr)
        ax_corr.set_title("Correlação entre Variáveis", fontsize=14, fontweight='bold')
        st.pyplot(fig_corr)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred, ax=ax2, alpha=0.7, color='#007A33')
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    ax2.set_xlabel("Valor Real")
    ax2.set_ylabel("Valor Previsto")
    ax2.set_title("Comparação: Valores Reais vs Previstos", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(df["Data do Jogo"], y, label="Real", marker="o", markersize=4, linewidth=2, color='#007A33')
    ax3.plot(df["Data do Jogo"], y_pred, label="Previsto", marker="x", markersize=4, linewidth=2, color='#BA9653')
    ax3.fill_between(df["Data do Jogo"], y_pred - rmse, y_pred + rmse, color="gray", alpha=0.2, label="Intervalo de Confiança")
    ax3.legend()
    ax3.set_title("Evolução Temporal: Valores Reais vs Previstos", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Data do Jogo")
    ax3.set_ylabel(y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)

# === VALIDAÇÃO DO MODELO ===
st.markdown("---")
st.markdown("### ☘️ Validação do Modelo")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⦾ Verificação Rápida")
    st.markdown("""
    **Para validar se o modelo está correto:**

    **R² entre 0–1**: Quanto mais próximo de 1, melhor.  
    **Coeficientes coerentes**: Ex.: mais assistências → mais pontos.  
    **Resíduos aleatórios**: Sem padrões óbvios no gráfico.  
    **Previsões próximas da linha**: No gráfico Real vs. Previsto.
    """)

# MAIS ESPAÇO entre as subseções de validação
st.markdown('<div class="spacing-medium"></div>', unsafe_allow_html=True)

with col2:
    st.markdown("#### ⦾ Teste de Sanidade")
    
    # Teste simples com dados conhecidos
    if st.button("Rodar Teste de Validação"):
        from sklearn.model_selection import train_test_split
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar novo modelo
        model_test = LinearRegression()
        model_test.fit(X_train, y_train)
        
        # Prever no teste
        y_pred_test = model_test.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        
        st.success(f"✓ R² no conjunto de teste: {r2_test:.3f}")
        st.info(f"  Comparação - Treino: {r2:.3f} | Teste: {r2_test:.3f}")
        
        if abs(r2 - r2_test) < 0.2:
            st.success("✓ Modelo está generalizando bem!")
        else:
            st.warning("✗ Pode haver overfitting - diferença grande entre treino e teste")

# MAIS ESPAÇO entre as subseções de validação
st.markdown('<div class="spacing-medium"></div>', unsafe_allow_html=True)

# Exemplo de cálculo manual para validação
st.markdown("#### ⦾ Cálculo Manual de Validação")
if st.checkbox("Mostrar exemplo de cálculo manual"):
    # Pegar primeira linha como exemplo
    sample_idx = 0
    sample_X = X.iloc[sample_idx].values
    manual_pred = modelo.intercept_ + np.sum(modelo.coef_ * sample_X)
    
    st.write(f"**Exemplo para o jogo {sample_idx + 1}:**")
    st.write(f"- Valores reais: {X.iloc[sample_idx].to_dict()}")
    st.write(f"- Predição do modelo: {y_pred[sample_idx]:.2f}")
    st.write(f"- Cálculo manual: {manual_pred:.2f}")
    st.write(f"- Valor real de {y_col}: {y.iloc[sample_idx]:.2f}")
    
    if abs(manual_pred - y_pred[sample_idx]) < 0.01:
        st.success("✓ Cálculos batem! Modelo está correto.")
    else:
        st.error("✗ Cálculos não batem! Verifique o modelo.")

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏀 Boston Celtics Stats Analyzer | Temporada 2024-25</p>
</div>
""", unsafe_allow_html=True)