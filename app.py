import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Risco de Turnover",
    page_icon="🔮",
    layout="centered"
)

# ============================================================
# DEBUG
# ============================================================
st.sidebar.write("Arquivos no diretorio:")
st.sidebar.write(os.listdir())

# ============================================================
# LOADERS
# ============================================================
@st.cache_resource
def carregar_modelo():
    caminho = "modelo_turnover_lgbm_v1.pkl"
    if not os.path.exists(caminho):
        raise FileNotFoundError("modelo_turnover_lgbm_v1.pkl NAO encontrado")
    return joblib.load(caminho)

@st.cache_data
def carregar_mercer():
    caminho = "Mercer.xlsx"
    if not os.path.exists(caminho):
        raise FileNotFoundError("Mercer.xlsx NAO encontrado")
    df = pd.read_excel(caminho)
    df.columns = ['cargo', 'p80', 'p100', 'p120', 'cargo2', 'codigo']
    df['cargo'] = df['cargo'].str.strip().str.upper()
    return df[['cargo', 'p80', 'p100', 'p120']].dropna()

try:
    modelo_dict = carregar_modelo()
    mercer      = carregar_mercer()
except Exception as e:
    st.error(f"Falha ao iniciar aplicacao: {e}")
    st.stop()

# ============================================================
# SENHA
# ============================================================
CORRECT_PASSWORD = st.secrets["general"]["password"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Simulador de Risco de Turnover")
    senha = st.text_input("Senha de acesso", type="password")
    if st.button("Entrar"):
        if senha == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Senha incorreta.")
    st.stop()

# ============================================================
# PIPELINE
# ============================================================
_T = 'Tempo de Casa (Meses)'

def feature_engineering(X):
    X = X.copy()
    t = _T
    X['Salario_por_Tempo']  = X['Salario'] / (X[t] + 1)
    X['Salario_por_Idade']  = X['Salario'] / (X['Idade'] + 1)
    X['Salario_Log']        = np.log1p(X['Salario'])
    X['Salario_Cubico']     = X['Salario'] ** (1/3)
    X['Idade_x_Salario']    = X['Idade'] * X['Salario']
    X['Salario_x_Nivel']    = X['Salario'] * X['Nivel_vs_Mercado']
    X['Tempo_Quadrado']     = X[t] ** 2
    X['Tempo_Log']          = np.log1p(X[t])
    X['Tempo_x_Idade']      = X[t] * X['Idade']
    X['Tempo_x_Mercado']    = X[t] * X['Dist_Mercado']
    X['Posicao_x_Tempo']    = X['Posicao_Relativa'] * X[t]
    X['Dist_Mercado_Abs']   = X['Dist_Mercado'].abs()
    X['Dist_x_Posicao']     = X['Dist_Mercado'] * X['Posicao_Relativa']
    X['Dist_x_Tempo_Log']   = X['Dist_Mercado'] * np.log1p(X[t])
    X['Nivel_Quadrado']     = X['Nivel_vs_Mercado'] ** 2
    X['Nivel_x_Posicao']    = X['Nivel_vs_Mercado'] * X['Posicao_Relativa']
    X['Idade_Quadrado']     = X['Idade'] ** 2
    X['Risco_Composto']     = X['Dist_Mercado'] * X[t] * X['Posicao_Relativa']
    X['Risco_3x']           = (
        (X['Dist_Mercado'] < -0.1).astype(int) +
        (X['Posicao_Relativa'] < 0.4).astype(int) +
        (X[t] < 24).astype(int)
    )
    X['Estagnado']      = ((X[t] > 24) & (X['Dist_Mercado'] < 0)).astype(int)
    X['Veterano']       = (X[t] > 60).astype(int)
    X['Idade_30_40']    = ((X['Idade'] >= 30) & (X['Idade'] <= 40)).astype(int)
    ct = X['Contrato'].astype(str).str.upper()
    X['Estagio']        = ct.str.contains('EST').astype(int)
    X['Prazo_Det']      = ct.str.contains('DETERMINADO').astype(int)
    X['Estagio_Longo']  = ((ct.str.contains('EST')) & (X[t] > 24)).astype(int)
    X['Aprendiz_Longo'] = ((ct.str.contains('APRENDIZ')) & (X[t] > 12)).astype(int)
    X.drop(columns=['Contrato'], inplace=True)
    return X

def encoding_perfil(X):
    X = X.copy()
    for col in ['Nivel', 'Escolaridade', 'Sexo', 'Estado_Civil', 'Vinculo']:
        if col in X.columns and X[col].dtype == object:
            X[col] = X[col].astype(str).str.strip().str.upper()

    nivel_map = {
        '2 - ESTAGIARIO': 2, '3 - ASSISTENTE': 3,
        '4 - ANALISTA':   4, '5 - ESPECIALISTA': 5,
        '5 - COORDENADOR': 5, '6 - GERENTE': 6,
        '7 - DIRETOR':    7,
    }
    X['Nivel'] = X['Nivel'].map(nivel_map).fillna(4)

    esc_map = {
        'ENSINO MEDIO COMPLETO':       1,
        'EDUCACAO SUPERIOR COMPLETA':  2,
        'POS-GRADUADO':                2,
        'MESTRADO COMPLETO':           2,
        'DOUTORADO COMPLETO':          3,
    }
    X['Escolaridade'] = X['Escolaridade'].map(esc_map).fillna(1)

    ufs = ['RJ', 'SP', 'MG']
    X['UF_norm'] = X['UF'].astype(str).str.strip().str.upper()
    for uf in ufs:
        X[f'UF_{uf}'] = (X['UF_norm'] == uf).astype(int)
    X['UF_OUTROS'] = (~X['UF_norm'].isin(ufs)).astype(int)
    X.drop(columns=['UF', 'UF_norm'], inplace=True)

    X['Vinculo_PJ'] = (X['Vinculo'].astype(str).str.upper() == 'PJ').astype(int)
    X.drop(columns=['Vinculo'], inplace=True)

    X['Sexo_upper'] = X['Sexo'].astype(str).str.upper()
    X['Sexo_MASCULINO'] = (X['Sexo_upper'] == 'MASCULINO').astype(int)
    X.drop(columns=['Sexo', 'Sexo_upper'], inplace=True)

    X['EstCivil_Solteiro'] = (X['Estado_Civil'].astype(str).str.upper().str.contains('SOLTEIRO')).astype(int)
    X.drop(columns=['Estado_Civil'], inplace=True)

    X['Cargo']              = 0.3334
    X['Centro de custo']    = 0.2857
    X['VP']                 = 0.2160

    return X

def renomear_para_modelo(X):
    renames = {
        'Salario'          : 'Salário',
        'Posicao_Relativa' : 'Posição Relativa no Nível',
        'Nivel'            : 'Nível',
        'Vinculo_PJ'       : 'Vínculo_PJ',
    }
    return X.rename(columns=renames)

def prever(row_dict):
    df_in = pd.DataFrame([row_dict])
    X = feature_engineering(df_in)
    X = encoding_perfil(X)
    X = renomear_para_modelo(X)
    X = X.reindex(columns=modelo_dict['columns'], fill_value=0)
    prob = modelo_dict['calibrador'].predict_proba(X)[:, 1][0]
    ct = str(row_dict['Contrato']).upper()
    if 'EST' in ct and row_dict[_T] > 24:
        prob = max(prob, 0.95)
    return float(np.clip(prob, 0, 1))

# ============================================================
# UI
# ============================================================
st.title("Simulador de Risco de Turnover")
st.caption("Conexa Saude - People Analytics")

lista_cargos = sorted(mercer['cargo'].tolist())

col1, col2 = st.columns(2)
with col1:
    cargo   = st.selectbox("Cargo (Mercer)", lista_cargos)
    salario = st.number_input("Salario (R$)", min_value=500, max_value=100000, value=8000, step=500)
    tempo   = st.number_input("Tempo de casa (meses)", min_value=0, max_value=360, value=24, step=1)
    idade   = st.number_input("Idade", min_value=16, max_value=70, value=30, step=1)

with col2:
    nivel   = st.selectbox("Nivel hierarquico", [
        "2 - Estagiario", "3 - Assistente", "4 - Analista",
        "5 - Especialista", "5 - Coordenador", "6 - Gerente", "7 - Diretor"
    ])
    sexo    = st.selectbox("Sexo", ["Feminino", "Masculino"])
    civil   = st.selectbox("Estado civil", ["Solteiro(a)", "Casado(a)", "Divorciado(a)", "Viuvo(a)"])
    uf      = st.selectbox("UF", ["SP", "RJ", "MG", "Outros"])

col3, col4 = st.columns(2)
with col3:
    vinculo  = st.selectbox("Vinculo", ["CLT", "PJ", "Estagiario", "Temporario"])
    contrato = st.selectbox("Contrato", ["Prazo Indeterminado", "Contrato de Estagio", "Contrato Determinado"])
with col4:
    escol  = st.selectbox("Escolaridade", [
        "Ensino Medio Completo", "Educacao Superior Completa",
        "Pos-Graduado", "Mestrado Completo", "Doutorado Completo"
    ])
    vp_sel = st.selectbox("VP", [
        "Saude", "Comercial", "Tecnologia",
        "Gente e Juridico", "Financeiro", "Presidencia", "Produto"
    ])

# Mercer
ref = mercer[mercer['cargo'] == cargo]
if not ref.empty:
    p80  = float(ref['p80'].values[0])
    p100 = float(ref['p100'].values[0])
    p120 = float(ref['p120'].values[0])
    dist_mercado  = (salario - p100) / p100
    nivel_vs_merc = dist_mercado
    pos_relativa  = float(np.clip((salario - p80) / (p120 - p80) if (p120 - p80) > 0 else 0.5, 0, 1))
    st.info(f"Ref. Mercer p100: R$ {p100:,.0f} | Dist. Mercado: {dist_mercado*100:+.1f}% | Posicao na faixa: {pos_relativa*100:.0f}%")
else:
    dist_mercado = nivel_vs_merc = pos_relativa = 0.0

# ============================================================
# PREDICAO
# ============================================================
if st.button("Calcular Risco de Desligamento"):

    vp_map = {
        "Saude": 0.25, "Comercial": 0.22, "Tecnologia": 0.20,
        "Gente e Juridico": 0.12, "Financeiro": 0.10,
        "Presidencia": 0.05, "Produto": 0.06
    }

    row = {
        'Salario'          : float(salario),
        _T                 : float(tempo),
        'Idade'            : float(idade),
        'Dist_Mercado'     : float(dist_mercado),
        'Nivel_vs_Mercado' : float(nivel_vs_merc),
        'Posicao_Relativa' : float(pos_relativa),
        'VP'               : vp_map.get(vp_sel, 0.15),
        'Sexo'             : sexo,
        'Estado_Civil'     : civil,
        'Vinculo'          : vinculo,
        'UF'               : uf,
        'Nivel'            : nivel,
        'Cargo'            : cargo,
        'Escolaridade'     : escol,
        'Centro de custo'  : 'OUTROS',
        'Contrato'         : contrato,
    }

    try:
        prob = prever(row)
        pct  = prob * 100

        if pct >= 65:
            st.error(f"RISCO ALTO: {pct:.1f}%")
        elif pct >= 40:
            st.warning(f"RISCO MEDIO: {pct:.1f}%")
        else:
            st.success(f"RISCO BAIXO: {pct:.1f}%")

        st.metric("Probabilidade de desligamento", f"{pct:.1f}%")

    except Exception as e:
        st.error(f"Erro na previsao: {e}")
        st.write("Colunas geradas:", list(pd.DataFrame([row]).pipe(feature_engineering).pipe(encoding_perfil).pipe(renomear_para_modelo).columns))
        st.write("Colunas esperadas pelo modelo:", modelo_dict['columns'])
