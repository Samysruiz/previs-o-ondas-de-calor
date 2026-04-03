import streamlit as st
import sqlite3
import hashlib
import os
import pickle
import requests
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
from pathlib import Path

# ── CONFIGURAÇÃO DA PÁGINA ────────────────────────────────────────
st.set_page_config(
    page_title="Alerta Onda de Calor",
    page_icon="🌡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CAMINHOS ──────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DB_PATH     = BASE_DIR / "data" / "app.db"
MODEL_PATH  = BASE_DIR / "data" / "modelo_xgboost_v2.pkl"
LTM_PATH    = BASE_DIR / "data" / "ltm_calculado_1991_2020_silver.nc"
PARQUET_PATH = BASE_DIR / "data" / "dados_modelo_PR_v2.parquet"
TMAX_PATH   = BASE_DIR / "data" / "tmax_atual.nc"
os.makedirs(BASE_DIR / "data", exist_ok=True)

# ── BANCO DE DADOS ────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            senha TEXT NOT NULL,
            criado_em TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS propriedades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id INTEGER NOT NULL,
            nome_prop TEXT NOT NULL,
            municipio TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            cultura TEXT NOT NULL,
            telefone TEXT,
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id)
        );
        CREATE TABLE IF NOT EXISTS historico_alertas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            propriedade_id INTEGER NOT NULL,
            data_alerta TEXT NOT NULL,
            data_previsao TEXT NOT NULL,
            nivel TEXT NOT NULL,
            probabilidade REAL,
            tmax REAL,
            anomalia REAL,
            acao TEXT,
            FOREIGN KEY (propriedade_id) REFERENCES propriedades(id)
        );
    """)
    con.commit()
    con.close()

def hash_senha(senha):
    return hashlib.sha256(senha.encode()).hexdigest()

def registrar_usuario(nome, email, senha):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute("INSERT INTO usuarios (nome, email, senha) VALUES (?,?,?)",
                    (nome, email, hash_senha(senha)))
        con.commit()
        con.close()
        return True, "Conta criada com sucesso!"
    except sqlite3.IntegrityError:
        return False, "Este e-mail já está cadastrado."

def autenticar(email, senha):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT id, nome FROM usuarios WHERE email=? AND senha=?",
                      (email, hash_senha(senha))).fetchone()
    con.close()
    return row  # (id, nome) ou None

def salvar_propriedade(uid, nome, mun, lat, lon, cultura, tel):
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT INTO propriedades
        (usuario_id,nome_prop,municipio,latitude,longitude,cultura,telefone)
        VALUES (?,?,?,?,?,?,?)""", (uid, nome, mun, lat, lon, cultura, tel))
    con.commit()
    con.close()

def buscar_propriedades(uid):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id,nome_prop,municipio,latitude,longitude,cultura FROM propriedades WHERE usuario_id=?",
        (uid,)).fetchall()
    con.close()
    return rows

def salvar_alerta(pid, data_alerta, data_prev, nivel, prob, tmax, anom, acao):
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT INTO historico_alertas
        (propriedade_id,data_alerta,data_previsao,nivel,probabilidade,tmax,anomalia,acao)
        VALUES (?,?,?,?,?,?,?,?)""",
        (pid, data_alerta, data_prev, nivel, prob, tmax, anom, acao))
    con.commit()
    con.close()

def buscar_historico(pid, limite=10):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""SELECT data_previsao,nivel,probabilidade,tmax,anomalia,acao
        FROM historico_alertas WHERE propriedade_id=?
        ORDER BY data_previsao DESC LIMIT ?""", (pid, limite)).fetchall()
    con.close()
    return rows

# ── MODELO ────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def carregar_ltm():
    if not LTM_PATH.exists():
        return None
    return xr.open_dataset(LTM_PATH, engine="netcdf4")

# ── DOWNLOAD TMAX ────────────────────────────────────────────────
def baixar_tmax():
    ano = datetime.today().year
    url = f"https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmax.{ano}.nc"
    with st.spinner("Baixando dados de temperatura (NOAA)..."):
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(TMAX_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                f.write(chunk)
    return True

# ── CALCULAR ALERTA ──────────────────────────────────────────────
def calcular_alerta(lat_prop, lon_prop):
    obj = carregar_modelo()
    ds_ltm = carregar_ltm()
    if obj is None or ds_ltm is None:
        return None, "Arquivos do modelo não encontrados."

    modelo    = obj["modelo"]
    threshold = obj["threshold"]
    features  = obj["features"]

    # Baixar TMAX se não existir ou for antigo
    if not TMAX_PATH.exists() or \
       (datetime.now() - datetime.fromtimestamp(TMAX_PATH.stat().st_mtime)).seconds > 3600:
        try:
            baixar_tmax()
        except Exception as e:
            return None, f"Erro ao baixar dados NOAA: {e}"

    try:
        # Converter lon para 0-360
        lon_360 = lon_prop % 360

        # Bounding box ao redor do ponto (+/- 1 grau)
        lat_max = lat_prop + 1.0
        lat_min = lat_prop - 1.0
        lon_min = lon_360 - 1.0
        lon_max = lon_360 + 1.0

        hoje = datetime.today()
        data_hoje = hoje.strftime("%Y-%m-%d")
        data_7d   = (hoje - timedelta(days=7)).strftime("%Y-%m-%d")

        ds = xr.open_dataset(TMAX_PATH, engine="netcdf4")
        ds_local = ds.sel(
            lat=slice(lat_max, lat_min),
            lon=slice(lon_min, lon_max),
            time=slice(data_7d, data_hoje)
        )

        ultimo_dia = str(ds_local.time.values[-1])[:10]

        # LTM local
        ds_ltm_local = ds_ltm.sel(
            lat=slice(lat_max, lat_min),
            lon=slice(lon_min, lon_max)
        )

        # DataFrame
        df = ds_local["tmax"].to_dataframe().reset_index()
        df = df.sort_values(["lat","lon","time"]).reset_index(drop=True)
        df["dayofyear"] = pd.to_datetime(df["time"]).dt.dayofyear

        # Anomalia
        ltm_df = ds_ltm_local["tmax_ltm"].to_dataframe().reset_index()
        df = df.merge(ltm_df[["lat","lon","dayofyear","tmax_ltm"]],
                      on=["lat","lon","dayofyear"], how="left")
        df["tmax_anomaly"] = df["tmax"] - df["tmax_ltm"]

        # Umidade do parquet histórico
        rh_medio = 65.0  # fallback razoável
        dew_medio = df["tmax"].mean() - 5.0
        if PARQUET_PATH.exists():
            try:
                df_hist = pd.read_parquet(PARQUET_PATH,
                    columns=["time","lat","lon","relative_humidity","dewpoint_2m"])
                df_hist["time"] = pd.to_datetime(df_hist["time"]).dt.normalize()
                df_hist["lon360"] = df_hist["lon"] % 360
                local = df_hist[
                    (df_hist["lat"].between(lat_min, lat_max)) &
                    (df_hist["lon360"].between(lon_min, lon_max))
                ]
                if len(local) > 0:
                    ult = local.sort_values("time").tail(30)
                    rh_medio  = ult["relative_humidity"].mean()
                    dew_medio = ult["dewpoint_2m"].mean()
            except:
                pass

        df["relative_humidity"] = rh_medio
        df["dewpoint_2m"]       = dew_medio
        df["heat_index"]        = df["tmax"]

        # Lags
        def lag_pt(d, col, n):
            return d.groupby(["lat","lon"])[col].shift(n)

        df["tmax_anomaly_lag_1d"]   = lag_pt(df,"tmax_anomaly",1)
        df["tmax_anomaly_lag_3d"]   = lag_pt(df,"tmax_anomaly",3)
        df["tmax_anomaly_lag_7d"]   = lag_pt(df,"tmax_anomaly",7)
        df["precip"]                = 0.0
        df["precip_lag_1d"]         = 0.0
        df["precip_lag_3d"]         = 0.0
        df["precip_sum_7d"]         = 0.0
        df["chuva_significativa_ontem"] = 0
        df["dewpoint_lag_1d"]       = dew_medio
        df["rh_lag_1d"]             = rh_medio
        df["thermal_stress"]        = df["tmax_anomaly"] * (rh_medio / 100.0)
        df["thermal_stress_lag_1d"] = lag_pt(df,"thermal_stress",1)
        df["doy_sin"] = np.sin(2*np.pi*df["dayofyear"]/365.25)
        df["doy_cos"] = np.cos(2*np.pi*df["dayofyear"]/365.25)
        df["ONI"] = df["ONI_lag_1m"] = df["ONI_lag_3m"] = 0.0

        df_hoje = df[pd.to_datetime(df["time"]).dt.date ==
                     pd.to_datetime(ultimo_dia).date()].copy()
        df_hoje = df_hoje.dropna(subset=["tmax_anomaly_lag_1d"])

        if len(df_hoje) == 0:
            return None, "Sem dados suficientes para este ponto."

        X = df_hoje[features].fillna(0)
        proba  = modelo.predict_proba(X)[:,1].mean()
        tmax   = float(df_hoje["tmax"].mean())
        anom   = float(df_hoje["tmax_anomaly"].mean())

        # Semáforo
        if proba < 0.40:
            cor, nivel = "🟢", "SEM RISCO"
            acao = "Irrigação normal. Sem necessidade de ação especial."
        elif proba < 0.65:
            cor, nivel = "🟡", "ATENÇÃO"
            acao = "Monitore as temperaturas. Considere irrigação preventiva."
        elif proba < 0.80:
            cor, nivel = "🟠", "INÍCIO PROVÁVEL DE ONDA DE CALOR"
            acao = "Irrigue hoje à noite. Antecipe colheita de culturas sensíveis."
        else:
            cor, nivel = "🔴", "ONDA DE CALOR EM ANDAMENTO"
            acao = "Irrigação emergencial. Proteja mudas. Antecipe operações para a manhã."

        amanha = (pd.to_datetime(ultimo_dia) + timedelta(days=1)).strftime("%d/%m/%Y")

        resultado = {
            "cor": cor, "nivel": nivel, "acao": acao,
            "probabilidade": proba,
            "tmax": tmax, "anomalia": anom,
            "data_base": ultimo_dia,
            "data_previsao": amanha,
            "threshold": threshold,
        }
        return resultado, None

    except Exception as e:
        return None, str(e)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .alerta-box {
        border-radius: 16px; padding: 24px 28px;
        margin: 16px 0; color: white;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .verde  { background: linear-gradient(135deg,#27ae60,#2ecc71); }
    .amarelo{ background: linear-gradient(135deg,#f39c12,#f1c40f); color: #333 !important; }
    .laranja{ background: linear-gradient(135deg,#e67e22,#f39c12); }
    .vermelho{background: linear-gradient(135deg,#c0392b,#e74c3c); }
    .nivel  { font-size: 1.5rem; font-weight: 800; margin-bottom: 4px; }
    .acao   { font-size: 1rem; margin-top: 8px; opacity: 0.95; }
    .metric-row { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
    .metric-card {
        background: rgba(255,255,255,0.2); border-radius: 10px;
        padding: 8px 16px; text-align: center; flex: 1; min-width: 80px;
    }
    .metric-val { font-size: 1.3rem; font-weight: 700; }
    .metric-lbl { font-size: 0.75rem; opacity: 0.85; }
    h1 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# ── ESTADO DE SESSÃO ──────────────────────────────────────────────
if "usuario_id" not in st.session_state:
    st.session_state.usuario_id = None
if "usuario_nome" not in st.session_state:
    st.session_state.usuario_nome = None
if "pagina" not in st.session_state:
    st.session_state.pagina = "login"

init_db()

# ══════════════════════════════════════════════════════════════════
# TELA DE LOGIN / CADASTRO
# ══════════════════════════════════════════════════════════════════
def tela_login():
    st.markdown("## 🌡️ Alerta de Onda de Calor")
    st.markdown("#### Sistema de alerta para agricultores do Paraná")
    st.markdown("---")

    aba = st.tabs(["Entrar", "Criar conta"])

    with aba[0]:
        with st.form("form_login"):
            email = st.text_input("E-mail")
            senha = st.text_input("Senha", type="password")
            btn   = st.form_submit_button("Entrar", use_container_width=True)
            if btn:
                row = autenticar(email, senha)
                if row:
                    st.session_state.usuario_id   = row[0]
                    st.session_state.usuario_nome = row[1]
                    st.session_state.pagina       = "inicio"
                    st.rerun()
                else:
                    st.error("E-mail ou senha incorretos.")

    with aba[1]:
        with st.form("form_registro"):
            nome  = st.text_input("Seu nome")
            email = st.text_input("E-mail")
            senha = st.text_input("Senha", type="password")
            senha2= st.text_input("Confirmar senha", type="password")
            btn   = st.form_submit_button("Criar conta", use_container_width=True)
            if btn:
                if senha != senha2:
                    st.error("As senhas não conferem.")
                elif len(senha) < 6:
                    st.error("Senha deve ter pelo menos 6 caracteres.")
                else:
                    ok, msg = registrar_usuario(nome, email, senha)
                    if ok:
                        st.success(msg + " Faça login.")
                    else:
                        st.error(msg)

# ══════════════════════════════════════════════════════════════════
# TELA PRINCIPAL
# ══════════════════════════════════════════════════════════════════
def tela_inicio():
    uid = st.session_state.usuario_id

    # Cabeçalho
    col1, col2 = st.columns([4,1])
    with col1:
        st.markdown(f"### 🌡️ Olá, {st.session_state.usuario_nome}!")
    with col2:
        if st.button("Sair"):
            st.session_state.usuario_id   = None
            st.session_state.usuario_nome = None
            st.session_state.pagina       = "login"
            st.rerun()

    st.markdown("---")

    props = buscar_propriedades(uid)

    # ── SEM PROPRIEDADE: CADASTRAR ────────────────────────────────
    if not props:
        st.info("👋 Bem-vindo! Cadastre sua propriedade para receber alertas.")
        tela_cadastro_propriedade(uid)
        return

    # ── SELETOR DE PROPRIEDADE ────────────────────────────────────
    nomes = [f"{p[1]} — {p[2]}" for p in props]
    escolha = st.selectbox("Propriedade", nomes)
    idx = nomes.index(escolha)
    prop = props[idx]  # (id, nome, mun, lat, lon, cultura)
    pid, nome_prop, mun, lat, lon, cultura = prop

    col_a, col_b = st.columns(2)
    col_a.metric("📍 Município", mun)
    col_b.metric("🌱 Cultura", cultura)
    col_lat, col_lon = st.columns(2)
    col_lat.metric("Latitude", f"{lat:.2f}°")
    col_lon.metric("Longitude", f"{lon:.2f}°")

    st.markdown("---")

    # ── VERIFICAÇÃO DOS ARQUIVOS DO MODELO ────────────────────────
    arquivos_ok = MODEL_PATH.exists() and LTM_PATH.exists()
    if not arquivos_ok:
        st.warning("⚠️ Arquivos do modelo não encontrados.")
        st.markdown("""
        Faça o upload dos arquivos do modelo abaixo.
        Você precisa de:
        - `modelo_xgboost_v2.pkl`
        - `ltm_calculado_1991_2020_silver.nc`
        - `dados_modelo_PR_v2.parquet` *(opcional, melhora a umidade)*
        """)
        up_modelo  = st.file_uploader("modelo_xgboost_v2.pkl", type="pkl")
        up_ltm     = st.file_uploader("ltm_calculado_1991_2020_silver.nc", type="nc")
        up_parquet = st.file_uploader("dados_modelo_PR_v2.parquet (opcional)", type="parquet")
        if up_modelo:
            with open(MODEL_PATH, "wb") as f: f.write(up_modelo.read())
            st.success("Modelo carregado!")
        if up_ltm:
            with open(LTM_PATH, "wb") as f: f.write(up_ltm.read())
            st.success("LTM carregado!")
        if up_parquet:
            with open(PARQUET_PATH, "wb") as f: f.write(up_parquet.read())
            st.success("Histórico carregado!")
        return

    # ── BOTÃO GERAR ALERTA ────────────────────────────────────────
    if st.button("🔍 Gerar Alerta para Amanhã", use_container_width=True, type="primary"):
        with st.spinner("Calculando alerta..."):
            res, erro = calcular_alerta(lat, lon)

        if erro:
            st.error(f"Erro: {erro}")
        else:
            # Cor do box
            cor_map = {
                "SEM RISCO": "verde",
                "ATENÇÃO": "amarelo",
                "INÍCIO PROVÁVEL DE ONDA DE CALOR": "laranja",
                "ONDA DE CALOR EM ANDAMENTO": "vermelho",
            }
            css_cor = cor_map.get(res["nivel"], "verde")
            txt_cor = "#333" if css_cor == "amarelo" else "white"

            st.markdown(f"""
            <div class="alerta-box {css_cor}" style="color:{txt_cor}">
                <div style="font-size:0.85rem;opacity:0.8">
                    Alerta para {res['data_previsao']} — dados de {res['data_base']}
                </div>
                <div class="nivel">{res['cor']} {res['nivel']}</div>
                <div class="acao">📋 {res['acao']}</div>
                <div class="metric-row">
                    <div class="metric-card" style="color:{txt_cor}">
                        <div class="metric-val">{res['probabilidade']*100:.1f}%</div>
                        <div class="metric-lbl">Probabilidade</div>
                    </div>
                    <div class="metric-card" style="color:{txt_cor}">
                        <div class="metric-val">{res['tmax']:.1f}°C</div>
                        <div class="metric-lbl">TMAX hoje</div>
                    </div>
                    <div class="metric-card" style="color:{txt_cor}">
                        <div class="metric-val">{res['anomalia']:+.1f}°C</div>
                        <div class="metric-lbl">Anomalia</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Salvar no histórico
            salvar_alerta(
                pid, datetime.today().strftime("%Y-%m-%d"),
                res["data_previsao"], res["nivel"],
                res["probabilidade"], res["tmax"],
                res["anomalia"], res["acao"]
            )

    # ── HISTÓRICO ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📅 Histórico de alertas")

    hist = buscar_historico(pid)
    if not hist:
        st.info("Nenhum alerta gerado ainda. Clique em 'Gerar Alerta' para começar.")
    else:
        emoji_map = {
            "SEM RISCO": "🟢",
            "ATENÇÃO": "🟡",
            "INÍCIO PROVÁVEL DE ONDA DE CALOR": "🟠",
            "ONDA DE CALOR EM ANDAMENTO": "🔴",
        }
        df_hist = pd.DataFrame(hist, columns=[
            "Data","Nível","Prob.(%)","TMAX(°C)","Anomalia(°C)","Ação"])
        df_hist["Prob.(%)"]    = (df_hist["Prob.(%)"] * 100).round(1)
        df_hist["TMAX(°C)"]   = df_hist["TMAX(°C)"].round(1)
        df_hist["Anomalia(°C)"] = df_hist["Anomalia(°C)"].round(1)
        df_hist["Nível"] = df_hist["Nível"].apply(
            lambda x: f"{emoji_map.get(x,'⚪')} {x}")
        df_hist = df_hist.drop(columns=["Ação"])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    # ── NOVA PROPRIEDADE ─────────────────────────────────────────
    st.markdown("---")
    with st.expander("➕ Cadastrar nova propriedade"):
        tela_cadastro_propriedade(uid)


# ══════════════════════════════════════════════════════════════════
# FORMULÁRIO DE CADASTRO DE PROPRIEDADE
# ══════════════════════════════════════════════════════════════════
def tela_cadastro_propriedade(uid):
    municipios_pr = [
        "Toledo","Cascavel","Foz do Iguaçu","Londrina","Maringá","Ponta Grossa",
        "Curitiba","Guarapuava","Campo Mourão","Umuarama","Francisco Beltrão",
        "Pato Branco","Cornélio Procópio","Apucarana","Paranavaí","Outro"
    ]
    culturas = ["Soja","Milho","Trigo","Feijão","Cana-de-açúcar","Mandioca","Outro"]

    with st.form("form_prop", clear_on_submit=True):
        nome  = st.text_input("Nome da propriedade", placeholder="Ex: Sítio Boa Esperança")
        mun   = st.selectbox("Município", municipios_pr)
        col1, col2 = st.columns(2)
        lat   = col1.number_input("Latitude", value=-24.7, min_value=-27.5, max_value=-22.0,
                                   step=0.1, format="%.4f",
                                   help="Latitude da propriedade (entre -22.0 e -27.5)")
        lon   = col2.number_input("Longitude", value=-53.7, min_value=-55.0, max_value=-48.0,
                                   step=0.1, format="%.4f",
                                   help="Longitude da propriedade (entre -55.0 e -48.0)")
        cultura = st.selectbox("Cultura principal", culturas)
        tel   = st.text_input("Telefone (WhatsApp)", placeholder="(45) 99999-9999")

        st.caption("💡 A localização deve estar dentro do Paraná.")

        btn = st.form_submit_button("Salvar propriedade", use_container_width=True)
        if btn:
            if not nome:
                st.error("Informe o nome da propriedade.")
            else:
                salvar_propriedade(uid, nome, mun, lat, lon, cultura, tel)
                st.success(f"Propriedade '{nome}' cadastrada com sucesso!")
                st.rerun()


# ══════════════════════════════════════════════════════════════════
# ROTEAMENTO
# ══════════════════════════════════════════════════════════════════
if st.session_state.usuario_id is None:
    tela_login()
else:
    tela_inicio()
