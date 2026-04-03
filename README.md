# 🌡️ Sistema de Alerta de Onda de Calor — Paraná

Aplicativo web para agricultores do Paraná receberem alertas de onda de calor com **um dia de antecedência**, baseado em modelo XGBoost treinado com 34 anos de dados climáticos (1991–2025).

## ✨ Funcionalidades

- **Login seguro** — cadastro único por e-mail e senha
- **Cadastro de propriedade** — nome, município, localização, cultura principal
- **Alerta diário** — 4 níveis de severidade com recomendação de ação
- **Histórico** — registro de todos os alertas gerados
- **Download automático** de dados NOAA atualizados

## 🚦 Níveis de Alerta

| Nível | Condição | Ação recomendada |
|-------|----------|-----------------|
| 🟢 Sem risco | Prob. < 40% | Irrigação normal |
| 🟡 Atenção | Prob. 40–65% | Irrigação preventiva |
| 🟠 Início provável | Prob. 65–80% | Irrigar hoje à noite |
| 🔴 Onda em andamento | Prob. ≥ 80% | Ação emergencial |

## 🚀 Como rodar localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/SEU_USUARIO/alerta-onda-calor.git
cd alerta-onda-calor

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar o app
streamlit run app.py
```

## ☁️ Deploy no Streamlit Cloud (grátis)

1. Suba o código para o GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório GitHub
4. Selecione `app.py` como arquivo principal
5. Clique em **Deploy**

## 📁 Arquivos do modelo (necessários)

Após o deploy, faça o upload dos arquivos na interface do app:

| Arquivo | Descrição |
|---------|-----------|
| `modelo_xgboost_v2.pkl` | Modelo treinado (gerado pelo Gold v2) |
| `ltm_calculado_1991_2020_silver.nc` | Normal climatológica 1991–2020 |
| `dados_modelo_PR_v2.parquet` | Histórico (opcional, melhora umidade) |

Esses arquivos ficam na pasta `data/` e são carregados automaticamente na sessão.

## 🏗️ Arquitetura

```
app.py              ← Aplicativo Streamlit principal
requirements.txt    ← Dependências Python
.streamlit/
  config.toml       ← Tema e configurações
data/               ← Criada automaticamente
  app.db            ← Banco SQLite (usuários, propriedades, histórico)
  modelo_*.pkl      ← Modelo XGBoost (upload manual)
  ltm_*.nc          ← Normal climatológica (upload manual)
  tmax_atual.nc     ← Baixado automaticamente da NOAA
```

## 🔬 Sobre o modelo

- **Algoritmo:** XGBoost com scale_pos_weight=54,76
- **Recall:** 91,3% — 9 em cada 10 ondas de calor detectadas
- **F1-Score:** 81,3%
- **Threshold:** 0,80
- **Dados:** NOAA CPC TMAX + ERA5 Dewpoint + NOAA Precipitation + ONI (1991–2025)
- **Período de teste:** 2022–2025

## 👩‍💻 Autora

**Samara Ruiz Silva**  
Curso de Inteligência Artificial — Faculdade Donaduzzi — BIOPARK  
Toledo, PR — samysruiz@gmail.com

---

> *Este sistema gera alertas baseados em dados históricos e modelo estatístico. Recomenda-se uso conjunto com informações locais e orientação técnica agrícola.*
