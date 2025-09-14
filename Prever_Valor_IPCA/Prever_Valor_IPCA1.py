#!/usr/bin/env python
# coding: utf-8

# # 🤖 Ferramenta de Previsão de IPCA com Macnine Learning
#     
#    **Objetivo**:
#         Esta aplicação realiza a correção monetária de valores com base no
#         histórico do IPCA e projeta valores futuros do índice utilizando um
#         modelo de Machine Learning (SARIMA). A ferramenta foi desenvolvida para
#         apoiar análises orçamentárias e decisões financeiras estratégicas,
#         oferecendo uma visão prospectiva da inflação.
# 
#    **Autor**:
#         Jonathas Gomes
# 
#    **Versão:**
#         1.0 (Setembro de 2025)
# 
#    **Principais Tecnologias:**
#         - Python 3
#         - Streamlit (para a interface web interativa)
#         - Pandas (para manipulação de dados)
#         - Pmdarima (para o modelo auto-SARIMA)
#         - Scikit-learn (para métricas de performance)
#         - Plotly (para visualização de gráficos)
# 
#    **Fonte de Dados:**
#         API de Séries Temporais do Sistema Gerenciador de Séries Temporais
#         (SGS) do Banco Central do Brasil (BCB). Série: 433 (IPCA).

# ### 1. Importando bibliotecas essenciais.
# 

# In[ ]:


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from bcb import sgs
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import io


# ### 2. BACK-END: LÓGICA DE DADOS E MACHINE LEARNING

# In[ ]:


@st.cache_data
def fetch_ipca_data(start_date='2000-01-01'):
    """Busca os dados do IPCA do SGS do Banco Central."""
    try:
        ipca_df = sgs.get({'IPCA': 433}, start=start_date)
        ipca_df.index.name = 'Data'
        return ipca_df
    except Exception as e:
        st.error(f"Erro ao buscar dados do BCB: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_trained_model(_ipca_data):
    """Treina o modelo SARIMA e o armazena em cache para performance."""
    predictor = IPCAPredictor(_ipca_data)
    predictor.train_and_evaluate()
    return predictor

class IPCAPredictor:
    def __init__(self, data):
        if data.empty:
            raise ValueError("O DataFrame de entrada não pode estar vazio.")
        self.data = data['IPCA']
        self.model = None
        self.performance_metrics = {}

    def train_and_evaluate(self):
        train_data = self.data[:-12]
        test_data = self.data[-12:]
        
        temp_model = pm.auto_arima(train_data, seasonal=True, m=12, 
                                   suppress_warnings=True, stepwise=True,
                                   error_action='ignore')
        
        predictions_on_test = temp_model.predict(n_periods=len(test_data))
        mae = mean_absolute_error(test_data, predictions_on_test)
        rmse = np.sqrt(mean_squared_error(test_data, predictions_on_test))
        mape = np.mean(np.abs(predictions_on_test - test_data) / np.abs(test_data)) * 100
        self.performance_metrics = {
            'MAE': f"{mae:.4f}",
            'RMSE': f"{rmse:.4f}",
            'MAPE': f"{mape:.2f}%"
        }
        
        temp_model.update(test_data)
        self.model = temp_model

    def predict_future(self, future_date):
        if self.model is None:
            raise RuntimeError("O modelo precisa ser treinado primeiro.")

        last_date = self.data.index.max()
        months_to_predict = (future_date.year - last_date.year) * 12 + (future_date.month - last_date.month)

        if months_to_predict <= 0:
            return pd.DataFrame(), pd.DataFrame()

        future_preds, conf_int = self.model.predict(n_periods=months_to_predict, return_conf_int=True)
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_to_predict, freq='MS')
        predictions_df = pd.DataFrame({'IPCA Previsto': future_preds}, index=future_dates)
        conf_int_df = pd.DataFrame(conf_int, index=future_dates, columns=['Limite Inferior', 'Limite Superior'])
        return predictions_df, conf_int_df


# ### 3. FRONT-END: INTERFACE WEB COM STREAMLIT

# In[ ]:


def main():
    st.set_page_config(page_title="Previsão de IPCA com Machine Learning", page_icon="💸", layout="wide")

    if 'resultados' not in st.session_state:
        st.session_state.resultados = None

    @st.cache_data
    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=True, sheet_name='Resultados')
        return output.getvalue()
        
    with st.sidebar:
        st.image("assets/logo-seplag.png", width=200)
        st.title("Parâmetros de Análise")
        valor_original = st.number_input("Valor a ser corrigido (R$)", min_value=0.01, value=493.72, step=10.0, format="%.2f")
        data_inicio_correcao = st.date_input("Data de Início da Correção", value=datetime(2024, 6, 1))
        data_previsao = st.date_input("Prever IPCA até (Data Futura)", value=datetime(2025, 5, 1))
        st.info("O modelo utiliza dados históricos desde 2000 para treinar e prever os valores futuros do IPCA.")
        
        # NOVO: Seção de Suporte
        st.markdown("---")
        st.subheader("Suporte e Dúvidas")
        st.markdown("Para suporte ou dúvidas, entre em contato:")
        st.markdown("📧 <jonathasmarques@seplag.mt.gov.br>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Sobre o Projeto")
        st.write("Esta aplicação utiliza um modelo de Machine Learning (SARIMA) para prever a inflação (IPCA). Desenvolvido por um Cientista de Dados para análises orçamentárias precisas.")

    st.title("🤖 Ferramenta de Previsão de IPCA com Machine Learning")
    st.markdown("### Análise Preditiva da Inflação para Decisões Orçamentárias Estratégicas")

    if st.button("📈 Realizar Previsão com IPCA", type="primary"):
        with st.spinner('Buscando dados e preparando o modelo preditivo...'):
            ipca_historico = fetch_ipca_data()
            if not ipca_historico.empty:
                predictor = get_trained_model(ipca_historico)
                previsoes, conf_interval = predictor.predict_future(data_previsao)
                
                st.session_state.resultados = {
                    "ipca_historico": ipca_historico, "previsoes": previsoes,
                    "conf_interval": conf_interval, "predictor": predictor,
                    "valor_original": valor_original, "data_inicio_correcao": data_inicio_correcao,
                    "data_previsao": data_previsao
                }
                st.rerun()

    if st.session_state.resultados:
        res = st.session_state.resultados
        ipca_historico, previsoes, conf_interval, predictor = res["ipca_historico"], res["previsoes"], res["conf_interval"], res["predictor"]
        
        st.markdown("---")
        
        col_resultados, col_correcao = st.columns(2)

        with col_resultados:
            st.subheader("Resultados da Análise Preditiva")
            last_registered_date = ipca_historico.index.max()
            label_ultimo_ipca = f"Último IPCA Registrado ({last_registered_date.strftime('%m/%Y')})"
            st.metric(label=label_ultimo_ipca, value=f"{ipca_historico['IPCA'].iloc[-1]:.2f}%", delta=f"{(ipca_historico['IPCA'].iloc[-1] - ipca_historico['IPCA'].iloc[-2]):.2f}% vs Mês Anterior")

            target_date = res['data_previsao'].replace(day=1)
            target_date_ts = pd.to_datetime(target_date)
            
            if target_date_ts in ipca_historico.index:
                valor_alvo = ipca_historico.loc[target_date_ts, 'IPCA']
                label_alvo = f"IPCA Registrado em {target_date.strftime('%m/%Y')}"
                st.metric(label=label_alvo, value=f"{valor_alvo:.2f}%")
            elif not previsoes.empty and target_date_ts in previsoes.index:
                valor_alvo = previsoes.loc[target_date_ts, 'IPCA Previsto']
                label_alvo = f"IPCA Previsto para {target_date.strftime('%m/%Y')}"
                st.metric(label=label_alvo, value=f"{valor_alvo:.2f}%")
            else:
                label_alvo = f"IPCA para {target_date.strftime('%m/%Y')}"
                st.metric(label=label_alvo, value="N/A")

            if not previsoes.empty:
                 st.metric(label=f"IPCA Acumulado Previsto (próx. 12 meses)", value=f"{((1 + previsoes['IPCA Previsto'][:12]/100).prod() - 1) * 100:.2f}%")

        with col_correcao:
            st.subheader(f"Cálculo da Correção Monetária")
            start_date = pd.to_datetime(res['data_inicio_correcao'])
            end_date = pd.to_datetime(res['data_previsao'].replace(day=1))
            historico_no_periodo = ipca_historico.loc[(ipca_historico.index >= start_date) & (ipca_historico.index <= end_date), 'IPCA']
            
            previsto_no_periodo = pd.Series(dtype='float64')
            if not previsoes.empty:
                ultimo_historico = ipca_historico.index.max()
                previsto_no_periodo = previsoes.loc[(previsoes.index > ultimo_historico) & (previsoes.index <= end_date), 'IPCA Previsto']
            
            serie_completa = pd.concat([historico_no_periodo, previsto_no_periodo])
            fator_acumulado = (1 + serie_completa / 100).prod()
            valor_corrigido = res['valor_original'] * fator_acumulado
            inflacao_acumulada_periodo = (fator_acumulado - 1) * 100
            diferenca_reais = valor_corrigido - res['valor_original']
            
            st.metric("Valor Original (R$)", f"{res['valor_original']:,.2f}")
            st.metric("Inflação Acumulada no Período", f"{inflacao_acumulada_periodo:.2f}%")
            st.metric("Fator de Correção Acumulado", f"{fator_acumulado:.4f}")
            st.metric(label="Diferença (Valor Corrigido - Original)", value=f"R$ {diferenca_reais:,.2f}", delta=f"{inflacao_acumulada_periodo:.2f}%")
            st.metric("Valor Corrigido e Projetado (R$)", f"{valor_corrigido:,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Gráfico de Projeção do IPCA")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ipca_historico.index, y=ipca_historico['IPCA'], mode='lines', name='IPCA Histórico', line=dict(color='blue')))
        if not previsoes.empty:
            # CORREÇÃO 1: Adicionando hovertemplate para rótulos de dados
            fig.add_trace(go.Scatter(
                x=previsoes.index, y=previsoes['IPCA Previsto'], 
                mode='lines', name='IPCA Previsto', line=dict(color='red', dash='dot'),
                hovertemplate = '<b>%{x|%m/%Y}</b><br>IPCA Previsto: %{y:.2f}%<extra></extra>'
            ))
            fig.add_trace(go.Scatter(x=previsoes.index.append(previsoes.index[::-1]), y=pd.concat([conf_interval['Limite Superior'], conf_interval['Limite Inferior'][::-1]]), fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Intervalo de Confiança', hoverinfo='skip'))
        
        fig.update_layout(title='IPCA Mensal: Histórico vs. Previsão', xaxis_title='Data', yaxis_title='Variação Mensal (%)', legend_title='Legenda', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver Tabela Detalhada e Fazer Download"):
            # CORREÇÃO 2: Lógica de junção das tabelas para exibição
            tabela_historica = ipca_historico.copy().rename(columns={'IPCA': 'IPCA Histórico'})
            
            if not previsoes.empty:
                tabela_previsoes = previsoes[['IPCA Previsto']]
                tabela_completa = pd.concat([tabela_historica, tabela_previsoes], axis=1)
            else:
                tabela_completa = tabela_historica

            st.dataframe(tabela_completa.tail(36).fillna(''))
            excel_data = to_excel(tabela_completa.reset_index())
            st.download_button(label="📥 Baixar dados em XLSX", data=excel_data, file_name=f"previsao_ipca_{datetime.now().strftime('%Y%m%d')}.xlsx")
            
        st.sidebar.markdown("---")
        st.sidebar.subheader("Performance do Modelo", help="Métricas baseadas na performance do modelo ao prever os últimos 12 meses de dados históricos conhecidos.")
        
        for key, value in predictor.performance_metrics.items():
            st.sidebar.text(f"{key}: {value}")
        
        st.sidebar.caption("O que essas métricas significam:")
        st.sidebar.markdown("""
        - **MAE:** O erro absoluto médio. Indica, em média, quantos pontos percentuais (p.p.) a previsão do IPCA errou.
        - **RMSE:** Similar ao MAE, mas penaliza erros maiores com mais intensidade.
        - **MAPE:** O erro percentual médio. Um MAPE de 10% significa que a previsão errou, em média, 10% do valor real do IPCA.
        """)
    
    else:
        st.info("👈 Configure os parâmetros na barra lateral e clique no botão para iniciar a análise.")

    st.markdown("---")
    st.caption("**Desenvolvido por:** Jonathas Gomes | **Objetivo:** Ferramenta de análise e projeção da inflação (IPCA) utilizando machine learning para apoiar decisões orçamentárias.")

if __name__ == '__main__':
    main()

