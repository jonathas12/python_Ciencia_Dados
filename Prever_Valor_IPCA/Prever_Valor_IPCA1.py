#!/usr/bin/env python
# coding: utf-8

# Como cientista e engenheiro de dados com vasta experiência no setor financeiro, compreendo perfeitamente a sua motivação. Sair de uma simples correção de cálculo para a criação de um modelo preditivo não apenas resolve o problema imediato, mas demonstra um valor estratégico imenso, transformando uma discussão operacional em uma análise prospectiva. É exatamente esse tipo de proatividade que diferencia um profissional de dados de alto impacto.
# 
# A ideia de usar um modelo de séries temporais como o SARIMA (ou similar) em vez de uma regressão linear simples é tecnicamente muito mais robusta para prever um índice como o IPCA, que possui características de sazonalidade, tendência e autocorrelação. A regressão linear sobre o tempo seria ingênua e provavelmente produziria resultados pouco confiáveis.
# 
# Abaixo, apresento um projeto completo, estruturado e comentado, que materializa a sua visão. Dividi o código e a explicação em seções claras, como se estivéssemos construindo um produto de dados real.

# **BACK END**

# ### 1. IMPORT DAS BIBLIOTECAS

# In[ ]:


#pip install streamlit pandas python-bcb pmdarima scikit-learn plotly openpyxl


# In[ ]:


# ==============================================================================
# 1. BACK-END: LÓGICA DE DADOS E MACHINE LEARNING
# ==============================================================================

# Importando bibliotecas essenciais
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from bcb import sgs
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import io
from fpdf import FPDF
import statsmodels.api as sm


# ### 2. CLASSE PARA BUSCA DE DADOS NO BANCO CENTRAL DO BRASIL

# In[ ]:


class BCBDataFetcher:
    """
    Classe responsável por buscar dados da série temporal do IPCA
    através da API do Banco Central do Brasil (SGS).
    """
    def __init__(self, series_code=433, start_date='2000-01-01'):
        self.series_code = series_code
        self.start_date = start_date

    def fetch_data(self):
        """
        Busca os dados do IPCA e retorna um DataFrame do pandas.
        """
        try:
            ipca_df = sgs.get({'IPCA': self.series_code}, start=self.start_date)
            ipca_df.index.name = 'Data'
            # O dado vem como variação % mensal, vamos manter assim.
            return ipca_df
        except Exception as e:
            st.error(f"Erro ao buscar dados do BCB: {e}")
            return pd.DataFrame()


# ### 3. CLASSE PARA O MODELO DE PREVISÃO

# In[ ]:


class IPCAPredictor:
    """
    Classe que encapsula o treinamento, previsão e avaliação
    do modelo de séries temporais para o IPCA.
    """
    def __init__(self, data):
        if data.empty:
            raise ValueError("O DataFrame de entrada não pode estar vazio.")
        self.data = data['IPCA']
        self.model = None
        self.train_data = None
        self.test_data = None
        self.performance_metrics = {}

    def _split_data(self, test_size=12):
        """Divide os dados em treino e teste (últimos 12 meses para teste)."""
        self.train_data = self.data[:-test_size]
        self.test_data = self.data[-test_size:]

    def train_model(self):
        """
        Treina um modelo SARIMA automaticamente com pmdarima.auto_arima.
        Esta função encontra os melhores parâmetros (p,d,q)(P,D,Q,m) para o modelo.
        """
        # O 'm=12' é crucial, pois indica a sazonalidade anual (12 meses).
        self.model = pm.auto_arima(self.data, # Usaremos todos os dados para o treino final
                                   start_p=1, start_q=1,
                                   test='adf',       # Teste de estacionariedade
                                   max_p=3, max_q=3,
                                   m=12,             # Frequência da série (mensal)
                                   d=None,           # Deixa o auto_arima encontrar o 'd'
                                   seasonal=True,
                                   start_P=0,
                                   D=1,              # Diferenciação sazonal
                                   trace=False,      # Não imprime o passo a passo
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)
        # st.sidebar.text(f"Melhores Parâmetros SARIMA: {self.model.order}, {self.model.seasonal_order}")

    def evaluate(self):
        """
        Avalia o modelo contra o conjunto de teste para gerar métricas de performance.
        """
        self._split_data()
        temp_model = pm.auto_arima(self.train_data, seasonal=True, m=12, suppress_warnings=True, stepwise=True)
        
        predictions_on_test = temp_model.predict(n_periods=len(self.test_data))
        
        mae = mean_absolute_error(self.test_data, predictions_on_test)
        rmse = np.sqrt(mean_squared_error(self.test_data, predictions_on_test))
        mape = np.mean(np.abs(predictions_on_test - self.test_data) / np.abs(self.test_data)) * 100

        self.performance_metrics = {
            'MAE (Erro Absoluto Médio)': f"{mae:.4f}",
            'RMSE (Raiz do Erro Quadrático Médio)': f"{rmse:.4f}",
            'MAPE (Erro Percentual Absoluto Médio)': f"{mape:.2f}%"
        }

    def predict_future(self, future_date):
        """
        Prevê os valores do IPCA até uma data futura.
        """
        if self.model is None:
            self.train_model() # Garante que o modelo está treinado

        last_date = self.data.index.max()
        months_to_predict = (future_date.year - last_date.year) * 12 + (future_date.month - last_date.month)

        if months_to_predict <= 0:
            return pd.DataFrame(), pd.DataFrame()

        # Fazendo a previsão
        future_preds, conf_int = self.model.predict(n_periods=months_to_predict, return_conf_int=True)

        # Criando o índice de datas para as previsões
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                     periods=months_to_predict,
                                     freq='MS') # 'MS' para o início do mês

        predictions_df = pd.DataFrame({'IPCA Previsto': future_preds}, index=future_dates)
        conf_int_df = pd.DataFrame(conf_int, index=future_dates, columns=['Limite Inferior', 'Limite Superior'])
        
        return predictions_df, conf_int_df


# ### 4. FRONT END

# In[ ]:


# ==============================================================================
# 2. FRONT-END: INTERFACE WEB COM STREAMLIT (VERSÃO FINAL E ROBUSTA)
# ==============================================================================

def main():
    # --- CONFIGURAÇÃO DA PÁGINA ---
    st.set_page_config(
        page_title="Previsão de IPCA com Machine Learning",
        page_icon="💸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- FUNÇÃO PARA CONVERTER DATAFRAME PARA XLSX EM MEMÓRIA ---
    @st.cache_data
    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=True, sheet_name='Resultados')
        processed_data = output.getvalue()
        return processed_data
        
    # --- SIDEBAR (MENU LATERAL) ---
    with st.sidebar:
        st.image("https://www.bcb.gov.br/Assets/img/logo-bcb-colorido.png", width=200)
        st.title("Parâmetros de Análise")

        valor_original = st.number_input(
            "Valor a ser corrigido (R$)", min_value=0.0, value=1000.0, step=100.0, format="%.2f"
        )
        data_inicio_correcao = st.date_input(
            "Data de Início da Correção", value=datetime(2022, 1, 1)
        )
        data_previsao = st.date_input(
            "Prever IPCA até (Data Futura)", value=datetime.now().date() + relativedelta(years=1)
        )

        st.info("O modelo utiliza dados históricos desde 2000 para treinar e prever os valores futuros do IPCA.")
        st.markdown("---")
        st.subheader("Sobre o Projeto")
        st.write("""
        Esta aplicação utiliza um modelo de Machine Learning (SARIMA) para prever a inflação (IPCA). 
        Desenvolvido por um Cientista de Dados para análises orçamentárias precisas.
        """)

    # --- TÍTULO PRINCIPAL ---
    st.title("🤖 Ferramenta de Previsão de IPCA")
    st.markdown("### Análise Preditiva da Inflação para Decisões Orçamentárias Estratégicas")

    try:
        data_fetcher = BCBDataFetcher()
        ipca_historico = data_fetcher.fetch_data()

        if ipca_historico.empty:
            st.error("Não foi possível carregar os dados. Verifique a conexão ou a API do BCB.")
            return

        predictor = IPCAPredictor(ipca_historico)
        
        with st.spinner('Avaliando performance do modelo...'):
            predictor.evaluate()
        
        with st.spinner(f'Treinando modelo e prevendo IPCA até {data_previsao.strftime("%m/%Y")}...'):
            previsoes, conf_interval = predictor.predict_future(data_previsao)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Último IPCA Registrado",
                value=f"{ipca_historico['IPCA'].iloc[-1]:.2f}%",
                delta=f"{(ipca_historico['IPCA'].iloc[-1] - ipca_historico['IPCA'].iloc[-2]):.2f}% vs Mês Anterior"
            )
        if not previsoes.empty:
            with col2:
                st.metric(
                    label=f"IPCA Previsto para {previsoes.index[0].strftime('%m/%Y')}",
                    value=f"{previsoes['IPCA Previsto'].iloc[0]:.2f}%"
                )
            with col3:
                st.metric(
                    label=f"IPCA Acumulado Previsto (próx. 12 meses)",
                    value=f"{((1 + previsoes['IPCA Previsto'][:12]/100).prod() - 1) * 100:.2f}%"
                )

        st.subheader("Visualização Histórica e Preditiva do IPCA")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ipca_historico.index, y=ipca_historico['IPCA'], mode='lines', name='IPCA Histórico (Observado)', line=dict(color='blue')
        ))

        if not previsoes.empty:
            fig.add_trace(go.Scatter(
                x=previsoes.index, y=previsoes['IPCA Previsto'], mode='lines', name='IPCA Previsto (Modelo)', line=dict(color='red', dash='dot')
            ))
            fig.add_trace(go.Scatter(
                x=previsoes.index.append(previsoes.index[::-1]),
                y=pd.concat([conf_interval['Limite Superior'], conf_interval['Limite Inferior'][::-1]]),
                fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Intervalo de Confiança (95%)'
            ))

        fig.update_layout(
            title='IPCA Mensal: Histórico vs. Previsão', xaxis_title='Data', yaxis_title='Variação Mensal (%)', legend_title='Legenda', template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"Cálculo da Correção Monetária Projetada")
        
        dados_relevantes = ipca_historico[ipca_historico.index >= pd.to_datetime(data_inicio_correcao)]
        serie_completa = dados_relevantes['IPCA']
        if not previsoes.empty:
             serie_completa = pd.concat([dados_relevantes['IPCA'], previsoes['IPCA Previsto']])
        
        fator_acumulado = (1 + serie_completa / 100).prod()
        valor_corrigido = valor_original * fator_acumulado
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Valor Original (R$)", f"{valor_original:,.2f}")
        col_res2.metric("Valor Corrigido e Projetado (R$)", f"{valor_corrigido:,.2f}")
        col_res3.metric("Fator de Correção Acumulado", f"{fator_acumulado:.4f}")
        
        with st.expander("Ver Tabela Detalhada e Fazer Download"):
            tabela_resultados = pd.DataFrame(ipca_historico).copy()
            tabela_resultados.rename(columns={'IPCA': 'IPCA Histórico'}, inplace=True)
            
            tabela_completa = tabela_resultados
            
            # AQUI ESTÁ A CORREÇÃO CRÍTICA: Verificamos se há previsões antes de tentar juntá-las
            if not previsoes.empty:
                tabela_completa_previsoes = pd.concat([previsoes, conf_interval], axis=1)
                tabela_completa_previsoes.rename(columns={'IPCA Previsto': 'IPCA Previsto'}, inplace=True, errors='ignore')
                tabela_completa = pd.concat([tabela_resultados, tabela_completa_previsoes], axis=1)

            tabela_completa.index.name = 'Data'
            st.dataframe(tabela_completa.tail(36))

            excel_data = to_excel(tabela_completa.reset_index())
            st.download_button(
                label="📥 Baixar dados em XLSX",
                data=excel_data,
                file_name=f"previsao_ipca_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        st.sidebar.markdown("---")
        st.sidebar.subheader("Performance do Modelo")
        st.sidebar.write("Métricas calculadas sobre um conjunto de teste (últimos 12 meses não vistos no treino):")
        for key, value in predictor.performance_metrics.items():
            st.sidebar.text(f"{key}: {value}")

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante a execução: {e}")
        st.exception(e)

