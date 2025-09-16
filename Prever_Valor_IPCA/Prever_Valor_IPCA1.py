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
#         2.0 (Setembro de 2025)
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import io
from fpdf import FPDF
from pathlib import Path
import statsmodels.api as sm


# ### 2. BACK-END: LÓGICA DE DADOS E MACHINE LEARNING

# In[ ]:


@st.cache_data
def fetch_ipca_data(start_date='2000-01-01'):
    try:
        ipca_df = sgs.get({'IPCA': 433}, start=start_date)
        ipca_df.index.name = 'Data'
        return ipca_df
    except Exception as e:
        st.error(f"Erro ao buscar dados do BCB: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_trained_model(_ipca_data):
    predictor = IPCAPredictor(_ipca_data)
    predictor.train_and_evaluate()
    return predictor

class IPCAPredictor:
    def __init__(self, data):
        self.data = data['IPCA']
        self.model_results = None
        self.performance_metrics = {}
        self.order = (1, 0, 0)
        self.seasonal_order = (1, 0, 1, 12)

    def train_and_evaluate(self):
        train_data = self.data[:-12]
        test_data = self.data[-12:]
        
        model = SARIMAX(train_data, order=self.order, seasonal_order=self.seasonal_order)
        model_fit = model.fit(disp=False)
        
        predictions_on_test = model_fit.forecast(steps=len(test_data))
        mae = mean_absolute_error(test_data, predictions_on_test)
        mape = mean_absolute_percentage_error(test_data, predictions_on_test) * 100

        self.performance_metrics = {
            'MAE': f"{mae:.4f}",
            'MAPE': f"{mape:.2f}%"
        }
        
        final_model = SARIMAX(self.data, order=self.order, seasonal_order=self.seasonal_order)
        self.model_results = final_model.fit(disp=False)

    def predict_future(self, future_date):
        if self.model_results is None:
            raise RuntimeError("O modelo precisa ser treinado primeiro.")

        last_date = self.data.index.max()
        steps_to_predict = (future_date.year - last_date.year) * 12 + (future_date.month - last_date.month)

        if steps_to_predict <= 0: return pd.DataFrame(), pd.DataFrame()

        forecast = self.model_results.get_forecast(steps=steps_to_predict)
        predictions_df = forecast.predicted_mean.to_frame('IPCA Previsto')
        conf_int_df = forecast.conf_int()
        conf_int_df.columns = ['Limite Inferior', 'Limite Superior']
        
        return predictions_df, conf_int_df

# ==============================================================================
# 3. FUNÇÕES AUXILIARES (DOWNLOADS)
# ==============================================================================

class PDF(FPDF):
    def header(self):
        #self.image('assets/logo-seplag.png', x=10, y=8, w=33)
        self.set_font('Helvetica', 'B', 20)
        self.cell(0, 10, 'Relatório de Projeção de IPCA', 0, 1, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
        self.cell(0, 10, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 0, 0, 'R')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def metric(self, label, value):
        self.set_font('Helvetica', 'B', 10)
        self.cell(95, 8, label, 1, 0, 'L')
        self.set_font('Helvetica', '', 10)
        self.cell(95, 8, str(value), 1, 1, 'R')

def gerar_relatorio_pdf(res, predictor, fig):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.chapter_title('1. Parâmetros da Análise')
    pdf.metric("Valor Original:", f"R$ {res['valor_original']:,.2f}")
    pdf.metric("Data de Início da Correção:", res['data_inicio_correcao'].strftime('%m/%Y'))
    pdf.metric("Data Final da Projeção:", res['data_previsao'].strftime('%m/%Y'))
    pdf.ln(5)

    pdf.chapter_title('2. Resumo da Correção Monetária')
    pdf.metric("Valor Corrigido e Projetado:", f"R$ {res['valor_corrigido']:,.2f}")
    pdf.metric("Inflação Acumulada no Período:", f"{res['inflacao_acumulada_periodo']:.2f}%")
    pdf.metric("Fator de Correção Acumulado:", f"{res['fator_acumulado']:.4f}")
    pdf.metric("Diferença (R$):", f"R$ {res['diferenca_reais']:,.2f}")
    pdf.ln(5)

    pdf.chapter_title('3. Gráfico de Projeção do IPCA')
    fig_copy = go.Figure(fig)
    fig_copy.update_xaxes(tickformat="%b %Y", dtick="M3")
    img_bytes = fig_copy.to_image(format="png", width=1000, height=500, scale=2)
    pdf.image(io.BytesIO(img_bytes), x=10, w=pdf.w - 20)
    pdf.ln(5)
    
    pdf.chapter_title('4. Análise de Performance do Modelo')
    pdf.chapter_body("Métricas baseadas na performance do modelo ao prever os últimos 12 meses de dados históricos conhecidos.")
    pdf.metric("Erro Médio Absoluto (MAE):", predictor.performance_metrics.get('MAE', 'N/A'))
    pdf.metric("Erro Percentual Médio (MAPE):", predictor.performance_metrics.get('MAPE', 'N/A'))
    pdf.ln(5)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 5, "Diagnóstico Estatístico (Testes de Resíduos)", 0, 1)
    
    diagnostics = predictor.model_results.summary().tables[2]
    prob_q = diagnostics.data[1][1].strip()
    prob_h = diagnostics.data[3][1].strip()

    pdf.metric("Teste Ljung-Box - Prob(Q):", f"{prob_q} (> 0.05 é bom)")
    pdf.metric("Teste de Heterocedasticidade - Prob(H):", f"{prob_h} (> 0.05 é bom)")
    pdf.ln(10)

    return bytes(pdf.output())


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
        st.markdown(f"""
        <style>
            [data-testid="stSidebar"] [data-testid="stImage"]{{
                text-align: center; display: block; margin-left: auto; margin-right: auto; width: 100%;
            }}
        </style>
        """, unsafe_allow_html=True)
        img_path = Path(__file__).parent / "assets" / "logo-seplag.png"
        st.image(str(img_path), width=150)
        st.title("Parâmetros")
        valor_original = st.number_input("Valor a ser corrigido (R$)", min_value=0.01, step=10.0, format="%.2f", value=493.72)
        data_inicio_correcao = st.date_input("Data de Início da Correção", value=datetime(2024, 6, 1))
        data_previsao = st.date_input("Prever IPCA até (Data Futura)", value=datetime(2025, 5, 1))
        
        st.markdown("---")
        if st.button("🧹 Limpar Análise"):
            st.session_state.resultados = None
            st.rerun()

        st.info("O modelo utiliza dados históricos desde 2000 para treinar e prever os valores futuros do IPCA.")
        st.markdown("---")
        st.subheader("Suporte e Dúvidas")
        st.markdown("📧 <jonathasmarques@seplag.mt.gov.br>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Sobre o Projeto")
        st.write("Esta aplicação utiliza um modelo de Machine Learning (SARIMA: Statsmodels) para prever a inflação (IPCA).")

    st.title("🤖 Ferramenta de Previsão de IPCA com Machine Learning.")
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
            
            res.update({
                'valor_corrigido': valor_corrigido, 'inflacao_acumulada_periodo': inflacao_acumulada_periodo,
                'fator_acumulado': fator_acumulado, 'diferenca_reais': diferenca_reais
            })
            
            st.metric("Valor Original (R$)", f"{res['valor_original']:,.2f}")
            st.metric("Inflação Acumulada no Período", f"{inflacao_acumulada_periodo:.2f}%")
            st.metric(label="Diferença (Valor Corrigido - Original)", value=f"R$ {diferenca_reais:,.2f}", delta=f"{inflacao_acumulada_periodo:.2f}%")
            st.metric("Valor Corrigido e Projetado (R$)", f"{valor_corrigido:,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Gráfico de Projeção do IPCA")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ipca_historico.index, y=ipca_historico['IPCA'], mode='lines', name='IPCA Histórico', line=dict(color='blue')))
        if not previsoes.empty:
            fig.add_trace(go.Scatter(x=previsoes.index, y=previsoes['IPCA Previsto'], mode='lines', name='IPCA Previsto', line=dict(color='red', dash='dot'), hovertemplate = '<b>%{x|%m/%Y}</b><br>IPCA Previsto: %{y:.2f}%<extra></extra>'))
            fig.add_trace(go.Scatter(x=previsoes.index.append(previsoes.index[::-1]), y=pd.concat([conf_interval['Limite Superior'], conf_interval['Limite Inferior'][::-1]]), fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Intervalo de Confiança', hoverinfo='skip'))
        fig.update_layout(title='IPCA Mensal: Histórico vs. Previsão', xaxis_title='Data', yaxis_title='Variação Mensal (%)', legend_title='Legenda', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Downloads")
        tabela_historica = ipca_historico.copy().rename(columns={'IPCA': 'IPCA Histórico'})
        if not previsoes.empty:
            tabela_previsoes = previsoes[['IPCA Previsto']]
            tabela_completa = pd.concat([tabela_historica, tabela_previsoes], axis=1)
        else:
            tabela_completa = tabela_historica

        col_btn1, col_btn2 = st.columns(2)
        excel_data = to_excel(tabela_completa.reset_index())
        col_btn1.download_button(label="📥 Baixar Tabela Completa (XLSX)", data=excel_data, file_name="relatorio_completo_ipca.xlsx", use_container_width=True)
        pdf_data = gerar_relatorio_pdf(res, predictor, fig)
        col_btn2.download_button(label="📄 Baixar Relatório Técnico (PDF)", data=pdf_data, file_name="relatorio_projecao_ipca.pdf", use_container_width=True)

        st.markdown("---")
        st.subheader("Análise de Performance do Modelo")
        st.caption("As métricas e testes abaixo avaliam a qualidade do ajuste do modelo aos dados históricos.")

        diagnostics = predictor.model_results.summary().tables[2]
        prob_q = float(diagnostics.data[1][1].strip())
        prob_h = float(diagnostics.data[3][1].strip())

        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
        col_perf1.metric(label="Erro Médio Absoluto (MAE)", value=predictor.performance_metrics.get('MAE', 'N/A'), help="Indica, em média, quantos pontos percentuais (p.p.) a previsão errou. Quanto menor, melhor.")
        col_perf2.metric(label="Erro Percentual Médio (MAPE)", value=predictor.performance_metrics.get('MAPE', 'N/A'), help="O erro percentual médio. Pode ser enganoso para valores de IPCA próximos de zero.")
        col_perf3.metric(label="Teste Ljung-Box - Prob(Q)", value=f"{prob_q:.2f}", help="Testa se os erros são aleatórios. Um valor > 0.05 (ótimo) indica que o modelo capturou bem os padrões.")
        col_perf4.metric(label="Teste de Heterocedasticidade - Prob(H)", value=f"{prob_h:.2f}", help="Testa se a variância dos erros é constante. Um valor > 0.05 (bom) indica que o modelo é estável.")
    
    else:
        st.info("👈 Configure os parâmetros na barra lateral e clique no botão para iniciar a análise.")

    st.markdown("---")
    st.caption("**Desenvolvido por:** Jonathas Gomes | **Objetivo:** Ferramenta de análise e projeção da inflação (IPCA) utilizando machine learning para apoiar decisões orçamentárias.")

if __name__ == '__main__':
    main()

