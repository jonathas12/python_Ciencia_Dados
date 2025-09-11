# app.py
import streamlit as st
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import calendar
import io

# --- Configuração da Página ---
st.set_page_config(
    page_title="Calculadora de Depreciação de Edificações",
    page_icon="🏗️",
    layout="wide",
)

# --- Funções de Backend ---

@st.cache_data
def get_tabela_conservacao():
    """
    Cria e retorna a Tabela de Estado de Conservação como um DataFrame.
    """
    data = {
        "Estado de Conservação": [
            "Novo", "Entre novo e regular", "Regular", "Entre regular e reparos simples",
            "Reparos Simples", "Entre reparos simples e importantes", "Reparos Importantes",
            "Entre Reparos importantes e sem valor", "Sem valor"
        ],
        "VIDA ÚTIL REMANESCENTE (Anos)": [25, 22, 19, 16, 13, 10, 7, 4, 0],
        "% MENSAL DE DEPRECIAÇÃO (%)": [0.33, 0.38, 0.44, 0.52, 0.64, 0.83, 1.19, 2.08, 0.00],
        "% ANUAL DE DEPRECIAÇÃO (%)": [4.00, 4.55, 5.26, 6.25, 7.69, 10.00, 14.29, 25.00, 0.00]
    }
    df = pd.DataFrame(data)
    df["N° de Cotas (mensal)"] = df["VIDA ÚTIL REMANESCENTE (Anos)"] * 12
    return df

def calcular_depreciacao(data_contabilizacao, valor_imovel, valor_terreno, estado_selecionado, df_conservacao):
    """
    Realiza todos os cálculos de depreciação com base nas entradas do usuário.
    """
    percentual_residual = 0.20
    valor_residual = valor_imovel * percentual_residual
    valor_depreciavel = valor_imovel - valor_residual

    dados_estado = df_conservacao[df_conservacao["Estado de Conservação"] == estado_selecionado].iloc[0]
    taxa_mensal_perc = dados_estado["% MENSAL DE DEPRECIAÇÃO (%)"]
    n_cotas = int(dados_estado["N° de Cotas (mensal)"])
    vida_util_remanescente = dados_estado["VIDA ÚTIL REMANESCENTE (Anos)"]
    taxa_anual_depreciacao = dados_estado["% ANUAL DE DEPRECIAÇÃO (%)"]
    cota_depreciacao_mensal = valor_depreciavel * (taxa_mensal_perc / 100)
    
    if n_cotas == 0:
        return {
            "percentual_residual": percentual_residual, "valor_residual": valor_residual,
            "valor_depreciavel": valor_depreciavel, "vida_util_remanescente": vida_util_remanescente,
            "taxa_anual_depreciacao": taxa_anual_depreciacao, "n_cotas": n_cotas,
            "taxa_mensal_perc": taxa_mensal_perc, "cota_depreciacao_mensal": 0
        }, pd.DataFrame()

    cronograma_data = []
    depreciacao_acumulada = 0
    valor_contabil_liquido = valor_imovel
    data_calculo = data_contabilizacao
    
    for mes in range(1, n_cotas + 1):
        data_calculo = data_calculo + relativedelta(months=1)
        ultimo_dia = calendar.monthrange(data_calculo.year, data_calculo.month)[1]
        data_depreciacao = date(data_calculo.year, data_calculo.month, ultimo_dia)

        depreciacao_acumulada += cota_depreciacao_mensal
        valor_contabil_liquido -= cota_depreciacao_mensal
        valor_total_imovel = valor_contabil_liquido + valor_terreno
        
        cronograma_data.append({
            "Mês": mes,
            "Data Depreciação": data_depreciacao,
            "Cota de Depreciação Mensal (R$)": cota_depreciacao_mensal,
            "Depreciação Acumulada (R$)": depreciacao_acumulada,
            "Valor Contábil Líquido (R$)": valor_contabil_liquido,
            "Valor Total do Imóvel (R$)": valor_total_imovel
        })

    df_cronograma = pd.DataFrame(cronograma_data)
    df_cronograma['Data Depreciação'] = pd.to_datetime(df_cronograma['Data Depreciação'])

    resultados_iniciais = {
        "percentual_residual": percentual_residual,
        "valor_residual": valor_residual,
        "valor_depreciavel": valor_depreciavel,
        "vida_util_remanescente": vida_util_remanescente,
        "taxa_anual_depreciacao": taxa_anual_depreciacao,
        "n_cotas": n_cotas,
        "taxa_mensal_perc": taxa_mensal_perc,
        "cota_depreciacao_mensal": cota_depreciacao_mensal
    }
    
    return resultados_iniciais, df_cronograma

# --- Interface do Usuário (Frontend) ---

st.title("🏗️ Calculadora de Depreciação de Edificações")
st.markdown("Esta ferramenta calcula a depreciação de uma edificação com base nos dados do imóvel e na tabela de estado de conservação.")

tabela_conservacao = get_tabela_conservacao()

with st.sidebar:
    st.header("Parâmetros do Imóvel")
    
    data_contabilizacao_input = st.date_input(
        "Data de Contabilização", 
        date.today(),
        help="Data base para o início do cálculo da depreciação."
    )
    
    valor_imovel_input = st.number_input(
        "Valor Contábil da Edificação (R$)", 
        min_value=0.01, value=500000.0, step=1000.0, format="%.2f",
        help="Valor inicial da edificação que será depreciado."
    )
    
    valor_terreno_input = st.number_input(
        "Valor do Terreno (R$)", 
        min_value=0.0, value=150000.0, step=1000.0, format="%.2f",
        help="Valor do terreno, que não sofre depreciação."
    )

    estado_conservacao_input = st.selectbox(
        "Estado de Conservação",
        tabela_conservacao["Estado de Conservação"].tolist(),
        index=0,
        help="Selecione o estado de conservação da edificação."
    )
    
    calcular_btn = st.button("Calcular Depreciação", type="primary", use_container_width=True)

if calcular_btn:
    resultados, df_final = calcular_depreciacao(
        data_contabilizacao_input,
        valor_imovel_input,
        valor_terreno_input,
        estado_conservacao_input,
        tabela_conservacao
    )

    st.header("Resultados do Cálculo")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Valor Residual (R$)", value=f"{resultados['valor_residual']:,.2f}")
    with col2:
        st.metric(label="Valor Depreciável (R$)", value=f"{resultados['valor_depreciavel']:,.2f}")
    with col3:
        st.metric(label="Cota Mensal de Depreciação (R$)", value=f"{resultados['cota_depreciacao_mensal']:,.2f}")
    with col4:
         st.metric(label="Nº de Cotas (Meses)", value=f"{resultados['n_cotas']}")

    st.divider()
    st.subheader("Cronograma de Depreciação Mensal")

    if not df_final.empty:
        df_display = df_final.copy()
        
        currency_columns = [
            "Cota de Depreciação Mensal (R$)", "Depreciação Acumulada (R$)",
            "Valor Contábil Líquido (R$)", "Valor Total do Imóvel (R$)"
        ]
        for col in currency_columns:
            df_display[col] = df_display[col].map('R$ {:,.2f}'.format)
        
        df_display["Data Depreciação"] = df_display["Data Depreciação"].dt.strftime('%d/%m/%Y')
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # --- SEÇÃO DO GRÁFICO CORRIGIDA ---
        st.divider()
        st.subheader("Tendência de Depreciação da Edificação")

        # Prepara o dataframe para o gráfico usando o dataframe completo
        df_grafico = df_final.set_index('Data Depreciação')[['Valor Contábil Líquido (R$)']]
        
        st.line_chart(df_grafico)
        st.caption("Gráfico mostrando a evolução do Valor Contábil Líquido da Edificação (R$) ao longo de todo o período.")
        # --- FIM DA SEÇÃO DO GRÁFICO ---
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(df_final)

        st.download_button(
           label="Baixar cronograma em CSV",
           data=csv,
           file_name=f"depreciacao_{estado_conservacao_input.lower().replace(' ', '_')}.csv",
           mime="text/csv",
        )
        
    else:
        st.warning("Não há depreciação a ser calculada para o estado 'Sem valor'.")

    with st.expander("Ver Tabela Base de Estado de Conservação"):
        st.dataframe(tabela_conservacao, use_container_width=True, hide_index=True)

else:
    st.info("Preencha os parâmetros ao lado e clique em 'Calcular Depreciação' para ver os resultados.")

st.markdown("---")
st.markdown("Desenvolvido por Jonathas Gomes como uma solução para cálculo de depreciação de bens imóveis.")