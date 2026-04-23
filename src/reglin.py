###Importando bibliotecas
#
import streamlit as st
import pandas as pd
from  sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path
#
base_dir=Path(__file__).resolve().parent
caminho_arquivo = base_dir.parent / "dados" / "slr12.csv"
dados=pd.read_csv(caminho_arquivo,sep=';')
X=dados[['FrqAnual']]                           # Variável independente => dataframe
y=dados['CusInic']                              # Variável dependente   => série
modelo = LinearRegression().fit(X,y)
#
st.title("Caso1 - Franquias e custos - Regressão Linear")
col1, col2 = st.columns(2)
with col1:
    st.header('Dados')
    dados.columns=['Valor da Franquia','Custo Inicial']
    st.dataframe(dados,height=250)
with col2:
    st.header('Gráfico')
    fig, ax = plt.subplots()
    ax.scatter(X,y, color='blue')
    ax.plot(X,modelo.predict(X),color='red')
    st.pyplot(fig) 

st.header('Valor Anual da Franquia')
novo_valor=st.number_input('Insira novo valor:', min_value=100.0, max_value=999999.0)
if st.button('Processar'):
    dados_novo_valor=pd.DataFrame([[novo_valor]],columns=['FrqAnual'])
    prev=modelo.predict(dados_novo_valor)
    st.header(f"Previsão de Custo Inicial: {prev[0]:.2f}")
