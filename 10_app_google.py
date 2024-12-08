import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from typing import Dict, List

# Carregar variáveis de ambiente
load_dotenv()

# Configurar chave da API do Google
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class SentimentAnalysisChain:
    def __init__(self, csv_path='analise_sentimentos.csv'):
        """
        Inicialização do pipeline de análise de sentimentos
        Chain of Thoughts - Etapa 0: Preparação
        """
        self.csv_path = csv_path
        self.raw_data = None
        self.processed_data = None
        self.insights = {}
        self.model = genai.GenerativeModel('gemini-pro')

    def load_and_validate_data(self) -> bool:
        """
        Carregar e validar dados do CSV
        Chain of Thoughts - Etapa 1: Carregamento e Validação
        """
        try:
            self.raw_data = pd.read_csv(self.csv_path)
            
            # Validações de estrutura
            required_columns = ['Métrica', 'Quantidade', 'Percentual']
            if not all(col in self.raw_data.columns for col in required_columns):
                st.error("Estrutura de colunas inválida!")
                return False
            
            return True
        except Exception as e:
            st.error(f"Erro ao carregar dados: {e}")
            return False

    def generate_data_summary_prompt(self) -> str:
        """
        Gerar prompt para resumo estatístico
        Chain of Thoughts - Etapa 2: Geração de Prompt Estatístico
        """
        data_summary = self.raw_data.to_string()
        
        return f"""
        Você é um estatístico especialista em análise de sentimentos.
        
        Dados Brutos:
        {data_summary}
        
        Tarefa: Forneça uma análise estatística detalhada:
        1. Total de registros
        2. Distribuição percentual dos sentimentos
        3. Variações significativas
        4. Interpretação inicial dos dados
        
        Formato de resposta: JSON estruturado
        """

    def process_statistical_summary(self) -> Dict:
        """
        Processar resumo estatístico
        Chain of Thoughts - Etapa 3: Processamento Estatístico
        """
        prompt = self.generate_data_summary_prompt()
        
        try:
            response = self.model.generate_content(prompt)
            # Aqui você pode adicionar parsing do JSON
            self.processed_data = {
                'total_dialogues': self.raw_data['Quantidade'].sum(),
                'sentiments': self.raw_data.to_dict(orient='records')
            }
            return self.processed_data
        except Exception as e:
            st.error(f"Erro no processamento estatístico: {e}")
            return {}

    def generate_contextual_insights_prompt(self) -> str:
        """
        Gerar prompt para insights contextuais
        Chain of Thoughts - Etapa 4: Geração de Insights Contextuais
        """
        statistical_context = str(self.processed_data)
        
        return f"""
        Você é um analista de comunicação especialista em interpretação de sentimentos.
        
        Contexto Estatístico:
        {statistical_context}
        
        Tarefa: Desenvolva uma narrativa profunda que responda:
        1. Qual a significância destes sentimentos?
        2. Quais padrões podem ser identificados?
        3. Que fatores podem influenciar esta distribuição?
        4. Recomendações baseadas nos dados
        
        Estilo: Narrativa acadêmica, objetiva e analítica
        Limite: 300-400 palavras
        """

    def generate_contextual_insights(self) -> str:
        """
        Gerar insights contextuais
        Chain of Thoughts - Etapa 5: Geração de Insights
        """
        prompt = self.generate_contextual_insights_prompt()
        
        try:
            response = self.model.generate_content(prompt)
            self.insights['contextual'] = response.text
            return response.text
        except Exception as e:
            st.error(f"Erro na geração de insights: {e}")
            return "Não foi possível gerar insights."

    def generate_visualization_prompt(self) -> str:
        """
        Gerar prompt para a etapa de visualização
        Chain of Thoughts - Etapa 6: Visualização
        """
        return """
        Gere uma representação visual adequada para os dados analisados.
        Tipo: Gráfico de pizza.
        Destaque: Percentual de cada sentimento (Métrica).
        """

    def create_visualization(self) -> px.pie:
        """
        Criar visualização de dados com base no prompt
        """
        try:
            prompt = self.generate_visualization_prompt()
            response = self.model.generate_content(prompt)  # Placeholder caso queira usar IA na geração
            color_map = {
                'Positive': '#2ecc71',   # Verde
                'Neutral': '#3498db',    # Azul
                'Negative': '#e74c3c'    # Vermelho
            }

            fig = px.pie(
                self.raw_data, 
                values='Percentual', 
                names='Métrica', 
                title='Distribuição de Sentimentos',
                color='Métrica',
                color_discrete_map=color_map
            )
            fig.update_traces(textinfo='percent+label')
            return fig
        except Exception as e:
            st.error(f"Erro na geração do gráfico: {e}")
            return None

def main():
    """
    Função principal orquestrando a cadeia de processamento
    Chain of Thoughts - Etapa Final: Orquestração
    """
    st.set_page_config(layout='wide', page_title='Análise de Sentimentos')
    st.title('Análise de Sentimentos - Cadeia Progressiva')

    # Iniciar pipeline
    pipeline = SentimentAnalysisChain()
    
    # Executar etapas em sequência
    if pipeline.load_and_validate_data():
        # Processar dados estatisticamente
        pipeline.process_statistical_summary()
        
        # Gerar insights contextuais
        contextual_insights = pipeline.generate_contextual_insights()
        
        # Criar visualização
        visualization = pipeline.create_visualization()
        
        # Layout Streamlit
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if visualization:
                st.plotly_chart(visualization, use_container_width=True)
        
        with col2:
            st.header('Métricas')
            total = pipeline.processed_data['total_dialogues']
            for sentiment in pipeline.processed_data['sentiments']:
                st.metric(
                    label=f"{sentiment['Métrica']} Sentiment", 
                    value=f"{sentiment['Quantidade']} ({sentiment['Percentual']}%)"
                )
            st.metric(label='Total de Diálogos', value=total)
        
        # Exibir insights
        st.header('Insights Contextuais')
        st.write(contextual_insights)

if __name__ == "__main__":
    main()
