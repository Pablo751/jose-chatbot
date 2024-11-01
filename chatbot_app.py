import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from typing import Dict, List, Tuple
import numpy as np
import re
import logging
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
from openai import OpenAI
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Descargar recursos NLTK
nltk.download('stopwords')

class ProductAssistant:
    def __init__(self, api_key: str):
        """Inicializa el asistente de productos."""
        self.client = OpenAI(api_key=api_key)
        self.categories = {}
        self.product_data = None
        self.embeddings = None
        self.faiss_index = None
        
    def load_data(self, df: pd.DataFrame):
        """Carga y procesa los datos de productos."""
        self.product_data = df
        # Extraer categorías únicas y crear jerarquía
        self._extract_categories()
        # Generar embeddings
        self._generate_embeddings()
        
    def _extract_categories(self):
        """Extrae y organiza las categorías en una jerarquía."""
        for _, row in self.product_data.iterrows():
            categories = str(row['categories']).split(',')
            current_dict = self.categories
            for category in categories:
                category = category.strip()
                if category not in current_dict:
                    current_dict[category] = {}
                current_dict = current_dict[category]
    
    def get_category_tree(self) -> Dict:
        """Retorna el árbol de categorías."""
        return self.categories
    
    def _generate_embeddings(self):
        """Genera embeddings para los productos."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = (self.product_data['name'] + " " + 
                self.product_data['description'] + " " + 
                self.product_data['short_description'])
        self.embeddings = model.encode(texts.tolist(), show_progress_bar=True)
        faiss.normalize_L2(self.embeddings)
        
        # Configurar índice FAISS
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings)
    
    def _filter_by_price_range(self, products: pd.DataFrame, max_price: float = None) -> pd.DataFrame:
        """Filtra productos por rango de precio."""
        if max_price:
            return products[products['price'] <= max_price]
        return products
    
    def _filter_by_category(self, products: pd.DataFrame, category: str) -> pd.DataFrame:
        """Filtra productos por categoría."""
        return products[products['categories'].str.contains(category, na=False)]
    
    def search_products(self, query: str, category: str = None, max_price: float = None, top_k: int = 5) -> List[Dict]:
        """
        Busca productos basado en la consulta, categoría y precio.
        """
        # Primero filtrar por categoría si se especifica
        filtered_df = self.product_data
        if category:
            filtered_df = self._filter_by_category(filtered_df, category)
        
        # Luego filtrar por precio si se especifica
        filtered_df = self._filter_by_price_range(filtered_df, max_price)
        
        if filtered_df.empty:
            return []
            
        # Generar embedding para la consulta
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query], normalize_embeddings=True)
        
        # Buscar productos similares
        D, I = self.faiss_index.search(query_embedding, top_k)
        
        results = []
        for distance, idx in zip(D[0], I[0]):
            if idx < len(filtered_df):
                product = filtered_df.iloc[idx].to_dict()
                if distance > 0:
                    results.append({
                        'product': product,
                        'score': float(distance)
                    })
        
        # Ordenar por precio si se especificó un máximo
        if max_price:
            results.sort(key=lambda x: float(x['product']['price']))
            
        return results

    def generate_response(self, query: str, product_info: Dict) -> str:
        """
        Genera una respuesta personalizada usando GPT.
        """
        # Crear un prompt estructurado para GPT
        prompt = f"""
        Actúa como un asistente de ventas experto. Basándote en la siguiente consulta del cliente y la información del producto,
        genera una respuesta informativa y útil. Incluye precio, características principales y beneficios clave.
        
        Consulta del cliente: {query}
        
        Información del producto:
        Nombre: {product_info.get('name')}
        Precio: ${float(product_info.get('price', 0)):,.2f}
        Descripción: {product_info.get('short_description')}
        Características: {product_info.get('additional_attributes')}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "Eres un asistente de ventas experto que ayuda a los clientes a encontrar los productos adecuados."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error al generar respuesta con GPT: {e}")
            return f"Te recomiendo el {product_info.get('name')} a ${float(product_info.get('price', 0)):,.2f}."

    def process_user_query(self, query: str) -> Tuple[List[Dict], str]:
        """
        Procesa la consulta del usuario y retorna resultados y respuesta.
        """
        # Detectar si es una consulta de precio
        price_related = re.search(r'económico|barato|precio|costo|valor', query.lower()) is not None
        
        # Detectar categoría mencionada
        category_match = None
        for category in self.categories.keys():
            if category.lower() in query.lower():
                category_match = category
                break
        
        # Si es consulta de precio, buscar opciones más económicas
        if price_related:
            avg_price = self.product_data['price'].mean()
            max_price = avg_price * 0.8
            results = self.search_products(query, category_match, max_price)
        else:
            results = self.search_products(query, category_match)
        
        if not results:
            return [], "Lo siento, no encontré productos que coincidan con tu consulta. ¿Podrías especificar más detalles?"
        
        # Generar respuesta para el mejor match
        response = self.generate_response(query, results[0]['product'])
        
        return results, response

def format_product_display(product: Dict) -> str:
    """
    Formatea la información del producto para mostrar.
    """
    return f"""
    Nombre: {product.get('name')}
    Precio: ${float(product.get('price', 0)):,.2f}
    Descripción: {product.get('short_description')}
    """

# Función principal de Streamlit
def main():
    st.set_page_config(
        page_title="💬 Asistente de Productos",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("💬 Asistente de Productos")
    
    # Inicializar el asistente
    if 'assistant' not in st.session_state:
        assistant = ProductAssistant(st.secrets["OPENAI_API_KEY"])
        # Cargar datos
        product_data = pd.read_csv('data/jose.csv')
        assistant.load_data(product_data)
        st.session_state['assistant'] = assistant
    
    # Mostrar categorías en el sidebar
    st.sidebar.title("Categorías Disponibles")
    categories = st.session_state['assistant'].get_category_tree()
    st.sidebar.json(json.dumps(categories, indent=2))
    
    # Input del usuario
    user_question = st.text_input("Escribe tu pregunta aquí:")
    
    if st.button("Enviar Pregunta"):
        if user_question:
            results, response = st.session_state['assistant'].process_user_query(user_question)
            
            st.markdown("### Respuesta:")
            st.write(response)
            
            if results:
                st.markdown("### Productos Relacionados:")
                for result in results:
                    st.markdown(format_product_display(result['product']))

if __name__ == "__main__":
    main()
