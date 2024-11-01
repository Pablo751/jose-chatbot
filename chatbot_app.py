import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from typing import Dict, List, Tuple, Optional
import numpy as np
import re
import logging
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
from openai import OpenAI

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Descargar recursos de NLTK
nltk.download('stopwords')

class ProductAssistant:
    def __init__(self, api_key: str):
        """Inicializa el asistente de productos."""
        self.client = OpenAI(api_key=api_key)
        self.categories = {}
        self.product_data = None
        self.embeddings = None
        self.faiss_index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_data(self, df: pd.DataFrame):
        """Carga y procesa los datos de productos."""
        self.product_data = df
        self._extract_categories()
        self._generate_embeddings()
        
    def _extract_categories(self):
        """Extrae y simplifica las categorías."""
        main_categories = set()
        for _, row in self.product_data.iterrows():
            categories = str(row['categories']).split(',')
            for category in categories:
                parts = category.strip().split('/')
                # Tomar solo la categoría principal después de "Default Category"
                for i, part in enumerate(parts):
                    if part.strip() == "Default Category" and i + 1 < len(parts):
                        main_categories.add(parts[i + 1].strip())
        
        # Organizar en diccionario
        self.categories = {cat: {} for cat in main_categories}
    
    def _generate_embeddings(self):
        """Genera embeddings para los productos."""
        texts = (self.product_data['name'] + " " + 
                self.product_data['description'] + " " + 
                self.product_data['short_description'])
        self.embeddings = self.model.encode(texts.tolist(), show_progress_bar=True)
        faiss.normalize_L2(self.embeddings)
        
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings)

    def search_products(self, query: str, category: str = None, max_price: float = None, top_k: int = 5) -> List[Dict]:
        """Búsqueda de productos con filtros."""
        filtered_df = self.product_data.copy()
        
        if category:
            filtered_df = filtered_df[filtered_df['categories'].str.contains(category, na=False)]
        if max_price:
            filtered_df = filtered_df[filtered_df['price'] < max_price]
            
        if filtered_df.empty:
            return []
            
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        D, I = self.faiss_index.search(query_embedding, min(top_k, len(filtered_df)))
        
        results = []
        for distance, idx in zip(D[0], I[0]):
            if distance > 0:  # Solo incluir resultados relevantes
                product = filtered_df.iloc[idx].to_dict()
                results.append({
                    'product': product,
                    'score': float(distance),
                    'query': query  # Guardar la consulta original
                })
        
        # Ordenar por precio si se especificó un máximo
        if max_price:
            results.sort(key=lambda x: float(x['product']['price']))
            
        return results

    def process_query_with_context(self, 
                                 query: str, 
                                 previous_results: Optional[List[Dict]] = None) -> Tuple[List[Dict], str]:
        """Procesa la consulta considerando el contexto anterior."""
        # Detectar si es una pregunta sobre precio
        price_related = re.search(r'más barato|más económico|menor precio|barato|económico', query.lower()) is not None
        
        if price_related and previous_results and previous_results[0]['product']:
            # Obtener precio del producto anterior
            prev_product = previous_results[0]['product']
            prev_price = float(prev_product['price'])
            prev_category = None
            
            # Extraer categoría del producto anterior
            if 'categories' in prev_product:
                categories = str(prev_product['categories']).split(',')
                if categories:
                    prev_category = categories[0].strip()
            
            # Buscar productos más baratos
            results = self.search_products(
                previous_results[0]['query'],  # Usar la consulta original
                category=prev_category,
                max_price=prev_price * 0.95,  # 5% más barato
                top_k=5
            )
            
            if results:
                # Generar respuesta comparativa
                response = self._generate_comparative_response(
                    query=query,
                    prev_product=prev_product,
                    new_product=results[0]['product']
                )
                return results, response
            else:
                return [], f"Lo siento, no encontré productos más económicos que el {prev_product['name']} (${prev_price:,.2f})."
        
        # Búsqueda normal si no es comparativa o no hay contexto
        results = self.search_products(query)
        if not results:
            return [], "No encontré productos que coincidan con tu búsqueda. ¿Podrías darme más detalles?"
        
        response = self._generate_response(query, results[0]['product'])
        return results, response

    def _generate_response(self, query: str, product: Dict) -> str:
        """Genera una respuesta para un producto."""
        prompt = f"""
        Como asistente de ventas experto, responde a esta consulta:
        
        Consulta: {query}
        
        Producto:
        Nombre: {product['name']}
        Precio: ${float(product['price']):,.2f}
        Descripción: {product.get('short_description', '')}
        Características: {product.get('additional_attributes', '')}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "Eres un experto en productos eléctricos que ayuda a los clientes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error en GPT: {e}")
            return f"Te recomiendo el {product['name']} que cuesta ${float(product['price']):,.2f}."

    def _generate_comparative_response(self, query: str, prev_product: Dict, new_product: Dict) -> str:
        """Genera una respuesta comparativa entre productos."""
        savings = float(prev_product['price']) - float(new_product['price'])
        
        prompt = f"""
        Como experto en ventas, compara estos productos respondiendo a: {query}

        Producto anterior:
        Nombre: {prev_product['name']}
        Precio: ${float(prev_product['price']):,.2f}

        Producto nuevo (más económico):
        Nombre: {new_product['name']}
        Precio: ${float(new_product['price']):,.2f}
        Ahorro: ${savings:,.2f}

        Características nuevo producto: {new_product.get('additional_attributes', '')}
        """

        try:
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "Eres un experto en productos eléctricos especializado en encontrar las mejores ofertas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error en GPT: {e}")
            return f"He encontrado el {new_product['name']} a ${float(new_product['price']):,.2f}, " \
                   f"que te permite ahorrar ${savings:,.2f} comparado con la opción anterior."

def main():
    st.set_page_config(
        page_title="💬 Asistente de Productos",
        page_icon="💬",
        layout="wide"
    )

    st.title("💬 Asistente de Productos")
    
    # Inicializar estado
    if 'assistant' not in st.session_state:
        assistant = ProductAssistant(st.secrets["OPENAI_API_KEY"])
        product_data = pd.read_csv('data/jose.csv')
        assistant.load_data(product_data)
        st.session_state['assistant'] = assistant
        st.session_state['previous_results'] = None
        st.session_state['conversation_history'] = []

    # Input del usuario
    user_question = st.text_input("¿En qué puedo ayudarte?")
    
    if st.button("Enviar"):
        if user_question:
            # Procesar consulta con contexto
            results, response = st.session_state['assistant'].process_query_with_context(
                user_question,
                st.session_state['previous_results']
            )
            
            # Actualizar contexto
            st.session_state['previous_results'] = results
            
            # Guardar en historial
            st.session_state['conversation_history'].append({
                'question': user_question,
                'response': response,
                'results': results
            })
            
            # Mostrar respuesta
            st.markdown("### 🤖 Respuesta:")
            st.write(response)
            
            if results:
                st.markdown("### 📦 Productos encontrados:")
                for result in results:
                    product = result['product']
                    st.markdown(f"""
                    **{product['name']}**
                    - 💰 Precio: ${float(product['price']):,.2f}
                    - 📑 Categoría: {product.get('categories', '').split(',')[0]}
                    """)
    
    # Mostrar historial
    if st.session_state['conversation_history']:
        st.markdown("---\n### 📝 Historial de Conversación")
        for i, entry in enumerate(reversed(st.session_state['conversation_history'])):
            with st.expander(f"Conversación {len(st.session_state['conversation_history'])-i}"):
                st.markdown(f"**👤 Pregunta:** {entry['question']}")
                st.markdown(f"**🤖 Respuesta:** {entry['response']}")

if __name__ == "__main__":
    main()
