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

class ProductAssistant:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.categories = {}
        self.product_data = None
        self.embeddings = None
        self.faiss_index = None
        
    def load_data(self, df: pd.DataFrame):
        """Carga y procesa los datos de productos."""
        self.product_data = df
        self._extract_categories()
        self._generate_embeddings()
        
    def _extract_categories(self):
        """Extrae y simplifica las categorías."""
        simplified_categories = {}
        
        for _, row in self.product_data.iterrows():
            categories = str(row['categories']).split(',')
            for category in categories:
                parts = category.strip().split('/')
                current_dict = simplified_categories
                
                # Ignorar "Default Category" y crear una estructura más limpia
                for part in parts:
                    if part != "Default Category":
                        if part not in current_dict:
                            current_dict[part] = {}
                        current_dict = current_dict[part]
        
        # Limpiar categorías vacías y redundantes
        self.categories = self._clean_categories(simplified_categories)
    
    def _clean_categories(self, categories: Dict) -> Dict:
        """Limpia y simplifica la estructura de categorías."""
        cleaned = {}
        for key, value in categories.items():
            # Ignorar categorías que son solo rutas
            if not key.startswith('Default Category'):
                if isinstance(value, dict):
                    # Recursivamente limpiar subcategorías
                    cleaned_sub = self._clean_categories(value)
                    if cleaned_sub or not value:  # Mantener categorías vacías como endpoints
                        cleaned[key] = cleaned_sub
                else:
                    cleaned[key] = value
        return cleaned
    
    def get_category_options(self) -> List[str]:
        """Retorna una lista plana de categorías para selección."""
        options = []
        
        def collect_categories(cat_dict: Dict, prefix: str = ""):
            for key, value in cat_dict.items():
                if not key.startswith('Default Category'):
                    full_path = f"{prefix}{key}" if prefix else key
                    options.append(full_path)
                    if isinstance(value, dict):
                        collect_categories(value, f"{full_path} > ")
        
        collect_categories(self.categories)
        return sorted(options)

    def process_query_with_context(self, 
                                 query: str, 
                                 previous_results: Optional[List[Dict]] = None,
                                 previous_category: Optional[str] = None) -> Tuple[List[Dict], str, Optional[str]]:
        """
        Procesa la consulta considerando el contexto anterior.
        """
        # Detectar si es una pregunta comparativa
        comparative_query = re.search(r'más barato|más económico|menor precio|más caro|mejor|similar', query.lower()) is not None
        
        if comparative_query and previous_results:
            # Usar el contexto anterior para la comparación
            if 'más barato' in query.lower() or 'más económico' in query.lower() or 'menor precio' in query.lower():
                # Obtener el precio del producto anterior
                prev_price = float(previous_results[0]['product']['price'])
                # Buscar productos más baratos en la misma categoría
                results = self.search_products(
                    query,
                    category=previous_category,
                    max_price=prev_price,
                    exclude_products=[r['product']['sku'] for r in previous_results]
                )
                if results:
                    response = self.generate_comparative_response(query, results[0]['product'], previous_results[0]['product'])
                    return results, response, previous_category
                else:
                    return [], "Lo siento, no encontré productos más económicos en esta categoría.", previous_category
        
        # Si no es comparativa o no hay contexto, realizar búsqueda normal
        category = self._detect_category(query) or previous_category
        results = self.search_products(query, category=category)
        
        if not results:
            return [], "No encontré productos que coincidan con tu búsqueda. ¿Podrías ser más específico?", category
        
        response = self.generate_response(query, results[0]['product'])
        return results, response, category

    def _detect_category(self, query: str) -> Optional[str]:
        """Detecta la categoría mencionada en la consulta."""
        query_lower = query.lower()
        categories = self.get_category_options()
        
        for category in categories:
            # Crear variantes de búsqueda (con y sin acentos)
            category_variants = [
                category.lower(),
                self._remove_accents(category.lower())
            ]
            
            # Buscar cada variante en la consulta
            for variant in category_variants:
                if variant in query_lower:
                    return category
        
        return None

    def _remove_accents(self, text: str) -> str:
        """Elimina acentos de un texto."""
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'
        }
        for a, b in replacements.items():
            text = text.replace(a, b)
        return text

    def search_products(self, 
                       query: str, 
                       category: Optional[str] = None, 
                       max_price: Optional[float] = None,
                       exclude_products: Optional[List[str]] = None,
                       top_k: int = 5) -> List[Dict]:
        """Búsqueda mejorada de productos."""
        filtered_df = self.product_data
        
        # Aplicar filtros
        if category:
            filtered_df = self._filter_by_category(filtered_df, category)
        if max_price:
            filtered_df = filtered_df[filtered_df['price'] < max_price]
        if exclude_products:
            filtered_df = filtered_df[~filtered_df['sku'].isin(exclude_products)]
            
        if filtered_df.empty:
            return []
            
        # Realizar búsqueda semántica
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query], normalize_embeddings=True)
        
        D, I = self.faiss_index.search(query_embedding, min(top_k, len(filtered_df)))
        
        results = []
        for distance, idx in zip(D[0], I[0]):
            if idx < len(filtered_df) and distance > 0:
                product = filtered_df.iloc[idx].to_dict()
                results.append({
                    'product': product,
                    'score': float(distance)
                })
        
        # Ordenar resultados
        if max_price:
            results.sort(key=lambda x: float(x['product']['price']))
            
        return results

    def generate_comparative_response(self, query: str, current_product: Dict, previous_product: Dict) -> str:
        """Genera una respuesta comparativa entre productos."""
        prompt = f"""
        Compara los siguientes productos y genera una respuesta que explique las diferencias,
        especialmente en precio y características principales:

        Producto anterior:
        Nombre: {previous_product.get('name')}
        Precio: ${float(previous_product.get('price', 0)):,.2f}
        Características: {previous_product.get('additional_attributes')}

        Producto actual:
        Nombre: {current_product.get('name')}
        Precio: ${float(current_product.get('price', 0)):,.2f}
        Características: {current_product.get('additional_attributes')}

        Consulta del cliente: {query}
        """

        try:
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "Eres un experto en productos eléctricos que ayuda a comparar opciones."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error en GPT: {e}")
            return f"Te recomiendo el {current_product.get('name')} que cuesta ${float(current_product.get('price', 0)):,.2f}, más económico que la opción anterior."

def main():
    st.set_page_config(
        page_title="💬 Asistente de Productos",
        page_icon="💬",
        layout="wide"
    )

    st.title("💬 Asistente de Productos")
    
    # Inicializar el asistente y estado de la sesión
    if 'assistant' not in st.session_state:
        assistant = ProductAssistant(st.secrets["OPENAI_API_KEY"])
        product_data = pd.read_csv('data/jose.csv')
        assistant.load_data(product_data)
        st.session_state['assistant'] = assistant
        st.session_state['previous_results'] = None
        st.session_state['current_category'] = None
        st.session_state['conversation_history'] = []

    # Sidebar con categorías
    st.sidebar.title("Categorías de Productos")
    categories = st.session_state['assistant'].get_category_options()
    selected_category = st.sidebar.selectbox(
        "Filtrar por categoría",
        ["Todas las categorías"] + categories
    )
    
    # Input del usuario
    user_question = st.text_input("¿En qué puedo ayudarte?")
    
    if st.button("Enviar"):
        if user_question:
            # Procesar la consulta con contexto
            results, response, category = st.session_state['assistant'].process_query_with_context(
                user_question,
                st.session_state['previous_results'],
                st.session_state['current_category']
            )
            
            # Actualizar estado
            st.session_state['previous_results'] = results
            st.session_state['current_category'] = category
            
            # Guardar en historial
            st.session_state['conversation_history'].append({
                'question': user_question,
                'response': response,
                'results': results
            })
            
            # Mostrar respuesta
            st.markdown("### Respuesta:")
            st.write(response)
            
            if results:
                st.markdown("### Productos relacionados:")
                for result in results:
                    product = result['product']
                    st.markdown(f"""
                    **{product['name']}**
                    - Precio: ${float(product['price']):,.2f}
                    - Categoría: {product.get('categories', '').split(',')[0]}
                    """)
    
    # Mostrar historial
    if st.session_state['conversation_history']:
        st.markdown("---\n### Historial de Conversación")
        for i, entry in enumerate(st.session_state['conversation_history']):
            st.markdown(f"**Pregunta {i+1}:** {entry['question']}")
            st.markdown(f"**Respuesta {i+1}:** {entry['response']}")
            st.markdown("---")

if __name__ == "__main__":
    main()
