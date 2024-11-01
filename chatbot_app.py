from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
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

class ProductAssistant:
    def __init__(self, api_key: str):
        """Inicializa el asistente de productos."""
        self.client = OpenAI(api_key=api_key)
        self.categories = {}
        self.product_data = None
        self.embeddings = None
        self.faiss_index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _generate_embeddings(self):
        """Genera embeddings para los productos."""
        texts = (self.product_data['name'] + " " + 
                self.product_data['description'] + " " + 
                self.product_data['short_description'])
        self.embeddings = self.model.encode(texts.tolist(), show_progress_bar=True)
        
        # Normalizar embeddings
        faiss.normalize_L2(self.embeddings)
        
        # Crear índice FAISS
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings)
    
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
                parts = [p.strip() for p in category.split('/') if p.strip() != "Default Category"]
                current_dict = simplified_categories
                
                for part in parts:
                    if part not in current_dict:
                        current_dict[part] = {}
                    current_dict = current_dict[part]
        
        self.categories = self._clean_categories(simplified_categories)
    
    def _clean_categories(self, categories: Dict) -> Dict:
        """Limpia y simplifica la estructura de categorías."""
        cleaned = {}
        for key, value in categories.items():
            if not key.startswith('Default Category'):
                if isinstance(value, dict):
                    cleaned_sub = self._clean_categories(value)
                    if cleaned_sub or not value:
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
        return sorted(list(set(options)))  # Eliminar duplicados

    def _filter_by_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Filtra productos por categoría."""
        category_parts = category.split(' > ')
        pattern = '.*'.join(map(re.escape, category_parts))
        return df[df['categories'].str.contains(pattern, regex=True, na=False)]

    def search_products(self, 
                       query: str, 
                       category: Optional[str] = None, 
                       max_price: Optional[float] = None,
                       exclude_products: Optional[List[str]] = None,
                       top_k: int = 5) -> List[Dict]:
        """Búsqueda mejorada de productos."""
        filtered_df = self.product_data.copy()
        
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
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        D, I = self.faiss_index.search(query_embedding, min(top_k, len(filtered_df)))
        
        results = []
        for distance, idx in zip(D[0], I[0]):
            if idx < len(filtered_df) and distance > 0:
                product = filtered_df.iloc[idx].to_dict()
                if float(product['price']) > 0:  # Solo incluir productos con precio válido
                    results.append({
                        'product': product,
                        'score': float(distance)
                    })
        
        # Ordenar resultados
        if max_price:
            results.sort(key=lambda x: float(x['product']['price']))
            
        return results

    def generate_response(self, query: str, product_info: Dict) -> str:
        """Genera una respuesta personalizada para un producto."""
        prompt = f"""
        Como asistente de ventas experto, genera una respuesta para esta consulta:

        Consulta: {query}

        Producto:
        Nombre: {product_info.get('name')}
        Precio: ${float(product_info.get('price', 0)):,.2f}
        Descripción: {product_info.get('short_description')}
        Características: {product_info.get('additional_attributes')}

        Responde de manera conversacional y destaca las características más relevantes para la consulta del cliente.
        """

        try:
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "Eres un experto en productos eléctricos que ayuda a los clientes a encontrar la mejor opción."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error en GPT: {e}")
            return f"Te recomiendo el {product_info.get('name')} que tiene un precio de ${float(product_info.get('price', 0)):,.2f}."

    def process_query_with_context(self, 
                                 query: str, 
                                 previous_results: Optional[List[Dict]] = None,
                                 previous_category: Optional[str] = None) -> Tuple[List[Dict], str, Optional[str]]:
        """Procesa la consulta considerando el contexto anterior."""
        if not query.strip():
            return [], "Por favor, hazme una pregunta sobre los productos.", previous_category

        # Detectar si es una pregunta comparativa
        comparative_query = re.search(r'más barato|más económico|menor precio|más caro|mejor|similar', query.lower()) is not None
        
        if comparative_query and previous_results:
            prev_price = float(previous_results[0]['product']['price'])
            if 'más barato' in query.lower() or 'más económico' in query.lower():
                results = self.search_products(
                    query,
                    category=previous_category,
                    max_price=prev_price,
                    exclude_products=[r['product']['sku'] for r in previous_results]
                )
                if results:
                    response = self.generate_comparative_response(query, results[0]['product'], previous_results[0]['product'])
                    return results, response, previous_category
                return [], "Lo siento, no encontré productos más económicos en esta categoría.", previous_category
        
        # Búsqueda normal
        category = self._detect_category(query) or previous_category
        results = self.search_products(query, category=category)
        
        if not results:
            suggestions = "\n".join([f"- {cat}" for cat in self.get_category_options()[:5]])
            return [], f"No encontré productos que coincidan. Prueba buscando en estas categorías:\n{suggestions}", category
        
        response = self.generate_response(query, results[0]['product'])
        return results, response, category

    def _detect_category(self, query: str) -> Optional[str]:
        """Detecta la categoría mencionada en la consulta."""
        query_lower = query.lower()
        for category in self.get_category_options():
            if category.lower() in query_lower:
                return category
        return None

    def generate_comparative_response(self, query: str, current_product: Dict, previous_product: Dict) -> str:
        """Genera una respuesta comparativa entre productos."""
        prompt = f"""
        Compara estos productos y explica las diferencias principales:

        Producto anterior:
        Nombre: {previous_product.get('name')}
        Precio: ${float(previous_product.get('price', 0)):,.2f}
        Características: {previous_product.get('additional_attributes')}

        Producto actual:
        Nombre: {current_product.get('name')}
        Precio: ${float(current_product.get('price', 0)):,.2f}
        Características: {current_product.get('additional_attributes')}

        Consulta: {query}
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
            return f"He encontrado el {current_product.get('name')} que cuesta ${float(current_product.get('price', 0)):,.2f}, más económico que la opción anterior."

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
        st.session_state['current_category'] = None
        st.session_state['conversation_history'] = []

    # Sidebar con categorías
    st.sidebar.title("Navegación por Categorías")
    categories = st.session_state['assistant'].get_category_options()
    selected_category = st.sidebar.selectbox(
        "Filtrar por categoría",
        ["Todas las categorías"] + categories
    )
    
    # Input del usuario
    user_question = st.text_input("¿En qué puedo ayudarte?")
    
    if st.button("Enviar"):
        if user_question:
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
            st.markdown(f"### 🤖 Respuesta:")
            st.write(response)
            
            if results:
                st.markdown("### 📦 Productos relacionados:")
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
