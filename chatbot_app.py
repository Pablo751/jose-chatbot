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

    def search_products(self, 
                       query: str, 
                       category: Optional[str] = None, 
                       max_price: Optional[float] = None,
                       top_k: int = 5) -> List[Dict]:
        """Búsqueda mejorada de productos con manejo correcto de índices."""
        try:
            # Crear una copia del DataFrame para no modificar el original
            filtered_df = self.product_data.copy()
            
            # Aplicar filtros
            if category:
                category_mask = filtered_df['categories'].str.contains(category, na=False, regex=False)
                filtered_df = filtered_df[category_mask].reset_index(drop=True)
            if max_price:
                price_mask = filtered_df['price'].astype(float) < max_price
                filtered_df = filtered_df[price_mask].reset_index(drop=True)
            
            if filtered_df.empty:
                return []
            
            # Generar embedding para la consulta
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            
            # Generar embeddings solo para los productos filtrados
            texts = (filtered_df['name'] + " " + 
                    filtered_df['description'].fillna('') + " " + 
                    filtered_df['short_description'].fillna(''))
            filtered_embeddings = self.model.encode(texts.tolist(), show_progress_bar=False)
            faiss.normalize_L2(filtered_embeddings)
            
            # Crear un índice temporal para los productos filtrados
            temp_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
            if len(filtered_embeddings) > 0:
                temp_index.add(filtered_embeddings)
                
                # Realizar la búsqueda
                k = min(top_k, len(filtered_df))
                D, I = temp_index.search(query_embedding, k)
                
                results = []
                for distance, idx in zip(D[0], I[0]):
                    if idx < len(filtered_df):  # Verificar índice válido
                        product = filtered_df.iloc[idx].to_dict()
                        try:
                            price = float(product['price'])
                            if price > 0:  # Solo incluir productos con precio válido
                                results.append({
                                    'product': product,
                                    'score': float(distance),
                                    'query': query
                                })
                        except (ValueError, TypeError):
                            continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda de productos: {e}")
            return []
    
    def process_query_with_context(self, 
                                 query: str, 
                                 previous_results: Optional[List[Dict]] = None) -> Tuple[List[Dict], str]:
        """Procesa la consulta considerando el contexto anterior."""
        try:
            if not query.strip():
                return [], "Por favor, hazme una pregunta sobre los productos."
            
            # Detectar si es una pregunta sobre precio
            price_related = re.search(r'más barato|más económico|menor precio|barato|económico', query.lower()) is not None
            
            if price_related and previous_results and len(previous_results) > 0:
                prev_product = previous_results[0].get('product')
                if not prev_product:
                    return [], "No encontré el producto anterior. ¿Podrías repetir tu pregunta inicial?"
                
                try:
                    prev_price = float(prev_product['price'])
                except (ValueError, TypeError):
                    return [], "Hubo un problema con el precio del producto anterior. ¿Podrías hacer tu pregunta de nuevo?"
                
                # Extraer categoría del producto anterior
                prev_category = None
                if 'categories' in prev_product:
                    categories = str(prev_product['categories']).split(',')
                    if categories:
                        # Tomar la categoría más específica (última)
                        prev_category = categories[-1].strip()
                
                # Buscar productos más baratos
                results = self.search_products(
                    query=previous_results[0].get('query', ''),  # Usar la consulta original
                    category=prev_category,
                    max_price=prev_price * 0.95,  # 5% más barato
                    top_k=5
                )
                
                if results:
                    response = self._generate_comparative_response(
                        query=query,
                        prev_product=prev_product,
                        new_product=results[0]['product']
                    )
                    return results, response
                else:
                    return [], f"No encontré productos más económicos que el {prev_product['name']} (${prev_price:,.2f}). ¿Te gustaría ver alternativas en otra categoría?"
            
            # Búsqueda normal
            results = self.search_products(query)
            if not results:
                return [], "No encontré productos que coincidan con tu búsqueda. ¿Podrías darme más detalles sobre lo que buscas?"
            
            response = self._generate_response(query, results[0]['product'])
            return results, response
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return [], "Lo siento, ocurrió un error. ¿Podrías reformular tu pregunta?"


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
        try:
            savings = float(prev_product['price']) - float(new_product['price'])
            
            prompt = f"""
            Como experto en ventas, compara estos productos respondiendo a: {query}

            Producto anterior:
            Nombre: {prev_product['name']}
            Precio: ${float(prev_product['price']):,.2f}
            Características: {prev_product.get('additional_attributes', '')}

            Producto nuevo (más económico):
            Nombre: {new_product['name']}
            Precio: ${float(new_product['price']):,.2f}
            Ahorro: ${savings:,.2f}
            Características: {new_product.get('additional_attributes', '')}
            """

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
            logger.error(f"Error generando respuesta comparativa: {e}")
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
