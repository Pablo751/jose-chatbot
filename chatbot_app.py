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
import openai  # Corregido

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Descargar recursos de NLTK
nltk.download('stopwords')

class ProductAssistant:
    def __init__(self, api_key: str):
        """Inicializa el asistente de productos."""
        openai.api_key = api_key  # Corregido
        self.categories = {}
        self.product_data = None
        self.embeddings = None
        self.faiss_index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_data(self, df: pd.DataFrame):
        """Carga y procesa los datos de productos."""
        # Verificar columnas necesarias
        required_columns = ['name', 'description', 'short_description', 'price', 'categories']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Falta la columna requerida: {col}")
                raise ValueError(f"Falta la columna requerida: {col}")
        
        # Manejar valores faltantes y tipos de datos
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['categories'] = df['categories'].fillna('Default Category')
        df['name'] = df['name'].fillna('Nombre Desconocido')
        df['description'] = df['description'].fillna('')
        df['short_description'] = df['short_description'].fillna('')
        
        self.product_data = df
        self._extract_categories()
        self._generate_embeddings()
        logger.info("Embeddings generados y FAISS índice creado.")
        
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
        logger.info(f"Categorías extraídas: {list(self.categories.keys())}")
    
    def _generate_embeddings(self):
        """Genera embeddings para los productos."""
        texts = (self.product_data['name'] + " " + 
                self.product_data['description'] + " " + 
                self.product_data['short_description'])
        self.embeddings = self.model.encode(texts.tolist(), show_progress_bar=True)
        faiss.normalize_L2(self.embeddings)
        logger.info("Embeddings normalizados.")
        
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.embeddings)
        logger.info(f"FAISS índice creado con dimensión: {dimension} y {self.embeddings.shape[0]} embeddings.")
    
    def search_products(self, 
                       query: str, 
                       category: Optional[str] = None, 
                       max_price: Optional[float] = None,
                       top_k: int = 5) -> List[Dict]:
        """Búsqueda mejorada de productos con manejo correcto de índices."""
        try:
            # Crear una copia del DataFrame para no modificar el original
            filtered_df = self.product_data.copy()
            
            # Aplicar filtro por categoría si se proporciona
            if category:
                category_mask = filtered_df['categories'].str.contains(category, na=False, regex=False)
                filtered_df = filtered_df[category_mask].reset_index(drop=True)
                logger.info(f"Aplicado filtro de categoría: '{category}'. Productos restantes: {len(filtered_df)}")
            else:
                logger.info("No se aplicó filtro de categoría.")
            
            # Aplicar filtro por precio si se proporciona
            if max_price is not None:
                filtered_df['price'] = pd.to_numeric(filtered_df['price'], errors='coerce')
                price_mask = filtered_df['price'] < max_price
                filtered_df = filtered_df[price_mask].reset_index(drop=True)
                logger.info(f"Aplicado filtro de precio: menor que {max_price}. Productos restantes: {len(filtered_df)}")
            else:
                logger.info("No se aplicó filtro de precio.")
            
            # Verificar si hay resultados después del filtrado
            if filtered_df.empty:
                logger.warning("No hay productos que cumplan con los filtros.")
                return []
            
            logger.info(f"Número de productos después del filtrado: {len(filtered_df)}")
            
            # Generar embedding para la consulta
            query_embedding = self.model.encode([query], show_progress_bar=False)
            faiss.normalize_L2(query_embedding)
            logger.debug(f"Embedding de la consulta: {query_embedding}")
            
            # Generar textos para los productos filtrados
            texts = []
            for idx, row in filtered_df.iterrows():
                text = f"{row['name']} "
                if pd.notna(row.get('description')):
                    text += f"{row['description']} "
                if pd.notna(row.get('short_description')):
                    text += f"{row['short_description']}"
                texts.append(text.strip())
            
            logger.info(f"Número de textos a codificar: {len(texts)}")
            
            # Verificar si hay textos para procesar
            if not texts:
                logger.warning("No hay textos para procesar después de generar los textos.")
                return []
            
            # Generar embeddings para los textos filtrados
            filtered_embeddings = self.model.encode(texts, show_progress_bar=False)
            faiss.normalize_L2(filtered_embeddings)
            logger.debug(f"Embeddings generados: {filtered_embeddings.shape}")
            
            # Verificar la correspondencia entre embeddings y DataFrame
            if len(filtered_embeddings) != len(filtered_df):
                logger.error("Mismatch entre el número de embeddings y el DataFrame filtrado.")
                logger.error(f"Embeddings: {len(filtered_embeddings)}, DataFrame: {len(filtered_df)}")
                return []
            
            # Crear índice temporal de FAISS
            temp_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
            temp_index.add(filtered_embeddings)
            logger.info(f"FAISS índice temporal creado con dimensión: {filtered_embeddings.shape[1]}")
            
            logger.info(f"Realizando búsqueda FAISS con top_k={top_k}")
            
            results = []
            if len(filtered_embeddings) > 0:
                k = min(top_k, len(filtered_df))
                if k == 0:
                    logger.warning("top_k es 0, no se realizan búsquedas.")
                    return []
                
                # Realizar búsqueda
                D, I = temp_index.search(query_embedding, k)
                logger.info(f"FAISS search completed. Distancias: {D}, Índices: {I}")
                
                # Verificar y procesar los resultados
                if len(D) > 0 and len(D[0]) > 0:
                    for distance, idx in zip(D[0], I[0]):
                        logger.debug(f"Procesando índice: {idx} con distancia: {distance}")
                        
                        # Verificar que el índice está dentro del rango
                        if 0 <= idx < len(filtered_df):
                            try:
                                product = filtered_df.iloc[idx].to_dict()
                                price = pd.to_numeric(product['price'], errors='coerce')
                                if pd.notna(price) and price > 0:
                                    results.append({
                                        'product': product,
                                        'score': float(distance),
                                        'query': query
                                    })
                                    logger.debug(f"Producto añadido: {product['name']}, Precio: {price}")
                                else:
                                    logger.warning(f"Producto con índice {idx} tiene un precio inválido: {price}")
                            except Exception as e:
                                logger.error(f"Error procesando producto con índice {idx}: {e}")
                                continue
                        else:
                            logger.error(f"Índice FAISS {idx} está fuera de los límites del DataFrame filtrado (0 - {len(filtered_df)-1})")
            
            logger.info(f"Número de productos encontrados: {len(results)}")
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
                
                logger.info(f"Buscando productos más baratos que {prev_product['name']} (${prev_price:,.2f}) en categoría '{prev_category}'")
                
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
            logger.info(f"Realizando búsqueda normal para la consulta: '{query}'")
            results = self.search_products(query)
            if not results:
                logger.info("No se encontraron productos que coincidan con la búsqueda.")
                return [], "No encontré productos que coincidan con tu búsqueda. ¿Podrías darme más detalles sobre lo que buscas?"
            
            response = self._generate_response(query, results[0]['product'])
            return results, response
            
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return [], "Lo siento, ocurrió un error. ¿Podrías reformular tu pregunta?"
    
    
    def _generate_response(self, query: str, product: Dict) -> str:
        """Genera una respuesta para un producto."""
        try:
            prompt = f"""
            Como asistente de ventas experto, responde a esta consulta:
            
            Consulta: {query}
            
            Producto:
            Nombre: {product['name']}
            Precio: ${float(product['price']):,.2f}
            Descripción: {product.get('short_description', '')}
            Características: {product.get('additional_attributes', '')}
            """
            
            # Usar OpenAI correctamente
            response = openai.ChatCompletion.create(
                model="chatgpt-4o-latest",  # Mantener según solicitud del usuario
                messages=[
                    {"role": "system", "content": "Eres un experto en productos eléctricos que ayuda a los clientes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            logger.info("Respuesta generada por OpenAI.")
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

            response = openai.ChatCompletion.create(
                model="chatgpt-4o-latest",  # Mantener según solicitud del usuario
                messages=[
                    {"role": "system", "content": "Eres un experto en productos eléctricos especializado en encontrar las mejores ofertas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            logger.info("Respuesta comparativa generada por OpenAI.")
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
    
    # Checkbox para habilitar modo de depuración
    debug_mode = st.checkbox("🔍 Habilitar Modo de Depuración")
    
    # Inicializar estado
    if 'assistant' not in st.session_state:
        assistant = ProductAssistant(st.secrets["OPENAI_API_KEY"])
        try:
            product_data = pd.read_csv('data/jose.csv')
            assistant.load_data(product_data)
            st.session_state['assistant'] = assistant
            st.session_state['previous_results'] = None
            st.session_state['conversation_history'] = []
            logger.info("Datos de productos cargados exitosamente.")
        except Exception as e:
            logger.error(f"Error cargando datos de productos: {e}")
            st.error("Hubo un error al cargar los datos de productos. Por favor, revisa los logs.")
    
    # Input del usuario
    user_question = st.text_input("¿En qué puedo ayudarte?")
    
    if st.button("Enviar"):
        if user_question:
            # Procesar consulta con contexto
            results, response = st.session_state['assistant'].process_query_with_context(
                user_question,
                st.session_state.get('previous_results')
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
            
            # Si el modo de depuración está habilitado, mostrar información adicional
            if debug_mode:
                st.markdown("### 🐞 Información de Depuración:")
                st.write("**Consulta:**", user_question)
                st.write("**Resultados obtenidos:**", results)
                if results:
                    st.write("**Detalles de los productos encontrados:**")
                    for i, result in enumerate(results, 1):
                        st.write(f"**Producto {i}:**")
                        st.json(result['product'])
                        st.write(f"**Score:** {result['score']}")
                        st.write(f"**Consulta Usada:** {result['query']}")
    
    # Mostrar historial
    if st.session_state.get('conversation_history'):
        st.markdown("---\n### 📝 Historial de Conversación")
        for i, entry in enumerate(reversed(st.session_state['conversation_history'])):
            with st.expander(f"Conversación {len(st.session_state['conversation_history'])-i}"):
                st.markdown(f"**👤 Pregunta:** {entry['question']}")
                st.markdown(f"**🤖 Respuesta:** {entry['response']}")
                if debug_mode:
                    st.write("**Resultados:**", entry['results'])
