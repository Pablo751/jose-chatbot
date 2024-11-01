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
        """Initializes the product assistant."""
        try:
            # Initialize the OpenAI client properly
            self.client = OpenAI(api_key=api_key)
            self.categories = {}
            self.product_data = None
            self.embeddings = None
            self.faiss_index = None
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer: {e}")
            raise e
            
    def load_data(self, df: pd.DataFrame):
        """Carga y procesa los datos de productos."""
        try:
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
            logger.info("Datos de productos cargados y procesados.")
            self._extract_categories()
            self._generate_embeddings()
            logger.info("Embeddings generados y FAISS índice creado.")
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise e
            
    def _extract_categories(self):
        """Extrae y simplifica las categorías."""
        try:
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
        except Exception as e:
            logger.error(f"Error al extraer categorías: {e}")
            raise e
    
    def _generate_embeddings(self):
        """Genera embeddings para los productos."""
        try:
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
        except Exception as e:
            logger.error(f"Error al generar embeddings: {e}")
            raise e

    def search_products(self, 
                       query: str, 
                       category: Optional[str] = None, 
                       max_price: Optional[float] = None,
                       top_k: int = 5) -> List[Dict]:
        """Búsqueda mejorada de productos con filtrado de campos relevantes."""
        try:
            # Crear una copia del DataFrame para no modificar el original
            filtered_df = self.product_data.copy()
            
            # Aplicar filtro por categoría si se proporciona
            if category:
                # Simplificar categorías para mejor coincidencia
                category_mask = filtered_df['categories'].str.contains(category, na=False, regex=False)
                filtered_df = filtered_df[category_mask].reset_index(drop=True)
                logger.info(f"Filtro de categoría aplicado: '{category}'. Productos restantes: {len(filtered_df)}")
            
            # Aplicar filtro por precio si se proporciona
            if max_price is not None:
                filtered_df['price'] = pd.to_numeric(filtered_df['price'], errors='coerce')
                price_mask = filtered_df['price'] < max_price
                filtered_df = filtered_df[price_mask].reset_index(drop=True)
                logger.info(f"Filtro de precio aplicado: menor que {max_price}. Productos restantes: {len(filtered_df)}")
            
            # Verificar si hay resultados después del filtrado
            if filtered_df.empty:
                logger.warning("No hay productos que cumplan con los filtros.")
                return []
            
            # Generar embedding para la consulta
            query_embedding = self.model.encode([query], show_progress_bar=False)
            faiss.normalize_L2(query_embedding)
            
            # Generar textos relevantes para los productos filtrados, excluyendo metadata
            texts = []
            for idx, row in filtered_df.iterrows():
                # Solo incluir campos relevantes para la búsqueda
                relevant_text = f"{row['name']} "
                
                # Agregar descripción corta si está disponible
                if pd.notna(row.get('short_description')):
                    relevant_text += f"{row['short_description']} "
                
                # Agregar atributos adicionales relevantes
                if pd.notna(row.get('additional_attributes')):
                    attrs = row['additional_attributes']
                    # Extraer solo características relevantes del producto
                    relevant_attrs = []
                    for attr in attrs.split(','):
                        if any(key in attr.lower() for key in ['color', 'voltale', 'corriente', 'material', 'linea']):
                            relevant_attrs.append(attr.split('=')[1])
                    if relevant_attrs:
                        relevant_text += ' '.join(relevant_attrs)
                
                texts.append(relevant_text.strip())
            
            # Generar embeddings para los textos filtrados
            filtered_embeddings = self.model.encode(texts, show_progress_bar=False)
            faiss.normalize_L2(filtered_embeddings)
            
            # Crear índice temporal de FAISS
            temp_index = faiss.IndexFlatIP(filtered_embeddings.shape[1])
            temp_index.add(filtered_embeddings)
            
            results = []
            if len(filtered_embeddings) > 0:
                k = min(top_k, len(filtered_df))
                if k == 0:
                    return []
                
                # Realizar búsqueda
                D, I = temp_index.search(query_embedding, k)
                
                # Procesar resultados
                if len(D) > 0 and len(D[0]) > 0:
                    for distance, idx in zip(D[0], I[0]):
                        if 0 <= idx < len(filtered_df):
                            try:
                                product = filtered_df.iloc[idx]
                                # Crear una versión limpia del producto para el resultado
                                clean_product = {
                                    'name': product['name'],
                                    'price': float(product['price']),
                                    'categories': product['categories'].split(',')[0],
                                    'short_description': product.get('short_description', ''),
                                    'sku': product.get('sku', '')
                                }
                                
                                # Agregar atributos relevantes
                                if pd.notna(product.get('additional_attributes')):
                                    attrs = dict(item.split('=') for item in product['additional_attributes'].split(',')
                                               if '=' in item and any(key in item.lower() 
                                               for key in ['color', 'voltale', 'corriente', 'material', 'linea']))
                                    clean_product['attributes'] = attrs
                                
                                results.append({
                                    'product': clean_product,
                                    'score': float(distance),
                                    'query': query
                                })
                            except Exception as e:
                                logger.error(f"Error procesando producto {idx}: {e}")
                                continue
            
            logger.info(f"Búsqueda completada. Productos encontrados: {len(results)}")
            return results
                    
        except Exception as e:
            logger.error(f"Error en búsqueda de productos: {e}")
            return []
    
    def process_query_with_context(self, 
    def process_query_with_context(self, 
                                 query: str, 
                                 previous_results: Optional[List[Dict]] = None) -> Tuple[List[Dict], str]:
        """Procesa la consulta considerando el contexto y presupuesto."""
        try:
            if not query.strip():
                return [], "Por favor, hazme una pregunta sobre los productos."
            
            # Extraer rango de precios de la consulta
            price_range = self._extract_price_range(query)
            min_price, max_price = price_range if price_range else (None, None)
            
            # Detectar si es una consulta relacionada con precio
            price_related = any(word in query.lower() for word in 
                              ['precio', 'cuesta', 'barato', 'económico', 'presupuesto', 'costo'])
            
            # Buscar productos considerando el rango de precios
            results = self.search_products(
                query=query,
                max_price=max_price,
                top_k=5
            )
            
            if not results:
                return [], self._generate_no_results_response(min_price, max_price)
            
            # Filtrar productos por precio y ordenar por relevancia
            filtered_results = self._filter_and_sort_results(results, min_price, max_price)
            
            if not filtered_results:
                return [], self._generate_out_of_budget_response(results, min_price, max_price)
            
            response = self._generate_budget_aware_response(filtered_results, query, min_price, max_price)
            return filtered_results, response
                
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return [], "Lo siento, ocurrió un error. ¿Podrías reformular tu pregunta?"
    
    def _extract_price_range(self, query: str) -> Optional[Tuple[float, float]]:
        """Extrae rango de precios de la consulta."""
        try:
            # Patrones comunes de rangos de precio
            patterns = [
                r'(?:entre|de)\s*(?:$|₱|₿)?\s*(\d+(?:,\d+)?(?:\.\d+)?)\s*(?:a|y|hasta)\s*(?:$|₱|₿)?\s*(\d+(?:,\d+)?(?:\.\d+)?)',
                r'(?:menos de|bajo|máximo)\s*(?:$|₱|₿)?\s*(\d+(?:,\d+)?(?:\.\d+)?)',
                r'(?:presupuesto de)\s*(?:$|₱|₿)?\s*(\d+(?:,\d+)?(?:\.\d+)?)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query.lower())
                if match:
                    if len(match.groups()) == 2:
                        min_price = float(match.group(1).replace(',', ''))
                        max_price = float(match.group(2).replace(',', ''))
                        return min_price, max_price
                    else:
                        max_price = float(match.group(1).replace(',', ''))
                        return 0, max_price
            
            return None
        except Exception as e:
            logger.error(f"Error extrayendo rango de precios: {e}")
            return None
    
    def _filter_and_sort_results(self, 
                               results: List[Dict], 
                               min_price: Optional[float], 
                               max_price: Optional[float]) -> List[Dict]:
        """Filtra y ordena resultados por precio y relevancia."""
        try:
            filtered_results = []
            for result in results:
                price = float(result['product']['price'])
                if price <= 0:  # Ignorar productos sin precio
                    continue
                if min_price is not None and price < min_price:
                    continue
                if max_price is not None and price > max_price:
                    continue
                filtered_results.append(result)
            
            # Ordenar por precio y luego por score
            return sorted(filtered_results, 
                         key=lambda x: (float(x['product']['price']), -x['score']))
        except Exception as e:
            logger.error(f"Error filtrando resultados: {e}")
            return []
    
    def _generate_budget_aware_response(self, 
                                      results: List[Dict], 
                                      query: str,
                                      min_price: Optional[float], 
                                      max_price: Optional[float]) -> str:
        """Genera una respuesta considerando el presupuesto del usuario."""
        try:
            best_match = results[0]['product']
            price = float(best_match['price'])
            
            response = f"¡He encontrado algunas opciones dentro de tu presupuesto! "
            response += f"Te recomiendo el {best_match['name']} a ${price:,.2f}, "
            
            if best_match.get('attributes'):
                attrs = best_match['attributes']
                features = []
                if attrs.get('corriente_bticino'): 
                    features.append(f"corriente de {attrs['corriente_bticino']}")
                if attrs.get('material_bticino'):
                    features.append(f"material {attrs['material_bticino']}")
                if features:
                    response += f"que cuenta con {' y '.join(features)}. "
            
            if len(results) > 1:
                response += f"\n\nTambién tengo otras {len(results)-1} opciones similares "
                if min_price is not None and max_price is not None:
                    response += f"entre ${min_price:,.2f} y ${max_price:,.2f}. "
                response += "¿Te gustaría conocer más detalles sobre alguna de ellas?"
            
            return response
        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return "He encontrado algunos productos que podrían interesarte. ¿Te gustaría más detalles?"
    
    
    def _generate_response(self, query: str, product: Dict) -> str:
        """Generates a response for a product."""
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
            
            # Updated OpenAI API call
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "Eres un experto en productos eléctricos que ayuda a los clientes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            logger.info("Response generated by OpenAI.")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in GPT: {e}")
            return f"Te recomiendo el {product['name']} que cuesta ${float(product['price']):,.2f}."
    
    def _generate_comparative_response(self, query: str, prev_product: Dict, new_product: Dict) -> str:
        """Generates a comparative response between products."""
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
    
            # Updated OpenAI API call
            response = self.client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "Eres un experto en productos eléctricos especializado en encontrar las mejores ofertas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            logger.info("Comparative response generated by OpenAI.")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating comparative response: {e}")
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
        try:
            with st.spinner("Cargando datos de productos..."):
                # Updated OpenAI client initialization
                assistant = ProductAssistant(st.secrets["OPENAI_API_KEY"])
                product_data = pd.read_csv('data/jose.csv')
                assistant.load_data(product_data)
                st.session_state['assistant'] = assistant
                st.session_state['previous_results'] = None
                st.session_state['conversation_history'] = []
                logger.info("Product data loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing assistant: {e}")
            st.error(f"There was an error initializing the assistant: {e}")
    
    # Input del usuario
    user_question = st.text_input("¿En qué puedo ayudarte?")
    
    if st.button("Enviar"):
        if user_question:
            try:
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
            except Exception as e:
                logger.error(f"Error al procesar la consulta: {e}")
                st.error(f"Hubo un error al procesar tu consulta: {e}")
    
    # Mostrar historial
    if st.session_state.get('conversation_history'):
        st.markdown("---\n### 📝 Historial de Conversación")
        for i, entry in enumerate(reversed(st.session_state['conversation_history'])):
            with st.expander(f"Conversación {len(st.session_state['conversation_history'])-i}"):
                st.markdown(f"**👤 Pregunta:** {entry['question']}")
                st.markdown(f"**🤖 Respuesta:** {entry['response']}")
                if debug_mode:
                    st.write("**Resultados:**", entry['results'])


if __name__ == "__main__":
    main()
