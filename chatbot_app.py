# chatbot_app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from typing import Dict, List
import numpy as np
import re
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import time
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
from openai import OpenAI
import os

# -------------------------------
# Configurar el Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Descargar recursos de NLTK si no están presentes
nltk.download('stopwords')

# -------------------------------
# 1. Configurar la Aplicación Streamlit
# -------------------------------

st.set_page_config(
    page_title="💬 Asistente de Productos",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💬 Asistente de Productos")

# -------------------------------
# 2. Configurar el Cliente de OpenAI
# -------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def call_gpt4o(prompt: str, max_tokens: int = 500) -> str:
    """
    Llama a la API de OpenAI GPT-4o para generar una respuesta basada en el prompt.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """
                Eres un asistente de ventas que ayuda a los clientes a encontrar productos en nuestra tienda.
                Utiliza únicamente la información proporcionada sobre el producto para responder. No inventes ni agregues detalles adicionales.
                """},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Temperatura baja para mayor precisión
            max_tokens=max_tokens,
            n=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error al llamar a GPT-4o: {e}")
        return "Lo siento, ocurrió un error al generar la respuesta. Por favor, inténtalo de nuevo más tarde."

def handle_api_errors(func):
    """
    Decorator para manejar errores de la API de OpenAI
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error en la API de OpenAI: {e}")
            st.error("Ocurrió un error al procesar tu solicitud. Por favor, inténtalo de nuevo más tarde.")
            return None
    return wrapper

# -------------------------------
# 3. Cargar y Preprocesar Datos
# -------------------------------

@st.cache_data(show_spinner=False)
def load_product_data(file_path: str) -> pd.DataFrame:
    """
    Carga y preprocesa los datos de productos con solo las columnas esenciales.
    Filtra productos con precio mayor a 0.
    """
    columns_to_load = ['sku', 'name', 'description', 'short_description', 'price', 
                       'additional_attributes', 'base_image', 'url_key']
    df = pd.read_csv(file_path, usecols=columns_to_load)
    
    # Limpiar y rellenar valores nulos
    df.fillna({'additional_attributes': 'Información no disponible', 
              'short_description': 'Información no disponible',
              'description': 'Información no disponible',
              'price': 0.0,
              'base_image': '',
              'url_key': '#'}, inplace=True)
    
    # Filtrar productos con precio > 0
    df = df[df['price'] > 0.0].reset_index(drop=True)
    
    return df

def validate_csv(df: pd.DataFrame) -> bool:
    """
    Valida que el DataFrame tenga las columnas esenciales y que no falten datos críticos.
    """
    expected_columns = ['sku', 'name', 'description', 'short_description', 'price', 
                        'additional_attributes', 'base_image', 'url_key']
    if not all(column in df.columns for column in expected_columns):
        st.error("El CSV no contiene todas las columnas requeridas.")
        return False
    
    # Verificar valores nulos en columnas esenciales (después de rellenar)
    essential_columns = ['sku', 'name', 'price']
    if df[essential_columns].isnull().any().any():
        st.warning("Algunos productos tienen información incompleta en campos esenciales. Revisar el CSV.")
    
    return True

# Cargar datos
product_file = 'data/jose.csv'
try:
    product_data = load_product_data(product_file)
    if not validate_csv(product_data):
        st.stop()
    st.sidebar.success("✅ Catálogo de productos cargado correctamente")
    logger.info(f"Productos cargados: {len(product_data)}")
except Exception as e:
    st.error(f"Error al cargar el catálogo de productos: {e}")
    logger.error(f"Error al cargar el catálogo de productos: {e}")
    st.stop()

# -------------------------------
# 4. Generar Embeddings y Configurar FAISS
# -------------------------------

@st.cache_resource(show_spinner=False)
def generate_embeddings_faiss_ivf(df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2', nlist: int = 50) -> (np.ndarray, faiss.Index):
    """
    Genera embeddings para los productos y configura el índice FAISS IVFFlat para mayor escalabilidad.
    """
    model = SentenceTransformer(model_name)
    # Concatenar campos relevantes para generar el embedding
    texts = df['name'] + " " + df['description'] + " " + df['short_description']
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    
    # Normalizar embeddings
    faiss.normalize_L2(embeddings)
    
    # Crear índice FAISS IVFFlat
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dimension)  # Cuantizador para el índice IVFFlat
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Entrenar el índice con un subconjunto de embeddings
    if not index.is_trained:
        logger.info("Entrenando el índice FAISS...")
        index.train(embeddings)
    
    index.add(embeddings)
    
    return embeddings, index

try:
    embeddings, faiss_index = generate_embeddings_faiss_ivf(product_data)
    logger.info("Embeddings generados y FAISS index configurado.")
except Exception as e:
    st.error(f"Error al generar embeddings: {e}")
    logger.error(f"Error al generar embeddings: {e}")
    st.stop()

# -------------------------------
# 5. Implementación del Monitoreo de Cambios en el CSV
# -------------------------------

class CSVChangeHandler(FileSystemEventHandler):
    def __init__(self, file_path):
        self.file_path = os.path.abspath(file_path)
    
    def on_modified(self, event):
        if os.path.abspath(event.src_path) == self.file_path:
            logger.info(f"Archivo {self.file_path} modificado.")
            st.session_state['file_changed'] = True

def start_file_watcher(file_path: str):
    event_handler = CSVChangeHandler(file_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(file_path)) or '.', recursive=False)
    observer.start()
    logger.info("Watcher iniciado.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# -------------------------------
# 6. Funciones de Búsqueda y Coincidencia
# -------------------------------

def extract_search_terms(user_query: str) -> List[str]:
    """
    Extrae términos relevantes de búsqueda de la consulta del usuario.
    Implementación mejorada con lematización y eliminación de stop words.
    """
    # Convertir a minúsculas
    user_query = user_query.lower()
    # Remover caracteres especiales
    user_query = re.sub(r'[^\w\s]', '', user_query)
    # Tokenizar
    terms = user_query.split()
    # Filtrar palabras irrelevantes usando NLTK
    stop_words = set(stopwords.words('spanish')).union({'puedes', 'recomendar', 'economico', 'económico', 'por', 'que', 'puedo', 'tienes', 'tenes'})
    filtered_terms = [term for term in terms if term not in stop_words]
    # Lematización
    stemmer = SnowballStemmer('spanish')
    lemmatized_terms = [stemmer.stem(term) for term in filtered_terms]
    logger.info(f"Términos de búsqueda extraídos: {lemmatized_terms}")
    return lemmatized_terms

def extract_price_filter(user_query: str, df: pd.DataFrame) -> Dict:
    """
    Extrae el rango de precios solicitado por el usuario.
    """
    price_filter = {}
    # Detectar frases como "más barato", "económico", o rangos específicos
    if any(phrase in user_query.lower() for phrase in ["más barato", "económico", "economico"]):
        # Definir un porcentaje de reducción, por ejemplo, 20% menos que el precio promedio
        avg_price = df['price'].mean()
        price_filter['max'] = avg_price * 0.8
    # Aquí puedes agregar más lógica para detectar rangos específicos
    return price_filter

def prioritize_by_price(matches: List[Dict], ascending: bool = True) -> List[Dict]:
    """
    Prioriza los productos por precio.
    """
    try:
        return sorted(matches, key=lambda x: x['product']['price'], reverse=not ascending)
    except Exception as e:
        logger.error(f"Error al priorizar por precio: {e}")
        return matches

def search_products_faiss(query: str, model: SentenceTransformer, index: faiss.Index, df: pd.DataFrame, top_k: int = 5, price_filter: Dict = None) -> List[Dict]:
    """
    Busca productos que coincidan con la consulta utilizando FAISS y embeddings.
    Se puede aplicar un filtro de precio opcional.
    """
    try:
        if not query.strip():
            logger.warning("Consulta vacía después de filtrar stop words.")
            return []
        
        # Aplicar filtro de precio si está especificado
        if price_filter:
            min_price = price_filter.get('min', df['price'].min())
            max_price = price_filter.get('max', df['price'].max())
            filtered_df = df[(df['price'] >= min_price) & (df['price'] <= max_price)].reset_index(drop=True)
            if filtered_df.empty:
                logger.info("No hay productos dentro del rango de precios especificado.")
                return []
            
            # Generar embeddings para el DataFrame filtrado
            texts = filtered_df['name'] + " " + filtered_df['description'] + " " + filtered_df['short_description']
            embeddings_filtered = model.encode(texts.tolist(), normalize_embeddings=True)
            faiss.normalize_L2(embeddings_filtered)
            
            # Crear un índice temporal para los productos filtrados
            dimension = embeddings_filtered.shape[1]
            quantizer = faiss.IndexFlatIP(dimension)
            temp_index = faiss.IndexIVFFlat(quantizer, dimension, 10, faiss.METRIC_INNER_PRODUCT)
            if not temp_index.is_trained:
                temp_index.train(embeddings_filtered)
            temp_index.add(embeddings_filtered)
            
            # Generar embedding para la consulta
            query_embedding = model.encode([query], normalize_embeddings=True)
            
            # Realizar búsqueda en el índice FAISS temporal
            D, I = temp_index.search(query_embedding, top_k)
            
            results = []
            for distance, idx in zip(D[0], I[0]):
                if distance > 0:  # Similaridad positiva
                    product = filtered_df.iloc[idx].to_dict()
                    results.append({'product': product, 'score': distance})
            
            logger.info(f"Productos encontrados con filtro de precio: {len(results)}")
            return results
        else:
            # Usar el índice principal
            query_embedding = model.encode([query], normalize_embeddings=True)
            D, I = index.search(query_embedding, top_k)
            results = []
            for distance, idx in zip(D[0], I[0]):
                if distance > 0:  # Similaridad positiva
                    product = df.iloc[idx].to_dict()
                    results.append({'product': product, 'score': distance})
            logger.info(f"Productos encontrados: {len(results)}")
            return results
    except Exception as e:
        logger.error(f"Error durante la búsqueda de productos: {e}")
        st.error("Ocurrió un error durante la búsqueda de productos. Por favor, inténtalo de nuevo más tarde.")
        return []

def format_features(features: str) -> str:
    """
    Formatea las características del producto como una lista con viñetas.
    """
    if features == 'Información no disponible':
        return features
    feature_pairs = [attr.split('=') for attr in features.split(',') if '=' in attr]
    feature_list = '\n'.join([f"- **{k.strip()}**: {v.strip()}" for k, v in feature_pairs])
    return feature_list

def validate_response(response: str, product_info: Dict[str, str]) -> str:
    """
    Valida que la respuesta generada contenga información precisa del producto.
    """
    # Validar el precio
    try:
        price = float(product_info.get('price', 0.0))
        price_formatted = f"${price:,.2f}"
        if price_formatted not in response:
            response += f"\n\n**Nota:** El precio actual es {price_formatted}."
    except:
        pass
    return response

def generate_product_response(product_info: Dict[str, str], price_filter: Dict = None) -> str:
    """
    Genera una respuesta personalizada utilizando GPT-4o basada únicamente en la información del producto.
    """
    # Formatear el precio
    try:
        price = float(product_info.get('price', 0.0))
        price_formatted = f"${price:,.2f}"
    except:
        price_formatted = "Información no disponible"
    
    # Formatear características
    features_formatted = format_features(product_info.get('additional_attributes', 'Información no disponible'))
    
    # URL del producto
    url_key = product_info.get('url_key', '#')
    product_url = f"https://tutienda.com/product/{url_key}"  # Reemplaza con la URL base de tu tienda
    
    # Imagen del producto
    base_image = product_info.get('base_image', '')
    if base_image and isinstance(base_image, str):
        image_url = f"https://tutienda.com/{base_image}"  # Reemplaza con la URL base de tus imágenes
    else:
        image_url = "https://via.placeholder.com/150"  # Imagen por defecto
    
    # Crear un resumen de la información del producto para el prompt
    product_summary = f"""
    Nombre del Producto: {product_info.get('name', 'Información no disponible')}
    Descripción: {product_info.get('short_description', 'Información no disponible')}
    Precio: {price_formatted}
    Características:
    {features_formatted}
    URL del Producto: {product_url}
    Imagen del Producto: {image_url}
    """
    
    # Incluir el rango de precios en el prompt si está especificado
    if price_filter:
        min_price = price_filter.get('min', 'sin mínimo')
        max_price = price_filter.get('max', 'sin máximo')
        price_instructions = f"El rango de precios solicitado es hasta {max_price}."
    else:
        price_instructions = "No hay restricciones de precio especificadas."
    
    # Crear el prompt para GPT-4o
    prompt = f"""
    {price_instructions}
    Utilizando únicamente la siguiente información sobre el producto, genera una descripción conversacional y atractiva para el cliente:
    
    {product_summary}
    """
    
    # Llamar a GPT-4o para generar la respuesta
    response = call_gpt4o(prompt, max_tokens=500)
    
    # Validar la respuesta
    response = validate_response(response, product_info)
    
    return response

# -------------------------------
# 7. Implementación de Feedback del Usuario
# -------------------------------

def add_feedback(product_name: str, feedback: str):
    """
    Almacena el feedback del usuario para un producto específico.
    """
    feedback_file = 'data/feedback.csv'
    new_feedback = {'product_name': product_name, 'feedback': feedback}
    try:
        if os.path.exists(feedback_file):
            df_feedback = pd.read_csv(feedback_file)
            df_feedback = df_feedback.append(new_feedback, ignore_index=True)
        else:
            df_feedback = pd.DataFrame([new_feedback])
    except Exception as e:
        logger.error(f"Error al agregar feedback: {e}")
        st.error("Ocurrió un error al guardar tu feedback. Por favor, inténtalo de nuevo más tarde.")
        return
    df_feedback.to_csv(feedback_file, index=False)
    logger.info(f"Feedback agregado para producto: {product_name}")
    st.success("¡Gracias por tu feedback!")

# -------------------------------
# 8. Función para Iniciar el Watcher en un Hilo Separado
# -------------------------------

# Iniciar el watcher en un hilo separado para no bloquear la interfaz
if 'watcher_started' not in st.session_state:
    watcher_thread = threading.Thread(target=start_file_watcher, args=(product_file,), daemon=True)
    watcher_thread.start()
    st.session_state['watcher_started'] = True

# -------------------------------
# 9. Interfaz Principal con Soporte de Follow-Up
# -------------------------------

# Inicializar historial de conversación y contexto del producto
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'current_product' not in st.session_state:
    st.session_state['current_product'] = None  # Producto actual para preguntas de seguimiento

# Verificar si el archivo ha cambiado y recargar los datos
if 'file_changed' in st.session_state and st.session_state['file_changed']:
    try:
        product_data = load_product_data(product_file)
        if not validate_csv(product_data):
            st.stop()
        faiss_index = generate_embeddings_faiss_ivf(product_data)[1]
        st.sidebar.success("✅ Catálogo de productos recargado correctamente")
        logger.info(f"Productos recargados: {len(product_data)}")
        st.session_state['file_changed'] = False
    except Exception as e:
        st.error(f"Error al recargar el catálogo de productos: {e}")
        logger.error(f"Error al recargar el catálogo de productos: {e}")
        st.stop()

# Entrada del usuario
st.write("👋 ¡Hola! Soy tu asistente de productos. ¿En qué puedo ayudarte?")
user_question = st.text_input("Escribe tu pregunta aquí:", key="user_input")

# Botón para enviar la pregunta
if st.button("Enviar Pregunta"):
    if not user_question.strip():
        st.warning("Por favor, ingresa una pregunta.")
    else:
        with st.spinner("Buscando la mejor respuesta..."):
            # Extraer términos de búsqueda
            search_terms = extract_search_terms(user_question)
            st.write(f"🔍 **Términos de Búsqueda:** {', '.join(search_terms)}")  # Mostrar términos extraídos
            
            # Reconstruir la consulta sin stop words para generar una búsqueda más efectiva
            reconstructed_query = ' '.join(search_terms)
            
            # Detectar si se solicita un filtro de precio
            price_filter = extract_price_filter(user_question, product_data)
            if price_filter:
                st.write(f"💲 **Filtro de Precio Aplicado:** Hasta {price_filter.get('max', 'sin máximo')}")
            
            # Cargar el modelo para generar embeddings de la consulta
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Buscar productos relevantes usando FAISS con filtro de precio si aplica
            matches = search_products_faiss(
                reconstructed_query, 
                model, 
                faiss_index, 
                product_data, 
                top_k=5, 
                price_filter=price_filter if price_filter else None
            )
            
            # Priorizar los resultados por precio si se aplicó un filtro
            if price_filter:
                matches = prioritize_by_price(matches, ascending=True)
            
            if matches:
                # Usar el producto más relevante para generar la respuesta
                best_match = matches[0]['product']
                
                # Verificar si esta es una pregunta de seguimiento
                if st.session_state['current_product'] and "más" in user_question.lower():
                    # Utilizar el mismo producto para la pregunta de seguimiento
                    response = generate_product_response(st.session_state['current_product'], price_filter)
                else:
                    # Actualizar el producto actual en el estado de sesión
                    st.session_state['current_product'] = best_match
                    response = generate_product_response(best_match, price_filter)
                
                # Agregar al historial
                st.session_state['conversation'].append({
                    "question": user_question,
                    "response": response,
                    "product": best_match['name']
                })
                
                # Mostrar la respuesta utilizando Markdown para un mejor formato
                st.markdown(response)
                
                # Mostrar productos alternativos con feedback
                if len(matches) > 1:
                    st.write("📌 **También podrían interesarte estos productos:**")
                    for match in matches[1:]:
                        product = match['product']
                        # Formatear el precio
                        try:
                            price = float(product.get('price', 0.0))
                            price_formatted = f"${price:,.2f}"
                        except:
                            price_formatted = "Información no disponible"
                        # URL del producto
                        url_key = product.get('url_key', '#')
                        product_url = f"https://tutienda.com/product/{url_key}"  # Reemplaza con la URL base de tu tienda
                        st.write(f"- [{product['name']}]({product_url}) - {price_formatted}")
                        
                        # Botones de feedback
                        feedback = st.radio(
                            f"¿Te gustó la recomendación de {product['name']}?",
                            options=["Sí", "No"],
                            key=f"feedback_{product['sku']}"
                        )
                        if st.button(f"Enviar Feedback para {product['sku']}", key=f"feedback_btn_{product['sku']}"):
                            add_feedback(product['name'], feedback)
            else:
                response = "Lo siento, no encontré productos que coincidan con tu consulta. ¿Podrías reformular tu pregunta?"
                st.session_state['conversation'].append({
                    "question": user_question,
                    "response": response,
                    "product": None
                })
                st.markdown(f"**Respuesta:** {response}")

# Mostrar historial de conversación
if st.session_state['conversation']:
    st.write("### Historial de Conversación")
    for i, entry in enumerate(reversed(st.session_state['conversation']), 1):
        st.write(f"**Pregunta {i}:** {entry['question']}")
        st.markdown(f"**Respuesta {i}:** {entry['response']}")
        if entry['product']:
            st.write(f"*Producto relacionado: {entry['product']}*")
        st.markdown("---")

# Botón para limpiar historial y resetear producto actual
if st.button("Limpiar Historial"):
    st.session_state['conversation'] = []
    st.session_state['current_product'] = None
    st.experimental_rerun()
