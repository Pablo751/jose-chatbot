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
import openai

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
openai.api_key = st.secrets["OPENAI_API_KEY"]

def call_gpt4o(prompt: str, max_tokens: int = 500) -> str:
    """
    Llama a la API de OpenAI GPT-4o para generar una respuesta basada en el prompt.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-08-06",  # Asegúrate de usar la versión correcta
            messages=[
                {"role": "system", "content": """
                Eres un asistente de ventas que ayuda a los clientes a encontrar productos en nuestra tienda.
                Utiliza únicamente la información proporcionada sobre el producto para responder. No inventes ni agregues detalles adicionales.
                """},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Baja temperatura para respuestas más precisas
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error al llamar a GPT-4o: {e}")
        return "Lo siento, ocurrió un error al generar la respuesta. Por favor, inténtalo de nuevo más tarde."

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
def generate_embeddings_faiss_ivf(df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2', nlist: int = 100) -> (np.ndarray, faiss.Index):
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
        self.file_path = file_path
    
    def on_modified(self, event):
        if event.src_path.endswith(self.file_path):
            logger.info(f"Archivo {self.file_path} modificado.")
            st.session_state['file_changed'] = True

def start_file_watcher(file_path: str):
    event_handler = CSVChangeHandler(file_path)
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
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

def search_products_faiss(query: str, model: SentenceTransformer, index: faiss.Index, df: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """
    Busca productos que coincidan con la consulta utilizando FAISS y embeddings.
    """
    try:
        if not query.strip():
            logger.warning("Consulta vacía después de filtrar stop words.")
            return []
        
        # Generar embedding para la consulta
        query_embedding = model.encode([query], normalize_embeddings=True)
        # Realizar búsqueda en el índice FAISS
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

def generate_product_response(product_info: Dict[str, str]) -> str:
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
    
    # Crear el prompt para GPT-4o
    prompt = f"""
    Utilizando únicamente la siguiente información sobre el producto, genera una descripción conversacional y atractiva para el cliente:

    {product_summary}
    """
    
    # Llamar a GPT-4o para generar la respuesta
    response = call_gpt4o(prompt, max_tokens=500)
    
    return response
