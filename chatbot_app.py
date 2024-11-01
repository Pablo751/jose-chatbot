# chatbot_app.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from typing import Optional, Dict, List
import numpy as np
import re

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
# 2. Cargar y Preprocesar Datos
# -------------------------------

@st.cache_data(show_spinner=False)
def load_product_data(file_path: str) -> pd.DataFrame:
    """
    Carga y preprocesa los datos de productos con solo las columnas esenciales.
    """
    columns_to_load = ['sku', 'name', 'description', 'short_description', 'price', 'additional_attributes', 'base_image', 'url_key']
    df = pd.read_csv(file_path, usecols=columns_to_load)
    
    # Limpiar y rellenar valores nulos
    df.fillna({'additional_attributes': 'Información no disponible', 
              'short_description': 'Información no disponible',
              'description': 'Información no disponible',
              'price': 0.0,
              'base_image': '',
              'url_key': '#'}, inplace=True)
    
    return df

def validate_csv(df: pd.DataFrame) -> bool:
    """
    Valida que el DataFrame tenga las columnas esenciales y que no falten datos críticos.
    """
    expected_columns = ['sku', 'name', 'description', 'short_description', 'price', 'additional_attributes', 'base_image', 'url_key']
    if not all(column in df.columns for column in expected_columns):
        st.error("El CSV no contiene todas las columnas requeridas.")
        return False
    
    # Verificar valores nulos en columnas esenciales
    essential_columns = ['sku', 'name', 'price']
    if df[essential_columns].isnull().any().any():
        st.warning("Algunos productos tienen información incompleta. Revisar el CSV.")
    
    return True

# Cargar datos
product_file = 'data/jose.csv'
try:
    product_data = load_product_data(product_file)
    if not validate_csv(product_data):
        st.stop()
    st.sidebar.success("✅ Catálogo de productos cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el catálogo de productos: {e}")
    st.stop()

# -------------------------------
# 3. Generar Embeddings y Configurar FAISS
# -------------------------------

@st.cache_resource(show_spinner=False)
def generate_embeddings(df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2') -> (np.ndarray, faiss.Index):
    """
    Genera embeddings para los productos y configura el índice FAISS.
    """
    model = SentenceTransformer(model_name)
    # Concatenar campos relevantes para generar el embedding
    texts = df['name'] + " " + df['description'] + " " + df['short_description']
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    
    # Normalizar embeddings
    faiss.normalize_L2(embeddings)
    
    # Crear índice FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity after normalization)
    index.add(embeddings)
    
    return embeddings, index

try:
    embeddings, faiss_index = generate_embeddings(product_data)
except Exception as e:
    st.error(f"Error al generar embeddings: {e}")
    st.stop()

# -------------------------------
# 4. Funciones de Búsqueda y Coincidencia
# -------------------------------

def extract_search_terms(user_query: str) -> List[str]:
    """
    Extrae términos relevantes de búsqueda de la consulta del usuario.
    Se pueden implementar reglas adicionales o utilizar NLP avanzado aquí.
    """
    # Convertir a minúsculas
    user_query = user_query.lower()
    # Remover caracteres especiales
    user_query = re.sub(r'[^\w\s]', '', user_query)
    # Tokenizar
    terms = user_query.split()
    # Filtrar palabras irrelevantes
    stop_words = {'de', 'un', 'una', 'me', 'puedes', 'recomendar', 'economico', 'económico', 'por', 'que', 'que', 'puedo', 'el', 'la', 'los', 'las'}
    filtered_terms = [term for term in terms if term not in stop_words]
    return filtered_terms

def search_products_faiss(query: str, model: SentenceTransformer, index: faiss.Index, df: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """
    Busca productos que coincidan con la consulta utilizando FAISS y embeddings.
    """
    # Generar embedding para la consulta
    query_embedding = model.encode([query], normalize_embeddings=True)
    # Realizar búsqueda en el índice FAISS
    D, I = index.search(query_embedding, top_k)
    results = []
    for distance, idx in zip(D[0], I[0]):
        if distance > 0:  # Similaridad positiva
            product = df.iloc[idx].to_dict()
            results.append({'product': product, 'score': distance})
    return results

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
    Genera una respuesta personalizada basada únicamente en la información del producto.
    Incluye formato Markdown para una mejor presentación.
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
    if base_image:
        image_url = f"https://tutienda.com/{base_image}"  # Reemplaza con la URL base de tus imágenes
    else:
        image_url = "https://via.placeholder.com/150"  # Imagen por defecto
    
    # Construir la respuesta con formato Markdown enriquecido
    response = f"""
**Producto:** [{product_info.get('name', 'Información no disponible')}]({product_url})

![{product_info.get('name', 'Producto')}]({image_url})

**Descripción:** {product_info.get('short_description', 'Información no disponible')}

**Precio:** {price_formatted}

**Características:**
{features_formatted}

**¿En qué más puedo ayudarte sobre este producto?**
"""
    return response

# -------------------------------
# 5. Interfaz Principal con Soporte de Follow-Up
# -------------------------------

# Inicializar historial de conversación y contexto del producto
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'current_product' not in st.session_state:
    st.session_state['current_product'] = None  # Producto actual para preguntas de seguimiento

# Entrada del usuario
st.write("👋 ¡Hola! Soy tu asistente de productos. ¿En qué puedo ayudarte?")
user_question = st.text_input("Escribe tu pregunta aquí:", key="user_input")

# Botón para enviar la pregunta
if st.button("Enviar Pregunta"):
    if not user_question:
        st.warning("Por favor, ingresa una pregunta.")
    else:
        with st.spinner("Buscando la mejor respuesta..."):
            # Extraer términos de búsqueda
            search_terms = extract_search_terms(user_question)
            st.write(f"🔍 **Términos de Búsqueda:** {', '.join(search_terms)}")  # Mostrar términos extraídos
            
            # Reconstruir la consulta sin stop words para generar una búsqueda más efectiva
            reconstructed_query = ' '.join(search_terms)
            
            # Cargar el modelo para generar embeddings de la consulta
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Buscar productos relevantes usando FAISS
            matches = search_products_faiss(reconstructed_query, model, faiss_index, product_data, top_k=5)
            
            if matches:
                # Usar el producto más relevante para generar la respuesta
                best_match = matches[0]['product']
                
                # Verificar si esta es una pregunta de seguimiento
                if st.session_state['current_product'] and "más" in user_question.lower():
                    # Utilizar el mismo producto para la pregunta de seguimiento
                    response = generate_product_response(st.session_state['current_product'])
                else:
                    # Actualizar el producto actual en el estado de sesión
                    st.session_state['current_product'] = best_match
                    response = generate_product_response(best_match)
                
                # Agregar al historial
                st.session_state['conversation'].append({
                    "question": user_question,
                    "response": response,
                    "product": best_match['name']
                })
                
                # Mostrar la respuesta utilizando Markdown para un mejor formato
                st.markdown(response)
                
                # Mostrar productos alternativos
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
