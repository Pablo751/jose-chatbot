# chatbot_app.py

import streamlit as st
import pandas as pd
from openai import OpenAI  
from typing import Optional, Dict, List
import numpy as np
from difflib import get_close_matches
from fuzzywuzzy import fuzz

# -------------------------------
# 1. Configurar el Cliente OpenAI
# -------------------------------

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# -------------------------------
# 2. Configuración de la Aplicación Streamlit
# -------------------------------

st.set_page_config(
    page_title="💬 Asistente de Productos",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("💬 Asistente de Productos")

# -------------------------------
# 3. Funciones de Búsqueda y Coincidencia
# -------------------------------

def extract_search_terms(user_query: str) -> List[str]:
    """
    Utiliza OpenAI para extraer términos relevantes de búsqueda de la consulta del usuario.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """
                Extrae palabras clave relevantes para búsqueda de productos. 
                Enfócate en:
                - Nombre del producto
                - Características técnicas
                - Color
                - Marca
                - Tipo de producto
                Devuelve solo las palabras clave separadas por comas, sin explicaciones.
                """},
                {"role": "user", "content": f"Extract keywords from: {user_query}"}
            ],
            temperature=0.3,
            max_tokens=100
        )
        keywords = response.choices[0].message.content.split(',')
        # Filtrar palabras vacías comunes
        stop_words = {'de', 'un', 'una', 'me', 'puedes', 'recomendar', 'economico', 'económico'}
        filtered_keywords = [k.strip().lower() for k in keywords if k.strip().lower() not in stop_words]
        return filtered_keywords
    except Exception as e:
        st.error(f"Error al extraer términos de búsqueda: {e}")
        return []

def search_products(df: pd.DataFrame, search_terms: List[str], threshold: float = 70) -> List[Dict]:
    """
    Busca productos que coincidan con los términos de búsqueda utilizando fuzzy matching.
    """
    matches = []
    
    # Campos relevantes para la búsqueda
    search_fields = ['name', 'description', 'short_description', 'additional_attributes']
    
    for _, row in df.iterrows():
        score = 0
        max_score = len(search_terms)
        
        # Combinar todos los campos de texto relevantes
        text_to_search = ' '.join(str(row[field]).lower() for field in search_fields if field in df.columns)
        
        # Calcular coincidencias usando fuzzy matching
        for term in search_terms:
            term = term.lower()
            # Dividir el texto en palabras para comparar individualmente
            words = text_to_search.split()
            best_ratio = max([fuzz.partial_ratio(term, word) for word in words] or [0])
            if best_ratio >= threshold:
                score += 1
        
        # Calcular puntuación normalizada
        normalized_score = score / max_score if max_score > 0 else 0
        
        if normalized_score >= (threshold / 100):
            matches.append({
                'product': row.to_dict(),
                'score': normalized_score
            })
    
    # Ordenar por puntuación
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:5]  # Retornar los 5 mejores resultados

def generate_product_response(product_info: Dict[str, str], user_question: str) -> str:
    """
    Genera una respuesta personalizada basada únicamente en la información del producto.
    """
    response = f"""
    **Producto:** {product_info.get('name', 'Información no disponible')}
    
    **Descripción:** {product_info.get('short_description', 'Información no disponible')}
    
    **Precio:** ${product_info.get('price', 'Información no disponible')}
    
    **Características:**
    {product_info.get('additional_attributes', 'Información no disponible')}
    
    **¿En qué más puedo ayudarte sobre este producto?**
    """
    return response

# -------------------------------
# 4. Cargar y Preprocesar Datos
# -------------------------------

@st.cache_data
def load_product_data(file_path: str) -> pd.DataFrame:
    """
    Carga y preprocesa los datos de productos con solo las columnas esenciales.
    """
    columns_to_load = ['sku', 'name', 'description', 'short_description', 'price', 'additional_attributes']
    df = pd.read_csv(file_path, usecols=columns_to_load)
    
    # Verificar valores nulos
    if df[columns_to_load].isnull().any().any():
        st.warning("Algunos productos tienen información incompleta. Revisar el CSV.")
    
    return df

def validate_csv(file_path: str) -> bool:
    try:
        df = pd.read_csv(file_path)
        # Verificar si todas las filas tienen el número correcto de columnas
        expected_columns = ['sku', 'store_view_code', 'attribute_set_code', 'product_type', 'categories', 'product_websites', 'name', 'description', 'short_description', 'weight', 'product_online', 'tax_class_name', 'visibility', 'price', 'special_price', 'special_price_from_date', 'special_price_to_date', 'url_key', 'meta_title', 'meta_keywords', 'meta_description', 'base_image', 'base_image_label', 'small_image', 'small_image_label', 'thumbnail_image', 'thumbnail_image_label', 'swatch_image', 'swatch_image_label', 'created_at', 'updated_at', 'new_from_date', 'new_to_date', 'display_product_options_in', 'map_price', 'msrp_price', 'map_enabled', 'gift_message_available', 'custom_design', 'custom_design_from', 'custom_design_to', 'custom_layout_update', 'page_layout', 'product_options_container', 'msrp_display_actual_price_type', 'country_of_manufacture', 'activity', 'gender', 'size', 'product_tag', 'manufacturer', 'additional_attributes', 'qty', 'out_of_stock_qty', 'use_config_min_qty', 'is_qty_decimal', 'allow_backorders', 'use_config_backorders', 'min_cart_qty', 'use_config_min_sale_qty', 'max_cart_qty', 'use_config_max_sale_qty', 'is_in_stock', 'notify_on_stock_below', 'use_config_notify_stock_qty', 'manage_stock', 'use_config_manage_stock', 'use_config_qty_increments', 'qty_increments', 'use_config_enable_qty_inc', 'enable_qty_increments', 'is_decimal_divided', 'website_id', 'related_skus', 'related_position', 'crosssell_skus', 'crosssell_position', 'upsell_skus', 'upsell_position', 'additional_images', 'additional_image_labels', 'hide_from_product_page', 'bundle_price_type', 'bundle_sku_type', 'bundle_price_view', 'bundle_weight_type', 'bundle_values', 'bundle_shipment_type', 'associated_skus', 'downloadable_links', 'downloadable_samples', 'configurable_variations', 'configurable_variation_labels']
        
        if list(df.columns) != expected_columns:
            st.error("Las columnas del CSV no coinciden con las esperadas.")
            return False
        
        # Verificar valores nulos en columnas esenciales
        essential_columns = ['sku', 'name', 'description', 'short_description', 'price', 'additional_attributes']
        if df[essential_columns].isnull().any().any():
            st.warning("Algunos productos tienen información incompleta. Revisar el CSV.")
        
        return True
    except Exception as e:
        st.error(f"Error al validar el CSV: {e}")
        return False

# -------------------------------
# 5. Interfaz Principal con Soporte de Follow-Up
# -------------------------------

# Cargar y validar datos
try:
    if validate_csv('data/jose.csv'):
        product_data = load_product_data('data/jose.csv')
        st.sidebar.success("✅ Catálogo de productos cargado correctamente")
    else:
        st.error("El CSV no pasó la validación. Por favor, revisa el archivo.")
        st.stop()
except Exception as e:
    st.error(f"Error al cargar el catálogo de productos: {e}")
    st.stop()

# Entrada del usuario
st.write("👋 ¡Hola! Soy tu asistente de productos. ¿En qué puedo ayudarte?")
user_question = st.text_input("Escribe tu pregunta aquí:", key="user_input")

# Inicializar historial de conversación y contexto del producto
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'current_product' not in st.session_state:
    st.session_state['current_product'] = None  # Producto actual para preguntas de seguimiento

# Procesar la pregunta
if st.button("Enviar Pregunta"):
    if not user_question:
        st.warning("Por favor, ingresa una pregunta.")
    else:
        with st.spinner("Buscando la mejor respuesta..."):
            # Extraer términos de búsqueda
            search_terms = extract_search_terms(user_question)
            st.write(f"🔍 **Términos de Búsqueda:** {', '.join(search_terms)}")  # Mostrar términos extraídos
            
            # Buscar productos relevantes
            matches = search_products(product_data, search_terms)
            
            if matches:
                # Usar el producto más relevante para generar la respuesta
                best_match = matches[0]['product']
                
                # Verificar si esta es una pregunta de seguimiento
                if st.session_state['current_product'] and "más" in user_question.lower():
                    # Utilizar el mismo producto para la pregunta de seguimiento
                    response = generate_product_response(st.session_state['current_product'], user_question)
                else:
                    # Actualizar el producto actual en el estado de sesión
                    st.session_state['current_product'] = best_match
                    response = generate_product_response(best_match, user_question)
                
                # Agregar al historial
                st.session_state['conversation'].append({
                    "question": user_question,
                    "response": response,
                    "product": best_match['name']
                })
                
                # Mostrar respuesta
                st.write(f"**Respuesta:** {response}")
                
                # Mostrar productos alternativos
                if len(matches) > 1:
                    st.write("📌 También podrían interesarte estos productos:")
                    for match in matches[1:]:
                        st.write(f"- **{match['product']['name']}** - ${match['product']['price']}")
            else:
                response = "Lo siento, no encontré productos que coincidan con tu consulta. ¿Podrías reformular tu pregunta?"
                st.session_state['conversation'].append({
                    "question": user_question,
                    "response": response,
                    "product": None
                })
                st.write(f"**Respuesta:** {response}")

# Mostrar historial de conversación
if st.session_state['conversation']:
    st.write("### Historial de Conversación")
    for i, entry in enumerate(reversed(st.session_state['conversation']), 1):
        st.write(f"**Pregunta {i}:** {entry['question']}")
        st.write(f"**Respuesta {i}:** {entry['response']}")
        if entry['product']:
            st.write(f"*Producto relacionado: {entry['product']}*")
        st.markdown("---")

# Botón para limpiar historial y resetear producto actual
if st.button("Limpiar Historial"):
    st.session_state['conversation'] = []
    st.session_state['current_product'] = None
    st.experimental_rerun()
