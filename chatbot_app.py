# chatbot_app.py

import streamlit as st
import pandas as pd
from openai import OpenAI  
from typing import Optional, Dict, List
import numpy as np
from difflib import get_close_matches

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
            model="gpt-4o-2024-08-06",
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
        return [k.strip().lower() for k in keywords]
    except Exception as e:
        st.error(f"Error al extraer términos de búsqueda: {e}")
        return []

def search_products(df: pd.DataFrame, search_terms: List[str], threshold: float = 0.2) -> List[Dict]:
    """
    Busca productos que coincidan con los términos de búsqueda.
    Utiliza múltiples campos para la búsqueda y puntuación.
    """
    matches = []
    
    # Campos relevantes para la búsqueda
    search_fields = ['name', 'description', 'short_description', 'meta_keywords', 'additional_attributes']
    
    for _, row in df.iterrows():
        score = 0
        max_score = len(search_terms)
        
        # Combinar todos los campos de texto relevantes
        text_to_search = ' '.join(str(row[field]).lower() for field in search_fields)
        
        # Calcular coincidencias
        for term in search_terms:
            if term.lower() in text_to_search:
                score += 1
        
        # Calcular puntuación normalizada
        normalized_score = score / max_score if max_score > 0 else 0
        
        if normalized_score >= threshold:
            matches.append({
                'product': row.to_dict(),
                'score': normalized_score
            })
    
    # Ordenar por puntuación
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:5]  # Retornar los 5 mejores resultados

def generate_product_response(product_info: Dict[str, str], user_question: str) -> str:
    """
    Genera una respuesta personalizada basada en la información del producto y la pregunta del usuario.
    """
    # Construir un prompt rico en contexto
    system_prompt = """
    Eres un experto asistente de ventas para productos eléctricos Bticino.
    Proporciona respuestas detalladas y precisas sobre los productos.
    Incluye información técnica cuando sea relevante.
    Sé profesional pero amigable.
    Si no tienes información específica sobre algo, indícalo claramente.
    """
    
    user_prompt = f"""
    Información del Producto:
    Nombre: {product_info.get('name', 'N/A')}
    SKU: {product_info.get('sku', 'N/A')}
    Precio: {product_info.get('price', 'N/A')}
    Descripción: {product_info.get('description', 'N/A')}
    Descripción Corta: {product_info.get('short_description', 'N/A')}
    Atributos Adicionales: {product_info.get('additional_attributes', 'N/A')}

    Pregunta del Cliente: {user_question}

    Por favor, proporciona una respuesta útil y relevante basada en esta información.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lo siento, ocurrió un error al generar la respuesta: {e}"

# -------------------------------
# 4. Cargar y Preprocesar Datos
# -------------------------------

@st.cache_data
def load_product_data(file_path: str) -> pd.DataFrame:
    """
    Carga y preprocesa los datos de productos.
    """
    df = pd.read_csv(file_path)
    return df

# -------------------------------
# 5. Interfaz Principal
# -------------------------------

# Cargar datos
try:
    product_data = load_product_data('data/jose.csv')
    st.sidebar.success("✅ Catálogo de productos cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el catálogo de productos: {e}")
    st.stop()

# Entrada del usuario
st.write("👋 ¡Hola! Soy tu asistente de productos. ¿En qué puedo ayudarte?")
user_question = st.text_input("Escribe tu pregunta aquí:", key="user_input")

# Inicializar historial de conversación
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Procesar la pregunta
if st.button("Enviar Pregunta"):
    if not user_question:
        st.warning("Por favor, ingresa una pregunta.")
    else:
        with st.spinner("Buscando la mejor respuesta..."):
            # Extraer términos de búsqueda
            search_terms = extract_search_terms(user_question)
            
            # Buscar productos relevantes
            matches = search_products(product_data, search_terms)
            
            if matches:
                # Usar el producto más relevante para generar la respuesta
                best_match = matches[0]['product']
                response = generate_product_response(best_match, user_question)
                
                # Agregar al historial
                st.session_state['conversation'].append({
                    "question": user_question,
                    "response": response,
                    "product": best_match['name']
                })
                
                # Mostrar productos alternativos
                if len(matches) > 1:
                    st.write("📌 También podrían interesarte estos productos:")
                    for match in matches[1:]:
                        st.write(f"- {match['product']['name']}")
            else:
                response = "Lo siento, no encontré productos que coincidan con tu consulta. ¿Podrías reformular tu pregunta?"
                st.session_state['conversation'].append({
                    "question": user_question,
                    "response": response,
                    "product": None
                })

# Mostrar historial de conversación
if st.session_state['conversation']:
    st.write("### Historial de Conversación")
    for i, entry in enumerate(reversed(st.session_state['conversation']), 1):
        st.write(f"**Pregunta {i}:** {entry['question']}")
        st.write(f"**Respuesta {i}:** {entry['response']}")
        if entry['product']:
            st.write(f"*Producto relacionado: {entry['product']}*")
        st.markdown("---")

# Botón para limpiar historial
if st.button("Limpiar Historial"):
    st.session_state['conversation'] = []
    st.experimental_rerun()
