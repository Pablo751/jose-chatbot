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
    if not user_question.strip():
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
