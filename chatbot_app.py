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
