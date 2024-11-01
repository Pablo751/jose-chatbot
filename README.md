# Chatbot de Productos Bticino

Un asistente virtual inteligente para consultas sobre productos Bticino.

## Configuración

1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/proyecto-chatbot.git
cd proyecto-chatbot
```

2. Instalar dependencias
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno
Crear archivo `.streamlit/secrets.toml` con:
```toml
OPENAI_API_KEY = "tu-api-key"
```

4. Ejecutar la aplicación
```bash
streamlit run chatbot_app.py
```

## Estructura del Proyecto

- `chatbot_app.py`: Aplicación principal
- `data/jose.csv`: Catálogo de productos
- `requirements.txt`: Dependencias del proyecto