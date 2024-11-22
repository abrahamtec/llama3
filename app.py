import streamlit as st
import pandas as pd
import numpy as np
from langchain_ollama import OllamaLLM
import openai
from fuzzywuzzy import process
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Configuración específica para Azure OpenAI
openai.api_type = "azure"
openai.api_key = "9C1zXmuBW4G65loM8KKPq04h2p4Gu2PWfkWBbq79anL9bZLa88gzJQQJ99AJAC4f1cMXJ3w3AAABACOGzNM6"  # Reemplaza con tu clave de Azure OpenAI
openai.api_base = "https://demo-master-ia.openai.azure.com/"  # Reemplaza con el endpoint de tu servicio
openai.api_version = "2023-05-15"  # Asegúrate de que esta sea la versión correcta para tu recurso

# Cargar datos desde el archivo CSV
df = pd.read_csv('test_question.csv')  # Asegúrate de tener columnas: 'Question', 'Tema', 'Subtema'

# Inicializar el modelo Llama 3.2 y el modelo Word2Vec
llm_llama = OllamaLLM(model="llama3.2")
data_sentences = df[['Question', 'Tema', 'Answers']].dropna().values.tolist()
word2vec_model = Word2Vec(sentences=[[word.lower() for word in q.split()] for q, _, _ in data_sentences], vector_size=100, window=5, min_count=1, workers=4)

# Inicializar el estado de la sesión para acumular similitudes
if 'similitud_acumulada' not in st.session_state:
    st.session_state.similitud_acumulada = pd.DataFrame(columns=["Pregunta", "Similitud_Llama", "Similitud_OpenAI"])

# Función para generar el prompt RAG
def generar_prompt_rag(pregunta, df):
    preguntas_df = df['Question'].tolist()
    mejor_coincidencia = process.extractOne(pregunta, preguntas_df)
    if mejor_coincidencia[1] >= 70:
        index = preguntas_df.index(mejor_coincidencia[0])
        tema, subtema = df.iloc[index][['Tema', 'Answers']]
        contexto = f"Temas disponibles: Tema: {tema}, Subtema: {subtema}"
    else:
        contexto = "\n".join([f"Tema: {row['Tema']}, Subtema: {row['Subtema']}" for _, row in df.iterrows()])

    prompt = f"""
    Basado en la siguiente información, responde la pregunta con el tema y subtema correctos:

    {contexto}

    Pregunta: {pregunta}
    Tema:
    Subtema:
    """
    return prompt

# Función para procesar respuesta Llama 3.2
def obtener_respuesta_llama(pregunta):
    prompt = generar_prompt_rag(pregunta, df)
    respuesta = llm_llama.generate(prompts=[prompt])
    return respuesta.generations[0][0].text.strip()

# Función para procesar respuesta GPT-4
def obtener_respuesta_openai(pregunta):
    temas_y_subtemas = "\n".join(
        [f"Tema: {row['Tema']}, Subtema: {row['Answers']}" for _, row in df.iterrows()]
    )
    
    prompt_openai = (
        f"Aquí tienes una pregunta sobre temas estudiantiles:\n\n"
        f"Pregunta: {pregunta}\n\nTemas y Subtemas disponibles:\n{temas_y_subtemas}\n\n"
        "¿Podrías proporcionar una respuesta apropiada a esta pregunta?"
    )

    response = openai.ChatCompletion.create(
        engine="gpt4o-demo-master-ia-model",  # Asegúrate de que el nombre del modelo esté correcto
        messages=[
            {"role": "system", "content": "Eres un asistente que ayuda con temas estudiantiles."},
            {"role": "user", "content": prompt_openai}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

# Función para calcular la similitud coseno
def calcular_similitud(pregunta, modelo, data_sentences):
    pregunta_vec = np.mean([modelo.wv[word] for word in pregunta.split() if word in modelo.wv] or [np.zeros(100)], axis=0)
    similitudes = [cosine_similarity([pregunta_vec], [np.mean([modelo.wv[word] for word in q.split() if word in modelo.wv], axis=0)])[0][0] for q, _, _ in data_sentences]
    return similitudes

# Configuración de la app en Streamlit
st.title("Comparador de Respuestas de Modelos (Llama 3.2 y GPT-4)")
pregunta = st.text_input("Haz una pregunta:")

# Mostrar respuestas de ambos modelos y limpiar campo después de enviar pregunta
if st.button("Enviar Pregunta"):
    if pregunta:
        st.write("**Respuesta de Llama 3.2:**")
        respuesta_llama = obtener_respuesta_llama(pregunta)
        st.write(respuesta_llama)

        st.write("**Respuesta de GPT-4:**")
        respuesta_openai = obtener_respuesta_openai(pregunta)
        st.write(respuesta_openai)

        # Calcular y mostrar la similitud coseno
        similitud_llama = calcular_similitud(pregunta, word2vec_model, data_sentences)[0]  # Ajusta según sea necesario
        similitud_openai = calcular_similitud(pregunta, word2vec_model, data_sentences)[1]  # Ajusta según sea necesario
        
        # Almacenar la similitud en el estado de la sesión
        nueva_similitud = pd.DataFrame([[pregunta, similitud_llama, similitud_openai]], columns=["Pregunta", "Similitud_Llama", "Similitud_OpenAI"])
        st.session_state.similitud_acumulada = pd.concat([st.session_state.similitud_acumulada, nueva_similitud], ignore_index=True)

        # Graficar similitudes
        st.write("### Similitud Coseno (Puntaje RAG)")
        if not st.session_state.similitud_acumulada.empty:
            fig, ax = plt.subplots()
            ax.plot(st.session_state.similitud_acumulada.index, st.session_state.similitud_acumulada["Similitud_Llama"], label="Similitud Coseno Llama 3.2", marker='o')
            ax.plot(st.session_state.similitud_acumulada.index, st.session_state.similitud_acumulada["Similitud_OpenAI"], label="Similitud Coseno GPT-4", marker='o')
            ax.set_xlabel("Índice de Pregunta")
            ax.set_ylabel("Similitud Coseno")
            ax.set_title("Puntaje RAG - Similitud Coseno entre preguntas")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No hay datos suficientes para graficar.")

        # Limpiar el campo de la pregunta después de mostrar la respuesta
        pregunta = ""
