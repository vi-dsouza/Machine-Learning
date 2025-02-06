import pyttsx3
import speech_recognition as sr
import webbrowser
import wikipedia
from geopy.geocoders import Nominatim
import os
import time

# --- Função de Texto para Fala (Text to Speech) ---
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Velocidade da fala
    engine.setProperty('volume', 1)  # Volume da fala
    engine.say(text)
    engine.runAndWait()

# --- Função de Fala para Texto (Speech to Text) ---
def speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Diga algo...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language='pt-BR')  # Reconhecimento em português
        print(f"Você disse: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Não entendi. Tente novamente.")
        return None
    except sr.RequestError:
        print("Não foi possível conectar ao serviço de reconhecimento de fala.")
        return None

# --- Função para abrir o YouTube ---
def open_youtube():
    text_to_speech("Abrindo o YouTube para você.")
    webbrowser.open("https://www.youtube.com")

# --- Função para pesquisar no Wikipedia ---
def search_wikipedia(query):
    try:
        result = wikipedia.summary(query, sentences=2, lang="pt")
        print(result)
        text_to_speech(result)
    except wikipedia.exceptions.DisambiguationError as e:
        text_to_speech(f"Eu encontrei várias opções, mas não posso decidir qual. Você pode ser mais específico?")
    except wikipedia.exceptions.HTTPTimeoutError:
        text_to_speech("Houve um erro ao acessar a Wikipedia. Tente novamente mais tarde.")
    except Exception as e:
        text_to_speech("Desculpe, não encontrei nada sobre isso.")

# --- Função para encontrar a farmácia mais próxima ---
def find_nearest_pharmacy():
    geolocator = Nominatim(user_agent="assistente_virtual")
    location = geolocator.geocode("Farmácia mais próxima")

    if location:
        print(f"Localização da farmácia mais próxima: {location.address}")
        text_to_speech(f"A farmácia mais próxima está localizada em: {location.address}")
    else:
        text_to_speech("Não consegui encontrar a farmácia mais próxima no momento.")

# --- Função principal de Comandos de Voz ---
def main():
    text_to_speech("Olá, eu sou o seu assistente virtual. Como posso ajudá-lo?")

    while True:
        command = speech_to_text()

        if command:
            if "pesquisar" in command and "wikipedia" in command:
                query = command.replace("pesquisar", "").replace("no wikipedia", "").strip()
                search_wikipedia(query)

            elif "youtube" in command:
                open_youtube()

            elif "farmácia" in command and "próxima" in command:
                find_nearest_pharmacy()

            elif "parar" in command or "sair" in command:
                text_to_speech("Até logo! Tenha um bom dia.")
                break

            else:
                text_to_speech("Desculpe, não entendi o comando. Pode repetir?")

        time.sleep(2)  # Espera um pouco antes de escutar o próximo comando

if __name__ == "__main__":
    main()
