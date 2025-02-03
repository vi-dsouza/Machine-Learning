import cv2
import numpy as np
from keras_facenet import FaceNet
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Inicializa o modelo FaceNet para extração de embeddings
embedder = FaceNet()

# Carregar o classificador Haar Cascade para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Função para pré-processar a face antes de alimentar no FaceNet
def preprocess_face(face):
    """
    Pré-processa a imagem da face para ser compatível com o modelo FaceNet.
    """
    face = cv2.resize(face, (160, 160))  # FaceNet usa imagens 160x160
    face = face.astype('float32')  # Converte para float32
    face = (face - 127.5) / 128.0  # Normaliza para a faixa [-1, 1]
    return np.expand_dims(face, axis=0)  # Adiciona dimensão extra

# Função para comparar uma face detectada com uma face conhecida
def compare_faces(face1, embedding2):
    """
    Compara uma face detectada com um embedding salvo.
    """
    face1 = preprocess_face(face1)  # Pré-processa a imagem
    embedding1 = embedder.embeddings(face1)[0]  # Extrai embedding
    distance = np.linalg.norm(embedding1 - embedding2)  # Distância Euclidiana
    return distance

# Banco de dados fictício com embeddings de faces conhecidas
database = {
    "Pessoa1": embedder.embeddings([np.random.rand(160, 160, 3)])[0],  # Simula embedding real
    "Pessoa2": embedder.embeddings([np.random.rand(160, 160, 3)])[0],
}

# Inicia a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Captura um frame do vídeo
    if not ret:
        break  # Se não conseguir ler o frame, sai do loop

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detecta faces no quadro

    for (x, y, w, h) in faces:
        # Recorta a face detectada do quadro
        detected_face = frame[y:y+h, x:x+w]

        # Percorre o banco de dados e compara cada face detectada com as cadastradas
        recognized = False
        for name, known_embedding in database.items():
            distance = compare_faces(detected_face, known_embedding)
            if distance < 0.6:  # Se a distância for menor que 0.6, reconhecemos a face
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)  # Exibe o nome
                recognized = True
                break  # Se a face for reconhecida, sai do loop

        if not recognized:
            cv2.putText(frame, "Desconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Se não reconheceu

        # Desenha um retângulo ao redor da face detectada
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibe o quadro com a detecção e reconhecimento das faces
    cv2.imshow("Reconhecimento Facial", frame)

    # Aguardar pela tecla 'q' para encerrar a aplicação
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura da câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
