import numpy as np
import cv2

# Especificar o caminho da imagem diretamente no código
image_path = "Reducao de dimencionalidade/imagens/animal_de_oculos.jpg"

# Carrega a imagem
image = cv2.imread(image_path)
if image is None:
    print("Erro: Caminho da imagem inválido ou arquivo não encontrado.")
    exit(1)

cv2.imshow("Imagem original", image)

# Converte para escala de cinza
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplica desfoque gaussiano
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Mostrar a imagem em tons de cinza
cv2.imshow("Imagem em tons de cinza", image)

# Limiarização adaptativa usando o método da média
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Limiar_Media", thresh)

# Limiarização adaptativa usando o método gaussiano
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Limiar_Gaussian", thresh)

cv2.waitKey(0)
