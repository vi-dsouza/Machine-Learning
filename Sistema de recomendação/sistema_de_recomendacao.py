import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# --- 1. Preparação do Modelo ---
# Carregar ResNet50 pré-treinada sem as camadas finais
base_model = ResNet50(weights='imagenet', include_top=False)

# Adicionar camadas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

# 4 classes
predictions = Dense(4, activation='softmax')(x)  

# Criar o modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Agora, o modelo foi completamente definido e suas camadas "construídas"
model.summary()  # Para ver a estrutura final do modelo

# Congelar as camadas da ResNet50 para treinamento apenas nas camadas finais
for layer in base_model.layers:
    layer.trainable = False

# Compilar o modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# --- 2. Carregar Dados e Treinar o Modelo ---
# Verificar a quantidade de imagens nos diretórios de treino e teste
diretorio_treino = 'Sistema de recomendação/images_treino'
diretorio_teste = 'Sistema de recomendação/imagens_teste'

# Gerador de dados para treinamento e validação
train_datagen = ImageDataGenerator(rescale=1./255)  # Normalizando as imagens
test_datagen = ImageDataGenerator(rescale=1./255)

# Geradores de treino e validação
train_generator = train_datagen.flow_from_directory(
    diretorio_treino,
    target_size=(224, 224),  # Tamanho das imagens
    batch_size=32,
    class_mode='categorical',  # Classes como uma distribuição one-hot
    shuffle=True  # Embaralhando as imagens para o treinamento
)

validation_generator = test_datagen.flow_from_directory(
    diretorio_teste,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Não embaralhando as imagens de validação
)

# Treinamento
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# --- 3. Funções para Extração de Características e Recomendação ---

# Função para extrair o vetor de características de uma imagem
def extrair_caracteristicas(imagem_path, modelo):
    img = image.load_img(imagem_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Para o ResNet50

    # Obter as características da imagem (camada intermediária)
    features = modelo.predict(img_array)
    return features.flatten()  # Achatar o vetor de características

# Função para buscar a imagem mais similar e retornar a classe
def buscar_imagem_similar(features_test, features_database, class_names):
    distancias = euclidean_distances([features_test], features_database)
    indice_similar = np.argmin(distancias)  # Encontrar o índice da menor distância
    return indice_similar, class_names[indice_similar]

# --- 4. Teste de Recomendação de Imagens ---

# Carregar as características das imagens do banco de dados
imagens_database = ['Sistema de recomendação/images_treino/bone/bone.jpg', 'Sistema de recomendação/images_treino/relogio/rrelogio.jpg', 'Sistema de recomendação/images_treino/carro/carro.jpg', 'Sistema de recomendação/images_treino/urso/urso.jpg']  # Adicione as imagens do seu banco de dados
features_database = [extrair_caracteristicas(img, model) for img in imagens_database]

# Obter as classes associadas
class_names = train_generator.class_indices
class_names = {v: k for k, v in class_names.items()}  # Reverter o dicionário para nome -> índice

# Escolher uma imagem de teste
imagem_test_path = 'Sistema de recomendação/imagens_teste/carro/carro.jpg'  # Substitua pelo caminho da sua imagem de teste
features_test_image = extrair_caracteristicas(imagem_test_path, model)

# Calcular as distâncias e obter a classe da imagem mais similar
indice_similar, nome_classe = buscar_imagem_similar(features_test_image, features_database, class_names)

# Mostrar a imagem mais semelhante
img = image.load_img(imagens_database[indice_similar], target_size=(224, 224))
plt.imshow(img)
plt.title(f"Imagem Similar: {nome_classe}")
plt.show()
