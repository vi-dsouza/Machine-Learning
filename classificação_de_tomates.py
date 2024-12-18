# -*- coding: utf-8 -*-
"""Classificação de tomates

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15ryMzGKNW1UxcGdkT7PlCdsIQeg8A3WK
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import keras

"""NOVA FORMULA

"""

import tensorflow as tf
import keras
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
import matplotlib.pyplot as plt

import tensorflow as tf
import matplotlib.pyplot as plt

# Carrega o dataset, com base na pasta das imagens do banco
data_dir = '/content/tomates'

# Verificar se todos os arquivos no diretório são imagens suportadas
def check_invalid_files(directory):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(supported_formats):
                print(f"Arquivo não suportado: {os.path.join(root, file)}")
                # Opcional: remover o arquivo inválido
                # os.remove(os.path.join(root, file))

# Executar a verificação
check_invalid_files(data_dir)

# Criar datasets de treino e validação
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)

# Salvar os nomes das classes
class_names = train_ds.class_names

# Normalizar os dados
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Imprimir as classes disponíveis
print("Classes disponíveis:", class_names)

# Melhorar o desempenho durante o treino

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Construir e compilar o modelo

from tensorflow.keras import layers, models

model = models.Sequential([
    normalization_layer,
    # camada convolucional
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    # pooling
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # achatar para camada densa
    layers.Flatten(),
    # camada totalmente conectada
    layers.Dense(128, activation='relu'),
    # saida com o numero de classes
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    # para rotulos inteiros (nao one-hot)
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']

)

# Treinar o modelo
history = model.fit(train_ds, validation_data=val_ds, epochs=10)
print('\n')

#Avaliar e visualizar o treinamento
import matplotlib.pyplot as plt

#Acuracia
plt.plot(history.history['accuracy'], label='Acurácia do treino')
#plt.plot(history.history['val_accuracy'], label='Acurácia da validação')
plt.legend()
plt.show()

print('\n')

#Perda
plt.plot(history.history['loss'], label='Perda do treino')
# Corrigido: Usar 'val_loss' em vez de 'val_ds' ou outro nome incorreto
#plt.plot(history.history['val_loss'], label='Perda da validação')
plt.legend()
plt.show()

#Salvar e carregar

model.save('modelo_tomate.h5')

# #Para carregar
# from tensorflow.keras.models import load_model
# model = load_model('diretorio do arquivo salvo')

#Fazer previsões
from tensorflow.keras.preprocessing import image
import numpy as np

#Carregar uma nova imagem
img_path = '/content/imagens/tomato.jpg'
img = image.load_img(img_path, target_size=(128, 128))
#Normalizar
img_array = image.img_to_array(img) / 255.0
#Adicioanr batch
img_array = np.expand_dims(img_array, axis=0)

#Fazer previsão
prediction = model.predict(img_array)
classe = class_names[np.argmax(prediction)]

print(f"\nClasse prevista: {classe}\n")

plt.imshow(img)
plt.axis('off')
plt.show()
