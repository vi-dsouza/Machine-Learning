import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
import matplotlib.pyplot as plt

# Carrega o dataset
data_dir = '/content/sample_data/tomates'

# Criar datasets de treino e validação
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
)

# Salvar os nomes das classes
class_names = train_ds.class_names

# Preparar os dados / Normalizar
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Visualizar imagens
def plot_images(dataset, num_images=9):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):  # Obtém um único lote do dataset
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)  # Grid de 3x3
            plt.imshow(images[i].numpy())  # Converte tensor para imagem
            plt.title(class_names[labels[i]])  # Exibe o rótulo da classe
            plt.axis("off")
    plt.show()

# Visualizar as imagens de treino
plot_images(train_ds)
print(class_names)


#Melhorar o desempenho durante o treino

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

#Construir e compilar o modelo

from tensorflow.keras import layers, models

model = models.Sequential([
    normalization_layer,
    #camada convolucional
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    #pooling
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    #achatar para camada densa
    layers.Flatten(),
    #camada totalmente conectada
    layers.Dense(128, activation='relu'),
    #saida com o numero de classes
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    #para rotulos inteiros (nao one-hot)
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Treinar o modelo

history = model.fit(
    train_ds,
    validation_data=val_ds,
    #Numero de epocas
    epochs=10
)



#Avaliar e visualizar o treinamento

import matplotlib.pyplot as plt

#Acuracia
plt.plot(history.history['accuracy'], label='Acurácia do treino')
plt.plot(history.history['val_accuracy'], label='Acurácia da validação')
plt.legend()
plt.show()

#Perda
plt.plot(history.history['loss'], label='Perda do treino')
plt.plot(history.history['val_loss'], label='Perda da validação')
plt.legend()
plt.show()


# #Salvar e carregar

# model.save('modelo_tomate.h5')

# #Para carregar
# from tensorflow.keras.models import load_model
# model = load_model('diretorio do arquivo salvo')

#Fazer previsões
from tensorflow.keras.preprocessing import image
import numpy as np

#Carregar uma nova imagem
img_path = 'diretorio da imagem'
img = image.load_img(img_pth, target_size=(128, 128))
#Normalizar
img_array = image.img_to_array(img) / 255.0
#Adicioanr batch
img_array = np.expand_dims(img_array, axis=0)

#Fazer previsão
prediction = model.predict(img_array)
classe = train_ds.class_names[np.argmax(prediction)]

print(f"Classe prevista: {classe}")

plt.imshow(img)
plt.axis('off')
plt.show()

