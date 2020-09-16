'''
Código elaborado a partir dos conceitos obtidos no artigo
 TensorFlow or PyTorch: Which is the best framework for programming Deep Learning networks? (disponível em: https://torres.ai/pytorch-vs-tensorflow/).
 Autora: Lauany Reis da Silva
'''
# %% [markdown]
# # Algoritmo comparativo entre TensorFlow e PyTorch
# 
# Código elaborado a partir dos conceitos obtidos no artigo  *TensorFlow or PyTorch: Which is the best framework for programming Deep Learning networks?* 
# elaborado pelo [Phd Doctor Jordi Torres](https://www.linkedin.com/in/jorditorresai/), o artigo original está disponível em: 
# [Torres.ai: PyTorch vs TensorFlow](https://torres.ai/pytorch-vs-tensorflow/). Para maiores dados sobre o assunto, consultar também o site 
# [The Gradient](https://thegradient.pub/state-of-ml-frameworks-2019-pytorch-dominates-research-tensorflow-dominates-industry/), na qual há dados 
# estatísticos comparativos entre as frameworks.

# %% [markdown]
# 0. Importando as bibliotecas necessárias:

# %% [code]
 # Bibliotecas funcionais
import numpy as np
import matplotlib.pyplot as plt

epochs = 10
batch_size = 70

# Biblioteca do TensorFlow
import tensorflow as tf

# Biblioteca do PyTorch
import torch
import torchvision
!pip install --upgrade pip
!pip install torchsummary
from torchsummary import summary

# %% [markdown]
# 1. Carregando e pré-processando os dados:

# %% [code]
# Obtendo dados do TensorFlow
(x_trainTF_, y_trainTF_), _ = tf.keras.datasets.mnist.load_data()
x_trainTF = x_trainTF_.reshape(60000, 784).astype('float32')/255
y_trainTF = tf.keras.utils.to_categorical(y_trainTF_, num_classes=10)

# %% [code]
# Obtendo dados do PyTorch
xy_trainPT = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

xy_trainPT_loader = torch.utils.data.DataLoader(xy_trainPT, batch_size=batch_size)

# %% [code]
# Verificando os dados do TensorFlow
print("TensorFlow:")
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(x_trainTF_[idx], cmap=plt.cm.binary)
    ax.set_title(str(y_trainTF_[idx]))

# %% [code]
# Verificando os dados do Pytorch
print("Pytorch:")
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    image, label = xy_trainPT [idx]
    ax.imshow(torch.squeeze(image, dim = 0).numpy(), cmap=plt.cm.binary)
    ax.set_title(str(label))

# %% [markdown]
# 2. Definindo o modelo:

# %% [code]
  # Modelo do TensorFlow com a API Keras
modelTF = tf.keras.Sequential([
                             tf.keras.layers.Dense(10,activation='sigmoid',input_shape=(784,)),
                             tf.keras.layers.Dense(10,activation='softmax')
            ])  
modelTF.summary()

# %% [code]
  # Modelo do PyTorch
modelPT=torch.nn.Sequential(torch.nn.Linear(784,10), 
                          torch.nn.Sigmoid(), 
                          torch.nn.Linear(10,10), 
                          torch.nn.LogSoftmax(dim=1) 
                         )  

summary(modelPT, (1,28,28))

# %% [markdown]
# 3. Definindo o otimizador e a função de perda:

# %% [code]
# Definindo o TensorFlow
modelTF.compile(
               loss="categorical_crossentropy",
               optimizer=tf.optimizers.SGD(lr=0.01),
               metrics = ['accuracy']
               )

# %% [code]
# Definindo o PyTorch
criterion = torch.nn.NLLLoss() 
optimizer = torch.optim.SGD(modelPT.parameters(), lr=0.01)

# %% [markdown]
# 4. Treinando o modelo:

# %% [code]
# Treinamento do TensorFlow
_ = modelTF.fit(x_trainTF, y_trainTF, epochs=epochs, batch_size=batch_size, verbose = 0)

# %% [code]
# Treinamento do PyTorch
for e in range(epochs):
    for images, labels in xy_trainPT_loader:
        images = images.view(images.shape[0], -1) 
        loss = criterion(modelPT(images), labels)        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# %% [markdown]
# 5. Avaliando o modelo:

# %% [code]
# Medindo acurácia do TensorFlow
_, (x_testTF_, y_testTF_)= tf.keras.datasets.mnist.load_data()
x_testTF = x_testTF_.reshape(10000, 784).astype('float32')/255
y_testTF = tf.keras.utils.to_categorical(y_testTF_, num_classes=10)

_ , test_accTF = modelTF.evaluate(x_testTF, y_testTF, verbose=0)
print('\n Acurácia do modelo TensorFlow = ', test_accTF)

# %% [code]
# Medindo acurácia do PyTorch

xy_testPT = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
xy_test_loaderPT  = torch.utils.data.DataLoader(xy_testPT)

correct_count, all_count = 0, 0
for images,labels in xy_test_loaderPT:
  for i in range(len(labels)):
    img = images[i].view(1, 784)

    logps = modelPT(img)
    ps = torch.exp(logps)
    probab = list(ps.detach().numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("\n Acurácia do modelo PyTorch = ", (correct_count/all_count))

# %% [markdown]
# 6. Gerando predições com o modelo 

# %% [code]
# Escolhendo uma imagem para predizer qual é o número com TensorFlow 
image = 7 
_ = plt.imshow(x_testTF_[image], cmap=plt.cm.binary)
prediction = modelTF.predict(x_testTF)
print("Predição do modelo: ", np.argmax(prediction[image]))

# %% [code]
# Escolhendo uma imagem para predizer qual é o número com PyTorch 
img = 7
image, label = xy_testPT[img]
_ = plt.imshow(torch.squeeze(image, dim = 0).numpy(), cmap=plt.cm.binary)
# Necessário implmentar funções descritas em https://github.com/davidezordan/ImageClassifier para exibir a predição

# %% [markdown]
# Considerando os códigos acima podemos concordar com o professor Jordi Torres sobre sua afirmação "Não se preocupe, 
# você começa com qualquer um, não importa qual você escolher, o importante é começar, vamos lá!". Porém, após começar 
# os estudos e implementações, percebi que tenho mais afinidade com o TensorFlow. Nada pessoal, só o achei ligeiramente 
# mais claro e prático, no entanto considero que as duas frameworks são boas ferramentas, desde que se saiba usar.
