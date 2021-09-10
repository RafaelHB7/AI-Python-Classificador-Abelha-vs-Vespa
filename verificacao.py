import numpy

from keras.preprocessing import image
from keras.models import model_from_json

arquivo = open('Versao 3/classificador_insetos.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('Versao 3/classificador_insetos.h5')

imagem_teste = image.load_img('imagens/eu.jpg', target_size=(64,64))

imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255

imagem_teste = numpy.expand_dims(imagem_teste, axis = 0)

previsao = classificador.predict(imagem_teste)
prev = (previsao > 0.5)

if prev:
    print("Vespa")
else:
    print("Abelha")