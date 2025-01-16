import numpy as np

class Objeto(): 
  def __init__(self):
    self.NV = 5
    self.matriz = np.array([[1, 7, 7, 1, 4], [1, 1, 1, 1, 7], [1, 1, 7, 7, 4], [1, 1, 1, 1, 1]], dtype=float)
    self.NS = 5
    self.NVPS = np.array([4, 3, 3, 3, 3], dtype=float)
    self.VS = None


def calcularVetorNormal(pontoA, pontoB, pontoC):
  vetorAB = np.array(pontoB, dtype=float) - np.array(pontoA, dtype=float)
  vetorCB = np.array(pontoC, dtype=float) - np.array(pontoB, dtype=float)

  vetorNormal = np.cross(vetorAB, vetorCB)

  return vetorNormal

def calcularMatrizPerspectiva(pontoVista, vetorNormal, pontoPlano):
  # d0 depende do plano e de um ponto sobre o plano
  d0 = np.dot(pontoPlano, vetorNormal)

  # d1 depende do plano e do ponto de vista
  d1 = np.dot(pontoVista, vetorNormal)

  d = d0 - d1

  matrizPerspectiva = [
    [(d + pontoVista[0] * vetorNormal[0]), (pontoVista[0] * vetorNormal[1]), (pontoVista[0] * vetorNormal[2]), (-pontoVista[0] * d0)], 
    [(pontoVista[1] * vetorNormal[0]), (d + pontoVista[1] * vetorNormal[1]), (pontoVista[1] * vetorNormal[2]), (-pontoVista[1] * d0)], 
    [(pontoVista[2] * vetorNormal[0]), (pontoVista[2] * vetorNormal[1]), (d + pontoVista[2] * vetorNormal[2]), (-pontoVista[2] * d0)], 
    [(vetorNormal[0]), (vetorNormal[1]), (vetorNormal[2]), (-d1)]
  ]

  return np.array(matrizPerspectiva)

def homogeneasParaCartesianasDoMundo(matrizHomogenea):
  qtdVertices = len(matrizHomogenea[0])

  for i in range(qtdVertices):
    quartoElemento = matrizHomogenea[-1][i]

    for linha in matrizHomogenea:
      teste = linha[i] / quartoElemento

      linha[i] = teste
      
  return matrizHomogenea[:2]


def main():
  objeto = Objeto()
  pontoVista = [20, 10, 30]

  # Vetor normal ao plano Z = 0
  vetorNormal = np.array([0, 0, 1], dtype=float)

  # Ponto sobre o plano Z = 0
  pontoPlano = np.array([0, 0, 0], dtype=float)

  matrizPerspectiva = calcularMatrizPerspectiva(pontoVista, vetorNormal, pontoPlano)

  matrizLinhaObjeto = np.dot(matrizPerspectiva, objeto.matriz)
  
  matrizCartesianaDoMundo = homogeneasParaCartesianasDoMundo(matrizLinhaObjeto)

  print(matrizCartesianaDoMundo)
main()