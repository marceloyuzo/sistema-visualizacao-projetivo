import numpy as np

def calcularVetorNormal(pontoA, pontoB, pontoC):
  vetorAB = np.array(pontoB) - np.array(pontoA)
  vetorCB = np.array(pontoC) - np.array(pontoB)

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

def homogeneasParaCartesianas(matrizHomogenea):
  print(matrizHomogenea)
  return matrizHomogenea

def main():
  matrizObjeto = np.array([[1, 7, 7, 1, 4], [1, 1, 1, 1, 7], [1, 1, 7, 7, 4], [1, 1, 1, 1, 1]])
  pontoVista = [20, 10, 30]

  # Vetor normal ao plano Z = 0
  vetorNormal = np.array([0, 0, 1])

  # Ponto sobre o plano Z = 0
  pontoPlano = np.array([0, 0, 0])

  matrizPerspectiva = calcularMatrizPerspectiva(pontoVista, vetorNormal, pontoPlano)

  matrizLinhaObjeto = np.dot(matrizPerspectiva, matrizObjeto)
  
  matrizCartesiana = homogeneasParaCartesianas(matrizLinhaObjeto)

main()