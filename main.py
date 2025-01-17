import numpy as np
import matplotlib.pyplot as plt

class Objeto(): 
   def __init__(self):
      self.NV = 5
      self.matriz = np.array([[1, 7, 7, 1, 4], [1, 1, 1, 1, 7], [1, 1, 7, 7, 4], [1, 1, 1, 1, 1]], dtype=float)
      self.NS = 5
      self.NVPS = np.array([4, 3, 3, 3, 3], dtype=float)
      self.VS = [[4, 3, 5, 4], [3, 2, 5, 3], [5, 2, 1, 5], [4, 5, 1, 4], [4, 1, 2, 3, 4]]

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

   matrizCartesianaDoMundo = matrizHomogenea[:2]
   matrizCartesianaDoMundo[1] = -matrizCartesianaDoMundo[1]
         
   return matrizCartesianaDoMundo

def janelaParaViewport(matrizCartesianaDoMundo, janela, viewport):
   # Limites da janela (mundo)
   x_min, y_min, x_max, y_max = janela

   aspectRatioJanela = (x_max - x_min) / (y_max - y_min)

   # Limites da viewport (tela)
   u_min, v_min, u_max, v_max = viewport

   aspectRatioViewport = (u_max - u_min) / (v_max - v_min)

   matrizViewport = []
   Sx = (u_max - u_min) / (x_max - x_min)
   Sy = (v_max - v_min) / (y_max - y_min)

   if(aspectRatioJanela > aspectRatioViewport):
      v_maxNovo = ((u_max - u_min) / aspectRatioJanela) + v_min

      matrizViewport = [[Sx, 0, u_min - (Sx * x_min)], [0, -Sy, (Sy * y_max) + (v_max / 2) - (v_maxNovo / 2) + v_min], [0, 0, 1]]
   
   if(aspectRatioJanela < aspectRatioViewport):
      u_maxNovo = aspectRatioJanela * (v_max - v_min) + u_min

      matrizViewport = [[Sx, 0, (-Sx * x_min) + (u_max / 2) - (u_maxNovo / 2) + u_min], [0, -Sy, (Sy * y_max) + v_min], [0, 0, 1]]

   if(aspectRatioJanela == aspectRatioViewport):
      matrizViewport = [[Sx, 0, u_min - (Sx * x_min)], [0, -Sy, (Sy * y_max) + v_min ], [0, 0, 1]]

   matrizViewport = np.array(matrizViewport, dtype=float)

   novaLinha = np.ones(matrizCartesianaDoMundo.shape[1])
   matrizCartesianaDoMundoAtualizada = np.vstack([matrizCartesianaDoMundo, novaLinha])

   return np.dot(matrizViewport, matrizCartesianaDoMundoAtualizada)


def plotarObjetoComArestas(matrizViewport, conexoes):
    plt.figure(figsize=(8, 6))
    
    # Separar as coordenadas X e Y dos vértices
    x_coords = matrizViewport[0]
    y_coords = matrizViewport[1]
    
    # Plotar os vértices
    plt.scatter(x_coords, y_coords, color='blue', label='Vértices')
    
    # Plotar as arestas
    for aresta in conexoes:
        for i in range(len(aresta) - 1):
            x = [x_coords[aresta[i] - 1], x_coords[aresta[i + 1] - 1]]
            y = [y_coords[aresta[i] - 1], y_coords[aresta[i + 1] - 1]]
            plt.plot(x, y, color='black', linewidth=1)

    # Configurações do gráfico
    plt.title("Objeto no Sistema Viewport")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()

def main():
   objeto = Objeto()

   # INPUT DE ENTRADA
   pontoVista = [20, 10, 30]

   # Vetor normal ao plano Z = 0 (FIXO)
   vetorNormal = np.array([0, 0, 1], dtype=float)

   # Ponto sobre o plano Z = 0 (FIXO)
   pontoPlano = np.array([0, 0, 0], dtype=float)

   matrizPerspectiva = calcularMatrizPerspectiva(pontoVista, vetorNormal, pontoPlano)

   matrizLinhaObjeto = np.dot(matrizPerspectiva, objeto.matriz)
   
   matrizCartesianaDoMundo = homogeneasParaCartesianasDoMundo(matrizLinhaObjeto)

   # Transformação janela-viewport
   janela = [-7, -5, 9, 7]  # Limites do mundo (ajuste conforme necessário)
   viewport = [0, 0, 32, 24]  # Resolução do dispositivo 32px x 24px
   matrizJanelaViewport = janelaParaViewport(matrizCartesianaDoMundo, janela, viewport)
   print(matrizJanelaViewport)

   plotarObjetoComArestas(matrizJanelaViewport, objeto.VS)

main()