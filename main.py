import numpy as np
import matplotlib.pyplot as plt

class ObjetoPiramide(): 
   def __init__(self):
      self.NV = 5 # Número de vértices
      self.matriz = np.array([[1, 7, 7, 1, 4], [1, 1, 1, 1, 7], [1, 1, 7, 7, 4], [1, 1, 1, 1, 1]], dtype=float) # Matriz do objeto
      self.NS = 5 # Número de superficies
      self.NVPS = np.array([4, 3, 3, 3, 3], dtype=float) # Número de vértices por superfície
      self.VS = [[4, 3, 5, 4], [3, 2, 5, 3], [5, 2, 1, 5], [4, 5, 1, 4], [4, 1, 2, 3, 4]] # Vértices de cada superfície

class ObjetoCubo(): 
    def __init__(self):
        self.NV = 8  # Número de vértices
        # Matriz com as coordenadas dos vértices de um cubo
        self.matriz = np.array([
            [1, 1, 1, 1, -1, -1, -1, -1],  # X
            [1, 1, -1, -1, 1, 1, -1, -1],  # Y
            [1, -1, -1, 1, 1, -1, -1, 1],   # Z
            [1, 1, 1, 1, 1, 1, 1, 1]        # Homogeneização
        ], dtype=float)
        
        self.NS = 6  # Número de superfícies (faces do cubo)
        self.NVPS = np.array([4, 4, 4, 4, 4, 4], dtype=float)  # Número de vértices por superfície
        self.VS = [
            [1, 2, 3, 4],  # Face frontal
            [5, 6, 7, 8],  # Face traseira
            [1, 2, 6, 5],  # Face inferior
            [2, 3, 7, 6],  # Face direita
            [3, 4, 8, 7],  # Face superior
            [4, 1, 5, 8]   # Face esquerda
        ]  # Vértices de cada superfície (faces do cubo)

def calcularVetorNormal(pontoA, pontoB, pontoC):
   vetorAB = np.array(pontoB, dtype=float) - np.array(pontoA, dtype=float)
   vetorCB = np.array(pontoB, dtype=float) - np.array(pontoC, dtype=float)

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
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.show()

def main():
   print("Escolha o objeto para renderizar:")
   print("1 - Cubo")
   print("2 - Pirâmide")
   escolha = int(input("Digite o número da opção desejada (1 ou 2): "))
   
   if escolha == 1:
      objeto = ObjetoCubo()
   elif escolha == 2:
      objeto = ObjetoPiramide()
   else:
      print("Opção inválida!")
      return

   # INPUT DE ENTRADA
   print("Digite as coordenadas do ponto de vista:")
   pontoVista = list(map(float, input("Digite x, y, z (separados por espaço): ").split()))

   # print("Digite as coordenadas dos pontos p1, p2, p3 para definição de um plano:")
   # p1 = list(map(float, input("Digite x, y, z para p1 (separados por espaço): ").split()))
   # p2 = list(map(float, input("Digite x, y, z para p2 (separados por espaço): ").split()))
   # p3 = list(map(float, input("Digite x, y, z para p3 (separados por espaço): ").split()))
   p1 = [1, 0, 0]
   p2 = [0, 0, 0]
   p3 = [0, 1, 0]

   vetorNormal = calcularVetorNormal(p1, p2, p3)
   pontoPlano = p2

   matrizPerspectiva = calcularMatrizPerspectiva(pontoVista, vetorNormal, pontoPlano)

   matrizLinhaObjeto = np.dot(matrizPerspectiva, objeto.matriz)
   
   matrizCartesianaDoMundo = homogeneasParaCartesianasDoMundo(matrizLinhaObjeto)

   # Transformação janela-viewport
   janela = [-7, -5, 9, 7]
   viewport = [0, 0, 32, 24]
   matrizJanelaViewport = janelaParaViewport(matrizCartesianaDoMundo, janela, viewport)

   plotarObjetoComArestas(matrizJanelaViewport, objeto.VS)

main()