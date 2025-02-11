import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

# --- Banco de rotas ---
# Carrega os dados (mesmo que neste trecho o arquivo não seja usado para cálculos posteriores)
ROTAS = np.loadtxt('dados/ROTAS_SS75.txt')

# --- Definição da área a ser mapeada ---
RaioMax = np.max(ROTAS)  # equivale a max(max(ROTAS)) no MATLAB
Min_x = -RaioMax
Min_y = -RaioMax
Max_x = RaioMax
Max_y = RaioMax
Delta_x = RaioMax / 250.0  # verificar sensibilidade
Delta_y = RaioMax / 250.0
eixo_x = np.arange(Min_x, Max_x + Delta_x, Delta_x)
eixo_y = np.arange(Min_y, Max_y + Delta_y, Delta_y)
# Inicializa MIL; observe que no MATLAB foi usado zeros(length(eixo_x), length(eixo_x)),
# porém o mais usual é usar (len(eixo_x), len(eixo_y))
MIL = np.zeros((len(eixo_x), len(eixo_y)))

# --- Gráficos iniciais ---
plt.figure()
# (Com "hold on" o Matplotlib mantém os plots; não é necessário comando explícito)

# Carrega o MIL salvo em arquivo
MIL = np.loadtxt('Malha_SS75.txt')
# %% O trecho abaixo (transposição) foi comentado no MATLAB; ajuste se necessário.
# MIL = MIL.T

# Plota contornos nos níveis especificados
C = plt.contour(eixo_x, eixo_y, MIL.T, levels=[0.01], colors='k')
C = plt.contour(eixo_x, eixo_y, MIL.T, levels=[0.001], colors='k')
plt.clabel(C, fontsize=8)
C = plt.contour(eixo_x, eixo_y, MIL.T, levels=[0.0001], colors='k')
plt.clabel(C, fontsize=8)

# --- Suavização do MIL ---

# Inicializa a cópia que será utilizada para suavização
MalhaMIL = MIL.copy()

# Série: 0.00001 (6 passadas)
for _ in range(6):
    MIL_new = MIL.copy()
    # Loop interno: em Python os índices vão de 1 até len(eixo)-2 (pois MATLAB: 2:length-1)
    for i in range(1, len(eixo_x)-1):
        for j in range(1, len(eixo_y)-1):
            MIL_new[i, j] = (
                MalhaMIL[i-1, j-1] + MalhaMIL[i-1, j] + MalhaMIL[i-1, j+1] +
                MalhaMIL[i, j-1]   + MalhaMIL[i, j]   + MalhaMIL[i, j+1] +
                MalhaMIL[i+1, j-1] + MalhaMIL[i+1, j] + MalhaMIL[i+1, j+1]
            ) / 9.0
    MIL = MIL_new.copy()
    MalhaMIL = MIL_new.copy()

C = plt.contour(eixo_x, eixo_y, MIL.T, levels=[0.00001], colors='k')
plt.clabel(C, fontsize=8)

# Série: 0.000001 (12 passadas)
for _ in range(12):
    MIL_new = MIL.copy()
    for i in range(1, len(eixo_x)-1):
        for j in range(1, len(eixo_y)-1):
            MIL_new[i, j] = (
                MalhaMIL[i-1, j-1] + MalhaMIL[i-1, j] + MalhaMIL[i-1, j+1] +
                MalhaMIL[i, j-1]   + MalhaMIL[i, j]   + MalhaMIL[i, j+1] +
                MalhaMIL[i+1, j-1] + MalhaMIL[i+1, j] + MalhaMIL[i+1, j+1]
            ) / 9.0
    MIL = MIL_new.copy()
    MalhaMIL = MIL_new.copy()

C = plt.contour(eixo_x, eixo_y, MIL.T, levels=[0.000001], colors='k')
plt.clabel(C, fontsize=8)

# Série: 0.0000001 (20 passadas)
for _ in range(20):
    MIL_new = MIL.copy()
    for i in range(1, len(eixo_x)-1):
        for j in range(1, len(eixo_y)-1):
            MIL_new[i, j] = (
                MalhaMIL[i-1, j-1] + MalhaMIL[i-1, j] + MalhaMIL[i-1, j+1] +
                MalhaMIL[i, j-1]   + MalhaMIL[i, j]   + MalhaMIL[i, j+1] +
                MalhaMIL[i+1, j-1] + MalhaMIL[i+1, j] + MalhaMIL[i+1, j+1]
            ) / 9.0
    MIL = MIL_new.copy()
    MalhaMIL = MIL_new.copy()

C = plt.contour(eixo_x, eixo_y, MIL.T, levels=[0.0000001], colors='k')
plt.clabel(C, fontsize=8)

# Série: 0.00000001 (24 passadas)
for _ in range(24):
    MIL_new = MIL.copy()
    for i in range(1, len(eixo_x)-1):
        for j in range(1, len(eixo_y)-1):
            MIL_new[i, j] = (
                MalhaMIL[i-1, j-1] + MalhaMIL[i-1, j] + MalhaMIL[i-1, j+1] +
                MalhaMIL[i, j-1]   + MalhaMIL[i, j]   + MalhaMIL[i, j+1] +
                MalhaMIL[i+1, j-1] + MalhaMIL[i+1, j] + MalhaMIL[i+1, j+1]
            ) / 9.0
    MIL = MIL_new.copy()
    MalhaMIL = MIL_new.copy()

C = plt.contour(eixo_x, eixo_y, MIL.T, levels=[0.00000001], colors='k')
plt.clabel(C, fontsize=8)

# Exibe o tempo total de execução
print("Tempo de execução: {:.2f} segundos".format(time.time() - start_time))
plt.show()
