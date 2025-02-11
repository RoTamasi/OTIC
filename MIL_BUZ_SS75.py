import numpy as np
import matplotlib.pyplot as plt
import math, time
from scipy.stats import lognorm
from matplotlib.path import Path
from multiprocessing import Pool, freeze_support
import os

# ================================
# Parâmetros de execução
# ================================
num_cores = (os.cpu_count() - 1) # ajuste o número de núcleos a serem usados

# ================================
# Carrega dados e define variáveis globais
# ================================
ROTAS = np.loadtxt('dados/ROTAS_SS75.txt')
(num_rotas, ncols) = ROTAS.shape
Rota_vert = int((ncols - 1) / 3)  # número de segmentos por rota
P_rota = np.ones(num_rotas) / num_rotas
RaioMax = np.max(ROTAS)

Min_x, Min_y = -RaioMax, -RaioMax
Max_x, Max_y = RaioMax, RaioMax
Delta = RaioMax / 250.0
eixo_x = np.arange(Min_x, Max_x + Delta, Delta)
eixo_y = np.arange(Min_y, Max_y + Delta, Delta)
Malha = np.zeros((len(eixo_x), len(eixo_y)))

Boca = 40
Comprimento = 299
Diagonal = math.sqrt(Boca ** 2 + Comprimento ** 2) / 2.0
Angulo = math.degrees(math.atan(Boca / Comprimento))

Tempo_delta = 1
Tempo_deriva = np.arange(0, 181, Tempo_delta)
Dist_recuperacao = lognorm.pdf(Tempo_deriva, s=1.52548, scale=np.exp(2.71863))
Prec_intervalo = np.zeros_like(Tempo_deriva, dtype=float)
for i in range(len(Tempo_deriva) - 1):
    Prec_intervalo[i] = (Dist_recuperacao[i] + Dist_recuperacao[i + 1]) / 2.0 * (Tempo_deriva[i + 1] - Tempo_deriva[i])
Prec_intervalo[-1] = 1 - np.sum(Prec_intervalo)
Prec_intervalo /= np.sum(Prec_intervalo)
Pnrec_acum = np.zeros_like(Prec_intervalo)
for i in range(len(Prec_intervalo)):
    Pnrec_acum[i] = np.sum(Prec_intervalo[i:])

P_deriva = 0.033  # probabilidade de entrar em deriva


def process_segment(task):
    nr, k = task

    if k == 1:
        Coord_x_1 = 0.0
        Coord_y_1 = 0.0
        angle_val = ROTAS[nr, 0]
        Coord_x_1_a = Coord_x_1 + Diagonal * math.sin(math.radians(-Angulo + angle_val))
        Coord_y_1_a = Coord_y_1 + Diagonal * math.cos(math.radians(-Angulo + angle_val))
        Coord_x_1_b = Coord_x_1 + Diagonal * math.sin(math.radians(Angulo + angle_val))
        Coord_y_1_b = Coord_y_1 + Diagonal * math.cos(math.radians(Angulo + angle_val))
        Coord_x_1_c = Coord_x_1 + Diagonal * math.sin(math.radians(-Angulo + angle_val + 180))
        Coord_y_1_c = Coord_y_1 + Diagonal * math.cos(math.radians(-Angulo + angle_val + 180))
        Coord_x_1_d = Coord_x_1 + Diagonal * math.sin(math.radians(Angulo + angle_val + 180))
        Coord_y_1_d = Coord_y_1 + Diagonal * math.cos(math.radians(Angulo + angle_val + 180))
    else:
        idx1_dist = k * 3 - 4
        idx1_ang = k * 3 - 5
        Coord_x_1 = ROTAS[nr, idx1_dist] * math.sin(math.radians(ROTAS[nr, idx1_ang]))
        Coord_y_1 = ROTAS[nr, idx1_dist] * math.cos(math.radians(ROTAS[nr, idx1_ang]))
        idx1_ang_next = k * 3 - 2
        Coord_x_1_a = Coord_x_1 + Diagonal * math.sin(math.radians(-Angulo + ROTAS[nr, idx1_ang_next]))
        Coord_y_1_a = Coord_y_1 + Diagonal * math.cos(math.radians(-Angulo + ROTAS[nr, idx1_ang_next]))
        Coord_x_1_b = Coord_x_1 + Diagonal * math.sin(math.radians(Angulo + ROTAS[nr, idx1_ang_next]))
        Coord_y_1_b = Coord_y_1 + Diagonal * math.cos(math.radians(Angulo + ROTAS[nr, idx1_ang_next]))
        Coord_x_1_c = Coord_x_1 + Diagonal * math.sin(math.radians(-Angulo + ROTAS[nr, idx1_ang_next] + 180))
        Coord_y_1_c = Coord_y_1 + Diagonal * math.cos(math.radians(-Angulo + ROTAS[nr, idx1_ang_next] + 180))
        Coord_x_1_d = Coord_x_1 + Diagonal * math.sin(math.radians(Angulo + ROTAS[nr, idx1_ang_next] + 180))
        Coord_y_1_d = Coord_y_1 + Diagonal * math.cos(math.radians(Angulo + ROTAS[nr, idx1_ang_next] + 180))

    if k == 1:
        idx2_dist = 2
        idx2_ang = 1
        idx2_ang_next = 3
    else:
        idx2_dist = k * 3 - 1
        idx2_ang = k * 3 - 2
        idx2_ang_next = k * 3

    Coord_x_2 = ROTAS[nr, idx2_dist] * math.sin(math.radians(ROTAS[nr, idx2_ang]))
    Coord_y_2 = ROTAS[nr, idx2_dist] * math.cos(math.radians(ROTAS[nr, idx2_ang]))
    Coord_x_2_a = Coord_x_2 + Diagonal * math.sin(math.radians(-Angulo + ROTAS[nr, idx2_ang_next]))
    Coord_y_2_a = Coord_y_2 + Diagonal * math.cos(math.radians(-Angulo + ROTAS[nr, idx2_ang_next]))
    Coord_x_2_b = Coord_x_2 + Diagonal * math.sin(math.radians(Angulo + ROTAS[nr, idx2_ang_next]))
    Coord_y_2_b = Coord_y_2 + Diagonal * math.cos(math.radians(Angulo + ROTAS[nr, idx2_ang_next]))
    Coord_x_2_c = Coord_x_2 + Diagonal * math.sin(math.radians(-Angulo + ROTAS[nr, idx2_ang_next] + 180))
    Coord_y_2_c = Coord_y_2 + Diagonal * math.cos(math.radians(-Angulo + ROTAS[nr, idx2_ang_next] + 180))
    Coord_x_2_d = Coord_x_2 + Diagonal * math.sin(math.radians(Angulo + ROTAS[nr, idx2_ang_next] + 180))
    Coord_y_2_d = Coord_y_2 + Diagonal * math.cos(math.radians(Angulo + ROTAS[nr, idx2_ang_next] + 180))

    denom = (Coord_x_2 - Coord_x_1)
    if abs(denom) < 1e-6:
        denom = 1e-6
    Reta_dir_a = (Coord_y_2 - Coord_y_1) / denom

    Coord_orig = np.array([
        [Coord_x_1_a, Coord_x_1_b, Coord_x_1_c, Coord_x_1_d,
         Coord_x_2_a, Coord_x_2_b, Coord_x_2_c, Coord_x_2_d],
        [Coord_y_1_a, Coord_y_1_b, Coord_y_1_c, Coord_y_1_d,
         Coord_y_2_a, Coord_y_2_b, Coord_y_2_c, Coord_y_2_d]
    ])
    Coord_rela = Coord_orig - np.array([[Coord_x_1], [Coord_y_1]])
    for i in range(8):
        a_val = math.hypot(Coord_rela[0, i], Coord_rela[1, i])
        ang = math.atan2(Coord_rela[1, i], Coord_rela[0, i]) - math.atan(Reta_dir_a)
        Coord_rela[0, i] = a_val * math.cos(ang)
        Coord_rela[1, i] = a_val * math.sin(ang)

    Ordem_inicial_x = Coord_rela[0, :4]
    Ordem_inicial_y = Coord_rela[1, :4]
    Ordem_final_x = Coord_rela[0, 4:]
    Ordem_final_y = Coord_rela[1, 4:]
    if np.max(Ordem_final_x) > np.max(Ordem_inicial_x):
        Ordem_ref = 0
        Ordem_0 = Ordem_inicial_y
        Ordem_2 = Ordem_final_y
    else:
        Ordem_ref = 1
        Ordem_0 = -Ordem_inicial_y
        Ordem_2 = -Ordem_final_y
        Coord_rela = -Coord_rela

    Ordem_1 = int(np.argmin(Ordem_0))
    Setor = np.zeros((2, 6))
    if Ordem_1 == 0:
        Setor[:, 0] = Coord_rela[:, 0]
        Setor[:, 1] = Coord_rela[:, 1] if k == 1 else Coord_rela[:, 3]
        Setor[:, 2] = Coord_rela[:, 2]
    elif Ordem_1 == 1:
        Setor[:, 0] = Coord_rela[:, 1]
        Setor[:, 1] = Coord_rela[:, 2] if k == 1 else Coord_rela[:, 0]
        Setor[:, 2] = Coord_rela[:, 3]
    elif Ordem_1 == 2:
        Setor[:, 0] = Coord_rela[:, 2]
        Setor[:, 1] = Coord_rela[:, 3] if k == 1 else Coord_rela[:, 1]
        Setor[:, 2] = Coord_rela[:, 0]
    elif Ordem_1 == 3:
        Setor[:, 0] = Coord_rela[:, 3]
        Setor[:, 1] = Coord_rela[:, 0] if k == 1 else Coord_rela[:, 2]
        Setor[:, 2] = Coord_rela[:, 1]

    i_max = int(np.argmax(Ordem_2))
    Ordem_3 = i_max + 4
    if Ordem_3 == 4:
        Setor[:, 3] = Coord_rela[:, 4]
        Setor[:, 4] = Coord_rela[:, 5]
        Setor[:, 5] = Coord_rela[:, 6]
    elif Ordem_3 == 5:
        Setor[:, 3] = Coord_rela[:, 5]
        Setor[:, 4] = Coord_rela[:, 6]
        Setor[:, 5] = Coord_rela[:, 7]
    elif Ordem_3 == 6:
        Setor[:, 3] = Coord_rela[:, 6]
        Setor[:, 4] = Coord_rela[:, 7]
        Setor[:, 5] = Coord_rela[:, 4]
    elif Ordem_3 == 7:
        Setor[:, 3] = Coord_rela[:, 7]
        Setor[:, 4] = Coord_rela[:, 4]
        Setor[:, 5] = Coord_rela[:, 5]

    if Ordem_ref == 1:
        Setor = -Setor

    for i in range(6):
        a_val = math.hypot(Setor[0, i], Setor[1, i])
        ang = math.atan2(Setor[1, i], Setor[0, i]) + math.atan(Reta_dir_a)
        Setor[0, i] = a_val * math.cos(ang)
        Setor[1, i] = a_val * math.sin(ang)
    Setor[0, :] += Coord_x_1
    Setor[1, :] += Coord_y_1

    poly_points = [(x, y) for x, y in zip(Setor[0, :], Setor[1, :])]
    poly_points.append(poly_points[0])

    Max_local_x = np.max(Setor[0, :])
    Min_local_x = np.min(Setor[0, :])
    Max_local_y = np.max(Setor[1, :])
    Min_local_y = np.min(Setor[1, :])

    imin = int(np.searchsorted(eixo_x, Min_local_x, side='left'))
    imax = int(np.searchsorted(eixo_x, Max_local_x, side='right')) - 1
    jmin = int(np.searchsorted(eixo_y, Min_local_y, side='left'))
    jmax = int(np.searchsorted(eixo_y, Max_local_y, side='right')) - 1
    imin = max(0, imin)
    imax = min(len(eixo_x) - 1, imax)
    jmin = max(0, jmin)
    jmax = min(len(eixo_y) - 1, jmax)

    X_sub = eixo_x[imin:imax + 1]
    Y_sub = eixo_y[jmin:jmax + 1]
    Xg, Yg = np.meshgrid(X_sub, Y_sub, indexing='ij')
    pts = np.column_stack((Xg.ravel(), Yg.ravel()))
    poly_path = Path(poly_points)
    mask = poly_path.contains_points(pts)
    mask = mask.reshape(Xg.shape)

    contrib_value = Pnrec_acum[k - 1] * P_rota[nr]
    contrib = np.zeros_like(mask, dtype=float)
    contrib[mask] = contrib_value

    return (imin, imax, jmin, jmax, contrib)


if __name__ == '__main__':
    freeze_support()
    start_time = time.time()

    tasks = []
    for nr in range(num_rotas):
        for k in range(1, Rota_vert + 1):
            tasks.append((nr, k))

    with Pool(processes=num_cores) as pool:
        results = pool.map(process_segment, tasks)

    for imin, imax, jmin, jmax, contrib in results:
        Malha[imin:imax + 1, jmin:jmax + 1] += contrib

    Malha *= P_deriva

    plt.figure()
    plt.contourf(eixo_x, eixo_y, Malha.T, levels=20)
    plt.colorbar()
    plt.title("Contorno do MIL")

    np.savetxt("Malha_SS75.txt", Malha, fmt="%.6e")

    MIL_1 = Malha.copy()
    plt.figure()
    plt.title("MIL (pontos coloridos)")
    for i in range(len(eixo_x)):
        for j in range(len(eixo_y)):
            val = Malha[i, j]
            if val < 1e-5:
                plt.plot(eixo_x[i], eixo_y[j], 'b.', markersize=1)
                MIL_1[i, j] = 0
            elif val < 1e-4:
                plt.plot(eixo_x[i], eixo_y[j], 'c.', markersize=1)
                MIL_1[i, j] = 1
            elif val < 1e-3:
                plt.plot(eixo_x[i], eixo_y[j], 'g.', markersize=1)
                MIL_1[i, j] = 2
            elif val < 1e-2:
                plt.plot(eixo_x[i], eixo_y[j], 'y.', markersize=1)
                MIL_1[i, j] = 4
            elif val < 1e-1:
                plt.plot(eixo_x[i], eixo_y[j], 'm.', markersize=1)
                MIL_1[i, j] = 6
            else:
                plt.plot(eixo_x[i], eixo_y[j], 'r.', markersize=1)
                MIL_1[i, j] = 7

    plt.figure()
    plt.contourf(eixo_x, eixo_y, MIL_1.T, levels=6)
    plt.title("Contorno MIL_1")

    MIL = Malha.copy()
    plt.figure()
    plt.contour(eixo_x, eixo_y, MIL.T, levels=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], colors='r')
    MalhaMIL = MIL.copy()
    for _ in range(8):
        MIL_new = MalhaMIL.copy()
        for i in range(1, len(eixo_x) - 1):
            for j in range(1, len(eixo_y) - 1):
                MIL_new[i, j] = np.mean(MalhaMIL[i - 1:i + 2, j - 1:j + 2])
        MalhaMIL = MIL_new.copy()
        MIL = MIL_new.copy()
    plt.contour(eixo_x, eixo_y, MIL.T, levels=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], colors='k')
    plt.title("MIL Suavizado")

    print("Tempo de execução: {:.2f} segundos".format(time.time() - start_time))
    plt.show()
