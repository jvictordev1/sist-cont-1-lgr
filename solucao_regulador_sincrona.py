#  Problema 71 (Nise) - Regulador automatico de maquina sincrona trifasica
#  Requer:  pip install numpy control matplotlib

import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

mu, M, Te = 4, 0.117, 0.5
zeros_sm = [-0.071 + 6.25j, -0.071 - 6.25j]      # z1,2  da maquina sincrona
polos_sm = [-0.047, -0.262 + 5.1j, -0.262 - 5.1j] # p1, p2,3 da maquina

# Montagem de G(s)/K = Gc(s)*Gsm(s)
# Gc(s) = (mu/Te) / (s + 1/Te)   -> ganho mu/Te e um polo em -1/Te = -2
num_gc, den_gc = [mu/Te], [1, 1/Te]
# Gsm(s) = M (s-z1)(s-z2) / (s-p1)(s-p2)(s-p3)
num_sm = M * np.poly(zeros_sm).real
den_sm = np.poly(polos_sm).real
# Multiplicacao das duas -> malha aberta SEM o ganho K
num_ol = np.polymul(num_gc, num_sm)
den_ol = np.polymul(den_gc, den_sm)
G = ctrl.tf(num_ol, den_ol)
print("G(s)/K = Gc*Gsm:\n", G)

# 3) (a) Ganho de estabilidade marginal
# Equacao caracteristica: den_ol(s) + K*num_ol(s) = 0
# Varre-se K e detecta-se onde o polo de maior parte real cruza o eixo jw.
K_marg = w_marg = None
Ks = np.linspace(1e-4, 60, 200000)
prev, Kprev = np.roots(np.polyadd(den_ol, Ks[0]*num_ol)).real.max(), Ks[0]
for K in Ks[1:]:
    mx = np.roots(np.polyadd(den_ol, K*num_ol)).real.max()
    if prev < 0 <= mx:                       # cruzou o eixo imaginario
        lo, hi = Kprev, K
        for _ in range(100):                 # bisseccao para refinar
            mid = 0.5*(lo+hi)
            if np.roots(np.polyadd(den_ol, mid*num_ol)).real.max() < 0: lo = mid
            else: hi = mid
        K_marg = 0.5*(lo+hi)
        rr = np.roots(np.polyadd(den_ol, K_marg*num_ol))
        w_marg = abs(rr[np.argmax(rr.real)].imag)
        break
    prev, Kprev = mx, K
print(f"\n(a) K marginal = {K_marg:.4f} | frequencia w = {w_marg:.4f} rad/s")

# 4) (b)(c) Ponto do LGR para 16% de ultrapassagem
zeta = -np.log(0.16)/np.sqrt(np.pi**2 + np.log(0.16)**2)   # zeta p/ 16% OS
# Reta de zeta (2o quadrante): s(wn) = wn*(-zeta + j*sqrt(1-zeta^2))
# Sobre o LGR vale  K = -den(s)/num(s) REAL e POSITIVO.
def Kc(s): return -np.polyval(den_ol, s)/np.polyval(num_ol, s)
wns = np.linspace(0.001, 25, 400000)
ray = wns*(-zeta + 1j*np.sqrt(1-zeta**2))
im = np.array([Kc(s).imag for s in ray])
sols = []
for i in range(len(wns)-1):
    if im[i]*im[i+1] < 0:                     # imag(K) muda de sinal -> K real
        w0 = wns[i] - im[i]*(wns[i+1]-wns[i])/(im[i+1]-im[i])
        s0 = w0*(-zeta + 1j*np.sqrt(1-zeta**2))
        if Kc(s0).real > 0: sols.append((w0, Kc(s0).real, s0))
w0, K_design, s_design = sorted(sols, key=lambda t: t[1])[0]   # menor K = dominante
print(f"\n(b)(c) zeta = {zeta:.4f} (angulo {np.degrees(np.arccos(zeta)):.2f} graus)")
print(f"     Ponto de projeto no LGR: s = {s_design:.4f}")
print(f"     Ganho de projeto: K = {K_design:.4f}")

# 5) Polos de malha fechada e T(s)
charpoly = np.polyadd(den_ol, K_design*num_ol)
clp = np.roots(charpoly)
print("\n     Polos de malha fechada:")
for pp in sorted(clp, key=lambda x: (round(x.real,4), x.imag)):
    print(f"        {pp:.4f}")
T = ctrl.tf(K_design*num_ol, charpoly)
print("\n     T(s) =\n", T)
print(f"     Ganho DC de T(s) = {ctrl.dcgain(T):.4f}")

# 6) (d) Resposta ao degrau unitario
t = np.linspace(0, 15, 15000)
tt, y = ctrl.step_response(T, T=t)
yf  = y[-1]                                  # valor final (regime permanente)
ypk = y.max(); tpk = tt[np.argmax(y)]        # pico e instante de pico
OS  = (ypk - yf)/yf*100                       # ultrapassagem percentual real
t10 = tt[np.where(y >= 0.10*yf)[0][0]]
t90 = tt[np.where(y >= 0.90*yf)[0][0]]
Tr  = t90 - t10                               # tempo de subida 10-90%
out = np.where(np.abs(y - yf) > 0.02*abs(yf))[0]
Ts  = tt[out[-1]]                             # tempo de acomodacao (+-2%)
print("\n(d) Resposta ao degrau:")
print(f"     valor final      = {yf:.4f}")
print(f"     %OS real         = {OS:.2f} %")
print(f"     Tp               = {tpk:.4f} s")
print(f"     Tr (10-90%)      = {Tr:.4f} s")
print(f"     Ts (+-2%)        = {Ts:.4f} s")

# 7) Graficos
fig, ax = plt.subplots(figsize=(8,7))
ctrl.root_locus(G, ax=ax, grid=False)
w = np.linspace(0, 7, 50)
ax.plot(-zeta*w,  w*np.sqrt(1-zeta**2), '--g', lw=1.3, label=f'reta zeta={zeta:.3f}')
ax.plot(-zeta*w, -w*np.sqrt(1-zeta**2), '--g', lw=1.3)
ax.plot([s_design.real]*2, [s_design.imag, -s_design.imag], 'ks', ms=9,
        label=f'projeto K={K_design:.3f}')
ax.plot([0,0], [w_marg,-w_marg], 'r^', ms=9, label=f'marginal K={K_marg:.3f}')
ax.set_xlim(-3, 0.5); ax.set_ylim(-7, 7); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title('Lugar Geometrico das Raizes'); plt.tight_layout()
plt.savefig('lgr.png', dpi=130)

fig, ax = plt.subplots(figsize=(9,5.5))
ax.plot(tt, y, 'b', lw=1.6)
ax.axhline(yf, ls='--', c='k', lw=.8)
ax.axhline(yf*1.02, ls=':', c='gray'); ax.axhline(yf*0.98, ls=':', c='gray')
ax.plot(tpk, ypk, 'ro'); ax.plot(Ts, y[out[-1]], 'gs')
ax.set_title('Resposta ao degrau unitario'); ax.grid(alpha=.3)
ax.set_xlabel('tempo (s)'); ax.set_ylabel('saida'); plt.tight_layout()
plt.savefig('degrau.png', dpi=130)
print("\nGraficos salvos: lgr.png e degrau.png")
