# spectral-solutions-to-pde

Målet med denna uppgift är att använda spektrala metoder för att lösa PDEr. Vi kommer kolla på transport-diffusionsproblemet:

$$
 \partial_t u - \lambda \Delta u + \overline{b}\cdot \nabla u = f
$$

På en periodisk domän $[0,1]^2$. Vi kommer använda en implicit-explicit bakåt Euler metod som följande:

$$
  u_{n+1} - \Delta t \lambda \Delta u_{n+1} = u_n + \Delta t(f_n - \overline{b}_n\cdot \nabla u_n)
$$

 Kurs SF1693, vi undersöker en spektral metod för att lösa transport + diffusion.
 Välj en funktion och implementera. Pusha sedan koden till github efter att ni har testat att den funkar!
 Kör ``main`` för att testa er kod. Den kommer att göra ett enkelt test för att verifiera att den funkar som väntat.
 OBS: Ni som implementerar solve() kommer behöva vänta tills alla andra är klara. Ni kan fråga mig om koden ser bra ut.
 
