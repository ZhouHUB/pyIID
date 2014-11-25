__author__ = 'christopher'
from sympy import *

uix, uiy, uiz, ujx, ujy, ujz, sax, say, saz, sbx, sby, sbz = symbols('uix, uiy, uiz, ujx, ujy, ujz, sax, say, saz, sbx, sby, sbz', real = True)
Q = symbols('Q')
r = symbols('r')
sx, sy, sz = symbols('s0 s1 s2')
ux, uy, uz = symbols('u0 u1 u2')
ui = Matrix([uix, uiy, uiz])
uj = Matrix([ujx, ujy, ujz])
sa = Matrix([sax, say, saz])
sb = Matrix([sbx, sby, sbz])
d = sa-sb
rij = d.T.norm()
rijv = d/rij
# pprint(rijv)
# pprint(ui-uj)
u = (ui-uj).T
s = (u*rijv)[0]
s2 = s**2
# pprint(s2)

subs_d = {rij: r, sax-sbx: sx, say-sby: sy, saz-sbz: sz,
                    uix-ujx: ux, uiy-ujy: uy, uiz-ujz: uz}
sub_fq = exp(-.5*s2*Q**2)*sin(Q*rij)/rij
pprint(sub_fq.subs(subs_d))

dir_fq = diff(sub_fq, sax)

dfq_subs = dir_fq.subs(subs_d).simplify()
print(latex(dfq_subs))
pprint(dfq_subs)

u_dir = diff(sub_fq, uix)
du_subs = u_dir.subs(subs_d).simplify()
print(latex(du_subs))
pprint(du_subs)