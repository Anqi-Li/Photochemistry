#%%
from operator import mul, add
from functools import reduce
import sympy as sym
from itertools import chain  # Py 2.7 does not support func(*args1, *args2)
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def prod(seq):
    return reduce(mul, seq) if seq else 1

def mk_exprs_symbs(rxns, names):
    # create symbols for reactants
    symbs = sym.symbols(names, real=True, nonnegative=True, positive=True)
    # map between reactant symbols and keys in r_stoich, net_stoich
    c = dict(zip(names, symbs))
    f = {n: 0 for n in names}
    k = []
    for coeff, r_stoich, net_stoich in rxns:
        k.append(sym.symbols(coeff, real=True, nonnegative=True, positive=True))
        r = k[-1]*prod([c[rk]**p for rk, p in r_stoich.items()])  # EXERCISE: c[rk]**p
        for net_key, net_mult in net_stoich.items():
            f[net_key] += net_mult*r  # EXERCISE: net_mult*r
    return [f[n] for n in names], symbs, tuple(k)

# def mk_exprs_symbs(rxns, names):
#     # create symbols for reactants (concentrations)
#     concs = sym.symbols(names, real=True, nonnegative=True, positive=True)
#     # map between reactant symbols and keys in r_stoich, net_stoich
#     c_dict = dict(zip(names, concs))
#     f = {n: 0 for n in names}
#     for coeff, r_stoich, net_stoich in rxns:
#         k0 = sym.symbols(coeff, real=True, nonnegative=True, positive=True)
#         r = k0*prod([c_dict[rk]**p for rk, p in r_stoich.items()])
#         for nk, nm in net_stoich.items():
#             f[nk] += nm*r
#     return [f[n] for n in names], concs

def loss_terms(rxns, names):
    # create symbols for reactants
    symbs = sym.symbols(names, real=True, nonnegative=True, positive=True)
    # map between reactant symbols and keys in r_stoich, net_stoich
    c = dict(zip(names, symbs))
    f = {n: 0 for n in names}
    k = []
    for coeff, r_stoich, net_stoich in rxns:
        k.append(sym.symbols(coeff, real=True, nonnegative=True, positive=True))
        r = k[-1]*prod([c[rk]**p for rk, p in r_stoich.items()])  # EXERCISE: c[rk]**p
        for net_key, net_mult in net_stoich.items():
            if net_mult < 0: #loss terms
                f[net_key] += -net_mult*r  # EXERCISE: net_mult*r
    return [f[n] for n in names]
    
def mk_rsys(ODEcls, reactions, names, params=(), **kwargs):
    f, symbs, _ = mk_exprs_symbs(reactions, names)
    return ODEcls(f, symbs, params=map(sym.S, params), **kwargs)

# %%
class ODEsys(object):

    default_integrator = 'odeint'

    def __init__(self, f, state, t=None, params=(), lambdify=None):
        assert len(f) == len(state), 'f is dy/dt'
        self.f = tuple(f) #differential equations for each state
        self.state = tuple(state) #states
        self.t = t #time
        self.p = tuple(params) #parameters
        self.j = sym.Matrix(self.nstate, 1, f).jacobian(state) #jacobian of f
        self.lambdify = lambdify or sym.lambdify 
        self.setup() #setup lambdified functions 

    @property
    def nstate(self):
        return len(self.state)

    def setup(self):
        self.lambdified_f = self.lambdify(self.state + self.p, self.f)
        self.lambdified_j = self.lambdify(self.state + self.p, self.j)

    def f_eval(self, state, t, *params):
        return self.lambdified_f(*chain(state, params))

    def j_eval(self, state, t, *params):
        return self.lambdified_j(*chain(state, params))

    def integrate(self, *args, **kwargs):
        integrator = kwargs.pop('integrator', self.default_integrator)
        return getattr(self, 'integrate_%s' % integrator)(*args, **kwargs)

    def integrate_odeint(self, tout, state0, params=(), rtol=1e-8, atol=1e-8, **kwargs):
        return odeint(self.f_eval, state0, tout, args=tuple(params), full_output=True,
                        Dfun=self.j_eval, rtol=rtol, atol=atol, **kwargs)

    def print_info(self, info):
        if info is None:
            return
        nrhs = info.get('num_rhs')
        if not nrhs:
            nrhs = info['nfe'][-1]
        njac = info.get('num_dls_jac_evals')
        if not njac:
            njac = info['nje'][-1]
        print("The rhs was evaluated %d times and the jacobian %d times" % (nrhs, njac))
    

# %%
class MOLsys(ODEsys):
    """ System of ODEs based on method of lines on the interval x = [0, x_end],
    where x represents a spatial dimension  """

    def __init__(self, *args, **kwargs):
        self.x = kwargs.pop('x') #grid
        self.n_x_bins = len(self.x)
        self.dx = np.gradient(self.x)
        # self.x_end = kwargs.pop('x_end') #end of the spatial grid
        # self.n_x_bins = kwargs.pop('n_x_bins') #number of grids
        # self.dx = self.x_end / self.n_x_bins #distance of the spatial grid
        self.D = kwargs.pop('D') #diffusion coefficient for each state
        super(MOLsys, self).__init__(*args, **kwargs)

    def f_eval(self, state_flat, t, *params):
        f_out = np.empty(self.nstate*self.n_x_bins)
        for i in range(self.n_x_bins):
            slc = slice(i*self.nstate, (i+1)*self.nstate)
            f_out[slc] = self.second_derivatives_spatial(i, state_flat, f_out[slc])
            f_out[slc] *= self.D
            f_out[slc] += self.lambdified_f(*chain(state_flat[slc], tuple(list(params)[slc])))
        return f_out

    def central_reference_bin(self, i):
        return np.clip(i, 1, self.nstate - 2)

    def j_eval(self, state_flat, t, *params):
        j_out = np.zeros((self.nstate*self.n_x_bins, self.nstate*self.n_x_bins))  # dense matrix
        for i in range(self.n_x_bins):
            slc = slice(i*self.nstate, (i+1)*self.nstate)
            j_out[slc, slc] = self.lambdified_j(*chain(state_flat[slc], tuple(list(params)[slc])))
            k = self.central_reference_bin(i)
            for j in range(self.nstate):
                j_out[i*self.nstate + j, (k-1)*self.nstate + j] +=    self.D[j]/self.dx[i]**2
                j_out[i*self.nstate + j, (k  )*self.nstate + j] += -2*self.D[j]/self.dx[i]**2
                j_out[i*self.nstate + j, (k+1)*self.nstate + j] +=    self.D[j]/self.dx[i]**2
        return j_out

    def second_derivatives_spatial(self, i, state_flat, out):
        k = self.central_reference_bin(i)
        out = np.empty(self.nstate)
        for j in range(self.nstate):
            left = state_flat[(k-1)*self.nstate + j]
            cent = state_flat[(k  )*self.nstate + j]
            rght = state_flat[(k+1)*self.nstate + j]
            out[j] = (left - 2*cent + rght)/self.dx[i]**2
        return out

    def integrate(self, tout, state0, params=(), **kwargs):
        state0 = np.array(np.vstack(state0).T.flat)
        params = tuple(np.array(np.vstack(params).T.flat))
        stateout, info = super(MOLsys, self).integrate(tout, state0, params, **kwargs)
        stateout = stateout.reshape((tout.size, self.n_x_bins, self.nstate)).transpose((0, 2, 1))
        return stateout, info

def family_equilibrium(families, reactions, species_lst):
    '''
    arguments:
    families: dictionary. Keys are familyname (str), values are species (list)
    reactions: a list of tuples that consists of rate constant (str), r_stoich (dict) and net_stoich(dict)
    species_lst: a lst of species
    returns:
    y_solve_family: dictionary. Keys are species symbols, values are the equation for each species  
    '''
    r_stoich_lst = [r[1] for r in reactions]
    net_stoich_lst = [r[2] for r in reactions]
    species_names = [s.name for s in species_lst]
    family_net_stoich_lst = [{familyname: sum([ns.get(s.name) for s in family if s.name in ns])
                                for familyname, family in families.items()}
                                for ns in net_stoich_lst]
    family_r_stoich_lst = [{familyname: sum([rs.get(s.name) for s in family if s.name in rs])
                                for familyname, family in families.items()}
                                for rs in r_stoich_lst]
    find_family_reactions = {familyname: [(rs[familyname]!=0) and (ns[familyname]==0)
                for rs, ns in zip(family_r_stoich_lst, family_net_stoich_lst)]
                for familyname in families.keys()}
    eqns_family = []
    for familyname in families.keys():
        family_reactions = [r for i,r in enumerate(reactions) if find_family_reactions[familyname][i]]
        # set up ode (ydot) equations of each species within the family
        ydot_within_family, y, _ = mk_exprs_symbs(family_reactions, species_names) 
        ydot_within_family = [ydot_within_family[species_lst.index(s)] for s in families[familyname]]
        eqns_family += ydot_within_family
        eqns_family += [familyname - reduce(add, [y[species_lst.index(s)] for s in families[familyname]])]
    solutions = sym.solve_poly_system(eqns_family, [y[species_lst.index(s)] for s in reduce(add, families.values())])
    while len(solutions)>1:
        print('two solutions exists')
        # eliminate solutions that are definately negative
        tf = [any([s.is_negative==True for s in sol]) for sol in solutions]
        if any(tf):
            print('eliminate the negative solution')
            _ = solutions.pop(tf.index(True))
        else:
            print('eliminate the last solution')
            _ = solutions.pop(-1)
    # y_solve_family = {s: sol for s,sol in 
    #     zip([y[species_lst.index(s)] for s in reduce(add, families.values())], solutions[0])}
    y_solve_family = dict(zip([y[species_lst.index(s)] for s in reduce(add, families.values())], 
                            solutions[0]))
    return y_solve_family
    
#%% for testing polynomial equilibrium functions
# family_net_stoich_lst = [{familyname: sum([ns.get(s.name) for s in family if s.name in ns])
#                             for familyname, family in families.items()}
#                             for ns in net_stoich_lst]
# family_r_stoich_lst = [{familyname: sum([rs.get(s.name) for s in family if s.name in rs])
#                             for familyname, family in families.items()}
#                             for rs in r_stoich_lst]
# find_family_reactions = {familyname: [(rs[familyname]!=0) and (ns[familyname]==0)
#             for rs, ns in zip(family_r_stoich_lst, family_net_stoich_lst)]
#             for familyname in families.keys()}
# eqns_family = []
# for familyname in families.keys():
#     family_reactions = [r for i,r in enumerate(reactions) if find_family_reactions[familyname][i]]
#     # set up ode (ydot) equations of each species within the family
#     ydot_within_family, y, _ = mk_exprs_symbs(family_reactions, species_names) 
#     ydot_within_family = [ydot_within_family[species_lst.index(s)] for s in families[familyname]]
#     eqns_family += ydot_within_family
#     eqns_family += [familyname - reduce(add, [y[species_lst.index(s)] for s in families[familyname]])]
#     # print(sym.Matrix([sym.Eq(sym.Function(fam.name)(t).diff(t), eq) for fam, eq in zip(reduce(add, families.values()), eqns_family) ] + [sym.Eq(familyname, reduce(add, [sym.symbols(s.name) for s in families[familyname]])) for familyname in families.keys()]))

##  only for linear case in one family (AO)  
# sol_AO, = sym.solve_poly_system(eqns_family[len(families[AH])+1:], *[y[species_lst.index(s)] for s in families[AO]])
# sol_AH, = sym.solve_poly_system(eqns_family[:len(families[AH])+1], *[y[species_lst.index(s)] for s in families[AH]])
# # sol_AH = [eq.subs({s: e for s,e in zip([y[species_lst.index(i)] for i in families[AO]], sol_AO[0])}) for eq in sol_AH[0]]
# sol_AH = [sym.lambdify([y[species_lst.index(s)] for s in families[AO]], eq)(*sol_AO) for eq in sol_AH]
# solutions = [list(sol_AO) + sol_AH]
#
##  only for non-linear where there are multiple solutions case
# while any([eq.is_positive for eq in eqns_family]):
#     for i, eq in enumerate(eqns_family):
#         if eq.is_positive:
#             eqns_family.pop(i)

# solutions = sym.solve_poly_system(eqns_family, *[y[species_lst.index(s)] for s in set(reduce(add, families.values()))])
# while len(solutions)>1:
#     print('two solutions exists')
#     # eliminate solutions that are definately negative
#     tf = [any([s.is_negative==True for s in sol]) for sol in solutions]
#     if any(tf):
#         print('eliminate the negative solution')
#         _ = solutions.pop(tf.index(True))
#     else:
#         print('eliminate the last solution')
#         _ = solutions.pop(-1)
# y_solve_family = {s: sol for s,sol in 
#     zip([y[species_lst.index(s)] for s in reduce(add, families.values())], solutions[0])}
