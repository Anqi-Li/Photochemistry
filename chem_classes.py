import numpy as np

class RateConstant:
    def __init__(self, value=None, unit=None, ref=None, name=None, reaction_name=None):
        self.value = value
        self.unit = unit
        self.ref = ref
        self.name = name
        self.reaction_name = reaction_name

    def set_value(self, value):
        self.value = value
        return self
    def set_unit(self, unit):
        self.unit = unit
        return self
    def set_ref(self, ref):
        self.ref = ref
    def set_name(self, name):
        self.name = name
        return self
    def set_reaction_name(self, reaction_name):
        self.reaction_name = reaction_name
        return self
    def remove_value(self):
        self.value = None

class Species:
    def __init__(self, density=None, unit=None, name=None):
        self.density = density
        self.unit = unit
        self.name = name

    def set_density(self, density):
        self.density = density
        return self
    def set_unit(self, unit):
        self.unit = unit
        return self
    def set_name(self, name):
        self.name = name
        return self
    def remove_density(self):
        self.density = None
            

class Reaction: #general reactions
    reaction_type = 'genral reaction'
    
    def __init__(self,  
                 rate_constant=RateConstant(), 
                 reactants=(), 
                 products=(),  
                 name=None):
        self.rate_constant = rate_constant
        self.reactants = reactants
        self.products = products
        self.name = name

    def set_rate_constant(self, rate_constant):
        self.rate_constant = rate_constant
        return self
    def set_reactants(self, reactants):
        self.reactants = reactants
        return self
    def set_products(self, products):
        self.products = products
        return self
    def set_name(self, name):
        self.name = name
        return self
    
    def rate(self):        
        rate = self.rate_constant.value * np.prod([s.density for s in self.reactants], axis=0)
        return rate
        
    def show(self):
        reactant_names = [x.name for x in self.reactants]
        product_names = [x.name for x in self.products]
        print(self.name, reactant_names, self.rate_constant.name, product_names)

    def show_reactant_names(self):
        print([x.name for x in self.reactants]) 

    def get_r_stoich(self):
        return {x.name: self.reactants.count(x) for x in self.reactants}

    def get_net_stoich(self, families=None):
        species = set(self.reactants + self.products)
        return {x.name: self.products.count(x)-self.reactants.count(x) for x in species}


#%% chemical reaction systems
import sympy as sym
T = sym.symbols('T', nonnegative=True, real=True, positive=True)
n2 = Species(name='N2')
o2 = Species(name='O2')
o3 = Species(name='O3')
o = Species(name='O')
o1d = Species(name='O_1D')
o2sig = Species(name='O2_Sigma')
o2del = Species(name='O2_Delta')
h2o = Species(name='H2O')
h = Species(name='H')
oh = Species(name='OH')
h2 = Species(name='H2')
h2o2 = Species(name='H2O2')
ho2 = Species(name='HO2')
m = Species(name='M')
ch4 = Species(name='CH4')
co = Species(name='CO')

#from Allen 1984
def get_reactions_allen1984():
    reaction_lst = [ 
        Reaction(reactants=(o2,), products=(o, o), name='R1'),
        Reaction(reactants=(o2,), products=(o, o1d), name='R2'),
        Reaction(reactants=(o3,), products=(o2, o), name='R3'),
        Reaction(reactants=(o3,), products=(o2, o1d), name='R4'),
        Reaction(reactants=(h2o,), products=(h, oh), name='R5'), #remove 
        Reaction(reactants=(h2o,), products=(h2, o1d), name='R6'), #remove
        Reaction(reactants=(h2o2,), products=(oh, oh), name='R7'),
        Reaction(reactants=(o1d, o2), products=(o, o2), name='R8'),
        Reaction(reactants=(o1d, n2), products=(o, n2), name='R9'),
        Reaction(reactants=(o1d, h2o), products=(oh, oh), name='R10'),
        Reaction(reactants=(o1d, h2), products=(h, oh), name='R11'),
        Reaction(reactants=(o, o, m), products=(o2, m), name='R12'),
        Reaction(reactants=(o, o, o2), products=(o3, o), name='R13'),
        Reaction(reactants=(o, o2, o2), products=(o3, o2), name='R14'),
        Reaction(reactants=(o, o2, n2), products=(o3, n2), name='R15'),
        Reaction(reactants=(o, o3), products=(o2, o2), name='R16'),
        Reaction(reactants=(o, oh), products=(o2, h), name='R17'),
        Reaction(reactants=(o, ho2), products=(oh, o2), name='R18'),
        Reaction(reactants=(o, h2o2), products=(oh, ho2), name='R19'),
        Reaction(reactants=(o, h2), products=(oh, h), name='R20'),
        Reaction(reactants=(oh, o3), products=(ho2, o2), name='R21'),
        Reaction(reactants=(oh, oh), products=(h2o, o), name='R22'),
        Reaction(reactants=(oh, ho2), products=(h2o, o2), name='R23'),
        Reaction(reactants=(oh, h2o2), products=(h2o, ho2), name='R24'),
        Reaction(reactants=(oh, h2), products=(h2o, h), name='R25'),
        Reaction(reactants=(ho2, o3), products=(oh, o2, o2), name='R26'),
        Reaction(reactants=(ho2, ho2), products=(h2o2, o2), name='R27'),
        Reaction(reactants=(h, o2, m), products=(ho2, m), name='R28'),
        Reaction(reactants=(h, o3), products=(oh, o2), name='R29'),
        Reaction(reactants=(h, ho2), products=(h2, o2), name='R30'),
        Reaction(reactants=(h, ho2), products=(oh, oh), name='R31'),
        Reaction(reactants=(h, ho2), products=(h2o, o), name='R32'),
        Reaction(reactants=(h, h, m), products=(h2, m), name='R33'),
        # r34 
        Reaction(reactants=(ch4, oh), products=(co, oh, h2o, h2o), name='R35'),
        Reaction(reactants=(ch4, o), products=(co, oh, oh, h2o), name='R36'),
        Reaction(reactants=(ch4, o1d), products=(co, oh, oh, h2o), name='R37'),
        ]

    rate_const_lst = [
        # from Allen 1984
        RateConstant(name='J1', unit='s-1', reaction_name='R1'), 
        RateConstant(name='J2', unit='s-1', reaction_name='R2'), 
        RateConstant(name='J3', unit='s-1', reaction_name='R3'), 
        RateConstant(name='J4', unit='s-1', reaction_name='R4'), 
        RateConstant(name='J5', unit='s-1', reaction_name='R5'), 
        RateConstant(name='J6', unit='s-1', reaction_name='R6'), 
        RateConstant(name='J7', unit='s-1', reaction_name='R7'),
        RateConstant(value=3.2e-11*sym.exp(117/T), name='k8', unit='cm3s-1', reaction_name='R8'),
        RateConstant(value=1.8e-11*sym.exp(157/T), name='k9', unit='cm3s-1', reaction_name='R9'),
        RateConstant(value=2.3e-10*sym.exp(-100/T), name='k10', unit='cm3s-1', reaction_name='R10'),
        RateConstant(value=1.1e-10, name='k11', unit='cm3s-1', reaction_name='R11'),
        RateConstant(value=9.59e-34*sym.exp(480/T), name='k12', unit='cm6s-1', reaction_name='R12'),
        RateConstant(value=2.15e-34*sym.exp(345/T), name='k13', unit='cm6s-1', reaction_name='R13'),
        RateConstant(value=2.15e-34*sym.exp(345/T), name='k14', unit='cm6s-1', reaction_name='R14'),
        RateConstant(value=8.82e-35*sym.exp(575/T), name='k15', unit='cm6s-1', reaction_name='R15'),
        RateConstant(value=1.5e-11*sym.exp(-2218/T), name='k16', unit='cm3s-1', reaction_name='R16'),
        RateConstant(value=2.3e-11*sym.exp(-90/T), name='k17', unit='cm3s-1', reaction_name='R17'),
        RateConstant(value=2.8e-11*sym.exp(172/T), name='k18', unit='cm3s-1', reaction_name='R18'),
        RateConstant(value=1.0e-11*sym.exp(-2500/T), name='k19', unit='cm3s-1', reaction_name='R19'),
        RateConstant(value=1.6e-11*sym.exp(-4570/T), name='k20', unit='cm3s-1', reaction_name='R20'),
        RateConstant(value=1.6e-12*sym.exp(-940/T), name='k21', unit='cm3s-1', reaction_name='R21'),
        RateConstant(value=4.5e-12*sym.exp(-275/T), name='k22', unit='cm3s-1', reaction_name='R22'),
        RateConstant(value=8.4e-11, name='k23', unit='cm3s-1', reaction_name='R23'),
        RateConstant(value=2.9e-12*sym.exp(-160/T), name='k24', unit='cm3s-1', reaction_name='R24'),
        RateConstant(value=7.7e-12*sym.exp(-2100/T), name='k25', unit='cm3s-1', reaction_name='R25'),
        RateConstant(value=1.4e-14*sym.exp(-580/T), name='k26', unit='cm3s-1', reaction_name='R26'),
        RateConstant(value=2.4e-14*sym.exp(1250/T), name='k27', unit='cm3s-1', reaction_name='R27'),
        RateConstant(value=1.76e-28*T**(-1.4), name='k28', unit='cm6s-1', reaction_name='R28'),
        RateConstant(value=1.4e-10*sym.exp(-270/T), name='k29', unit='cm3s-1', reaction_name='R29'),
        RateConstant(value=6.0e-12, name='k30', unit='cm3s-1', reaction_name='R30'),
        RateConstant(value=7.0e-11, name='k31', unit='cm3s-1', reaction_name='R31'),
        RateConstant(value=2.3e-12, name='k32', unit='cm3s-1', reaction_name='R32'),
        RateConstant(value=1.0e-30*T**(-0.8), name='k33', unit='cm6s-1', reaction_name='R33'),
        #       RateConstant(value=6.0e-12, name='k34'),
        RateConstant(value=2.4e-12*sym.exp(-1710/T), name='k35', unit='cm3s-1', reaction_name='R35'),
        RateConstant(value=3.5e-11*sym.exp(-4550/T), name='k36', unit='cm3s-1', reaction_name='R36'),
        RateConstant(value=1.4e-10, name='k37', unit='cm3s-1', reaction_name='R37'),
        ]
    #set available rate constant to each reaction
    reaction_lst = [r.set_rate_constant(next(k for k in rate_const_lst if k.reaction_name==r.name)) for r in reaction_lst]
    return reaction_lst

                
# from Thomas 1984
def get_reactions_thomas1984():
    reaction_lst = [
        Reaction(reactants=(o3,), products=(o1d, o2del), name='J_o3'), #R4 in allen1984
        Reaction(reactants=(o1d, o2), products=(o, o2sig), name='Q_1do2'), #R8 in allen1984
        Reaction(reactants=(o1d, n2), products=(o, n2), name='Q_1dn2'), #R9 in allen1984
        Reaction(reactants=(o2sig, m), products=(o2del, m), name='Q_Sigma_m'),
        Reaction(reactants=(o2del, o2), products=(o2, o2), name='Q_Delta_o2'),
        Reaction(reactants=(o2sig,), products=(o2,), name='A_Sigma'),
        Reaction(reactants=(o2del,), products=(o2,), name='A_Delta'),
        Reaction(reactants=(o2,), products=(o2sig,), name='g_A')
        ]
    rate_const_lst = [
        RateConstant(name='J_o3', reaction_name='J_o3', unit='s-1'), 
        RateConstant(value=2.9e-11*sym.exp(-67/T), name='k_1do2', reaction_name='Q_1do2', unit='cm3s-1'), 
        RateConstant(value=1.0e-11*sym.exp(-107/T), name='k_1dn2', reaction_name='Q_1dn2', unit='cm3s-1'), 
        RateConstant(value=1.0e-15, name='k_Sigma_m', reaction_name='Q_Sigma_m', unit='cm3s-1'), 
        RateConstant(value=2.22e-18*sym.exp(T/300)**0.78, name='k_Delta_o2', reaction_name='Q_Delta_o2', unit='cm3s-1'), 
        RateConstant(value=0.085, name='A_Sigma', reaction_name='A_Sigma', unit='s-1'), 
        RateConstant(value=2.58e-4, name='A_Delta', reaction_name='A_Delta', unit='s-1'),
        RateConstant(name='g_A', reaction_name='g_A', unit='s-1')
        ]
    #set available rate constant to each reaction
    reaction_lst = [r.set_rate_constant(next(k for k in rate_const_lst if k.reaction_name==r.name)) for r in reaction_lst]
    return reaction_lst

#from Frederick 1979
def get_reactions_frederick1979():
    reaction_lst = [
        Reaction(reactants=(o, o2, m), products=(o3, m), name='k1'),
        Reaction(reactants=(o3,), products=(o, o2), name='J3'), #J3 in allen 1984
        Reaction(reactants=(o3, h), products=(o2, oh), name='k3'), 
        Reaction(reactants=(o, oh), products=(o2, h), name='k4'),
        Reaction(reactants=(h, o2, m), products=(ho2, m), name='k5'),
        Reaction(reactants=(oh, o3), products=(ho2, o2), name='k6'),
        Reaction(reactants=(o, ho2), products=(o2, oh), name='k7')]
    rate_const_lst = [
        RateConstant(value=1.07e-34*sym.exp(510/T), name='k1', unit='cm3s-1', reaction_name='k1'), 
        RateConstant(name='J3', unit='s-1', reaction_name='J3'), 
        RateConstant(value=1e-10*sym.exp(-516/T), name='k3', unit='cm3s-1', reaction_name='k3'), 
        RateConstant(value=4.2e-11, name='k4', unit='cm3s-1', reaction_name='k4'), 
        RateConstant(value=2.08e-32*sym.exp(290/T), name='k5', unit='cm3s-1', reaction_name='k5'), 
        RateConstant(value=1.5e-12*sym.exp(-1000/T), name='k6', unit='cm3s-1', reaction_name='k6'), 
        RateConstant(value=3.5e-11, name='k7', unit='cm3s-1', reaction_name='k7')]
    reaction_lst = [r.set_rate_constant(next(k for k in rate_const_lst if k.reaction_name==r.name)) for r in reaction_lst]
    return reaction_lst

def get_reactions_custom():
    reaction_lst = get_reactions_allen1984()
    reaction_set = ['R{}'.format(n) for n in '1 2 3 14 15 17 18 21 28 29'.split()]
    reaction_lst = [r for r in reaction_lst if r.name in reaction_set]
    reaction_lst += get_reactions_thomas1984()
    return reaction_lst

def get_reactions_custom2():
    reaction_lst = [ 
        Reaction(reactants=(o2,), products=(o, o), name='R1'),
        Reaction(reactants=(o2,), products=(o, o1d), name='R2')]
    rate_const_lst = [
        RateConstant(name='J1', unit='s-1', reaction_name='R1'), 
        RateConstant(name='J2', unit='s-1', reaction_name='R2')]
    reaction_lst = [r.set_rate_constant(next(k for k in rate_const_lst if k.reaction_name==r.name)) for r in reaction_lst]

    reaction_lst += get_reactions_thomas1984() 
    reaction_lst += get_reactions_frederick1979() 
    return reaction_lst
    