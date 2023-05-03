from typing import List, Union, Optional, Tuple, Dict, Callable
from itertools import product, chain
from fractions import Fraction

from ..materials.material import Material, Element, PeriodicTable
from .phase_diagram import composition_distance

from ase import Atoms, Atom
from ase.neighborlist import mic
from ase.geometry import wrap_positions
from ase.build import make_supercell, find_optimal_cell_shape
import numpy as np
import random as python_random

class DefectSupercellBuilder():

    unitcell: Atoms

    __ALLOWED_LOCS = [ 'auto', 'center']
    __ALLOWED_SUPERCELL_MATRIX_METHODS = [ 'fcc', 'sc', 'ppd' ]

    def __init__(self, unitcell: Atoms):
        self.unitcell = unitcell
        self.substitutions = {}
        self.interstitials = {}
        self.vacancies = {}
        self.supercell_matrix = np.eye(3)

    def get_unitcell_material(self):
        return Material(self.unitcell.get_chemical_formula())

    def get_unitcell_size(self):
        return len(self.unitcell)

    def add_substitution(self, sub: Union[Element, str], elem: Union[Element,str], n: Union[int,float]=1, loc: Union[str,Tuple[float]] = 'auto'):
        # get periodic table entry for element:
        elem = PeriodicTable[elem]

        # validate location:
        if isinstance(loc, str) and loc not in DefectSupercellBuilder.__ALLOWED_LOCS:
            raise Exception(f'loc value must be one of: {DefectSupercellBuilder.__ALLOWED_LOCS}.')
        elem_config = (elem, sub, loc)
        
        # update substitutions table:,
        if elem_config not in self.vacancies:
            self.substitutions[elem_config] = 0
        self.substitutions[elem_config] += n

    def add_vacancy(self, elem: Union[Element, str], n: Union[int,float] = 1, loc: Union[str,Tuple[float]] = 'auto'):
        # get periodic table entry for element:
        elem = PeriodicTable[elem]

        # validate location:
        if isinstance(loc, str) and loc not in DefectSupercellBuilder.__ALLOWED_LOCS:
            raise Exception(f'loc value must be one of: {DefectSupercellBuilder.__ALLOWED_LOCS}.')
        elem_config = (elem, loc)
        
        # update vacancies table:
        if elem_config not in self.vacancies:
            self.vacancies[elem_config] = 0
        self.vacancies[elem_config] += n
    

    def add_interstitial(self, elem: Union[Element, str], n:Union[int,float] = 1, loc: Union[str,Tuple[float]] = 'auto'):
        # get periodic table entry for element:
        elem = PeriodicTable[elem]

        # validate location:
        if isinstance(loc, str) and loc not in DefectSupercellBuilder.__ALLOWED_LOCS:
            raise Exception(f'loc value must be one of: {DefectSupercellBuilder.__ALLOWED_LOCS}.')
        elem_config = (elem, loc)
        
        # update vacancies table:
        if elem_config not in self.interstitials:
            self.interstitials[elem_config] = 0
        self.interstitials[elem_config] += n

    def set_supercell_matrix(self, m: Union[str,int,List[int],np.ndarray], n_unitcells: Optional[int]=None):

        if isinstance(m, int):
            m = np.eye(3)*m
        elif isinstance(m, str):
            if m not in DefectSupercellBuilder.__ALLOWED_SUPERCELL_MATRIX_METHODS:
                raise Exception(f'Method {m} must be one of {DefectSupercellBuilder.__ALLOWED_SUPERCELL_MATRIX_METHODS}')
            if n_unitcells is None:
                    raise Exception('A value of n_unitcells must be given.')
            if m == 'sc' or m == 'fcc':
                m = find_optimal_cell_shape(self.unitcell.cell,n_unitcells)
            if m == 'ppd':
                factors = sorted(_split_into_similar_factors(n_unitcells,3),reverse=True)
                basis_vectors = sorted((np.linalg.norm(b),i) for i,b in enumerate(self.unitcell.cell))
                m = np.zeros(3)
                for (_,i),factor in zip(basis_vectors, factors):
                    m[i] = factor
        
        m = np.array(m)
        if len(m.shape) == 1:
            m = np.diag(m)

        self.supercell_matrix = m

    def dope_like_material(self, 
                           material: Union[str,Material],
                           method: str =  'ppd',
                           n_unitcells: int = 45,
                           matching_tolerance: float = 4.0,
                           centering = True,
                           compatibility_fn: Optional[Callable[[Element,Element],bool]] = None):
        
        # generate unitcell material:
        unitcell_material = Material(self.unitcell.get_chemical_formula())

        # solve for supercell expansion ratio (i.e. N : M, if one exists within tolerance):
        nm, nm_dist = _find_optimal_ratio_match(unitcell_material,material, tolerance=matching_tolerance)
        if nm is None:
            raise Exception(f'Unit cell does not match material within matching tolerance {matching_tolerance}.')
        supercell_N = nm[0]
        material_M = nm[1]

        # detect pairing of ratios for substitutions, vacancies, and defects based on indicated material:
        supercell_rounding_comp = {
            k : v*supercell_N for k,v in unitcell_material.get_composition().items()
        }
        material_rounding_comp = {
            k : v*material_M for k,v in material.get_composition().items()
        }

        found_pairs, found_vacancies, found_interstitials = \
            _get_rounded_material_defects(supercell_rounding_comp, material_rounding_comp)

        # determine supercell defects:
        supercell_substitutions = {
            (k1,k2) : int(np.round(v1*n_unitcells/supercell_N))
            for (k1, k2), (v1,v2) in found_pairs.items()
            if int(np.round(v1*n_unitcells/supercell_N))
        }
        supercell_vacancies = {
            k : int(np.round(v*n_unitcells/supercell_N))
            for k, v in found_vacancies.items()
            if int(np.round(v*n_unitcells/supercell_N))
        }
        supercell_interstitials = {
            k : int(np.round(v*n_unitcells/supercell_N))
            for k, v in found_interstitials.items()
            if int(np.round(v*n_unitcells/supercell_N))
        }

        #print('substitutions: ', supercell_substitutions)
        #print('vacancies: ', supercell_vacancies)
        #print('interstitials: ', supercell_interstitials)

        # set supercell matrix:
        #print('n_unitcells: ', n_unitcells)
        self.clear()
        self.set_supercell_matrix(method,n_unitcells)

        # add defects to this supercell builder:
        n_lattice_defects = sum(chain(supercell_substitutions.values(),
                                supercell_vacancies.values()))
        n_interstitial_defects = sum(supercell_interstitials.values())
        lattice_defect_loc = 'center' if centering and n_lattice_defects <= 1 else 'auto'
        interstitial_defect_loc = 'center' if centering and n_interstitial_defects <= 1 else 'auto'

        for (e1,e2), v in supercell_substitutions.items():
            self.add_substitution(e1,e2,v,lattice_defect_loc)
        for e, v in supercell_vacancies.items():
            self.add_vacancy(e, v, lattice_defect_loc)
        for e, v in supercell_interstitials.items():
            self.add_interstitial(e,v, interstitial_defect_loc)

    def get_supercell_material(self):
        sc_formula = make_supercell(self.unitcell, self.supercell_matrix).get_chemical_formula()
        return Material(sc_formula)

    def get_supercell_size(self):
        return len(make_supercell(self.unitcell, self.supercell_matrix))

    def get(self, random: bool = False):

        # fix python and numpy random number generator:
        if not random:
            np.random.seed(seed=0)
            python_random.seed(0)

        # construct supercell:
        sc = make_supercell(self.unitcell, self.supercell_matrix).copy()
        #print(self.supercell_matrix)
        #print('Supercell: ', Material(sc.get_chemical_formula()).str())

        #print('Adding vacancies: ', self.vacancies)
        # add vacancies:
        if self.vacancies:
            vacancy_sites = _select_defect_sites(sc, self.vacancies)
            vacancy_indexes = [ i for idxs in vacancy_sites.values() for i in idxs ]
            vacancy_indexes.sort(reverse=True)
            for i in vacancy_indexes:
                sc.pop(i)

        #print('Adding substitutions: ', self.substitutions)
        # add substitutions (select sites by grouping together substitutions applied to the same element):
        substitution_groups = {}
        substitution_group_counts = {}
        for (elem, sub, loc), n in self.substitutions.items():
            elem_config = (elem, loc)

            if elem_config not in substitution_groups:
                substitution_groups[elem_config] = 0
            substitution_groups[elem_config] += int(n)

            if elem not in substitution_group_counts:
                substitution_group_counts[elem] = []
            substitution_group_counts[elem].append((sub, int(n)))

        # select substitution sites:
        if substitution_groups:
            substitution_sites = _select_defect_sites(sc, substitution_groups)
            for elem, idxs in substitution_sites.items():
                python_random.shuffle(idxs)
                for (sub, count) in substitution_group_counts[elem]:
                    for _ in range(count):
                        sc.numbers[idxs.pop()] = PeriodicTable[sub].atomic_number

        #print('Adding interstitials: ', self.interstitials)
        # select interstitial sites:
        if self.interstitials:
            interstitial_sites = _select_interstitial_sites(sc, self.interstitials)
            for elem, points in interstitial_sites.items():
                for p in points:
                    sc.append(Atom(str(elem), p))

        return sc

    def clear(self):
        # clear all defects:
        self.supercell_matrix = np.eye(3)
        self.substitutions = {}
        self.interstitials = {}
        self.vacancies = {}

def _select_defect_sites(supercell, defects, rbf_potential='gaussian', gamma=None) -> List[int]:

    positions = supercell.positions
    scaled_positions = supercell.get_scaled_positions()
    center = np.mean(scaled_positions, axis=0)
    supercell_elements = [ PeriodicTable[int(n)] for n in supercell.numbers ]

    # estimate the value of gamma using an empirical rule of thumb:
    if gamma is None:
        total_n = np.ceil(sum(defects.values()))
        gamma = (supercell.cell.volume / max(1,total_n))**(1/3)

    # determine possible vacant sites:
    element_sites = {}     # element -> list of indexes
    selected_sites = {}    # element -> list of indexes (fixed non-'auto' sites only)
    auto_defects = {}      # element -> count ('auto' defects only)
    auto_defect_sites = {} # element -> list of indexes for element ('auto' defects only)

    for i, elem in enumerate(supercell_elements):
        if elem not in element_sites:
            element_sites[elem] = []
            auto_defect_sites[elem] = []
        element_sites[elem].append(i)
        auto_defect_sites[elem].append(i)
    
    # select vacancies closest to specified positions:
    for (elem,loc), n in defects.items():
        if isinstance(loc, str) and loc == 'auto':
            auto_defects[elem] = n
            continue
        if isinstance(loc, str) and loc == 'center':
            loc = center
        else:
            loc = np.array(loc)
        auto_defect_sites[elem].sort(key=lambda i : np.sum((scaled_positions[i]-loc)**2))

        if elem not in selected_sites:
            selected_sites[elem] = []
        
        selected_sites[elem].extend(auto_defect_sites[elem][:int(n)])
        auto_defect_sites[elem] = auto_defect_sites[elem][int(n):]

    # build reduced site vectors:
    auto_idxs = [ i for idxs in auto_defect_sites.values() for i in idxs ]
    auto_defect_sites_idxs = {
        elem : [ auto_idxs.index(i) for i in idxs ]
        for elem, idxs in auto_defect_sites.items()
    }

    # build distance matrix:
    N_auto = len(auto_idxs)
    D = np.zeros((N_auto,N_auto))
    R_inv = np.zeros((N_auto,N_auto))
    for i in range(N_auto):
        for j in range(i):
            R_inv[i,:] = 1./gamma
            p1, p2 = positions[auto_idxs[i]], positions[auto_idxs[j]]
            D[i,j] = D[j,i] = np.sqrt(np.sum(mic(p2 - p1,supercell.cell)**2))

    # apply rbf function to distance matrix to obtain kernel matrix:
    K = np.zeros((N_auto,N_auto))
    if rbf_potential == 'gaussian':
        K = np.exp(-0.5 * (D*R_inv)**2)
    elif rbf_potential == 'inverse_quadratic':
        K = 1.0/(1.0 + (D*R_inv)**2)

    # if necessary, perform genetic algorithm search for automatically placed defect sites:
    history = None
    v_optimal = np.zeros(N_auto)
    if sum(auto_defects.values()) > 0:
        history = _ga_search(K, v_optimal, auto_defects, auto_defect_sites_idxs, poolsize=max(16,N_auto))
        _, p_optimal, v_optimal = min(history, key=lambda t: t[0])

    # construct dictionary of chosen defect sites:
    defect_idxs = set( auto_idxs[i] for (i,x) in enumerate(v_optimal) if x > 0)

    defect_sites = {} # final map of element -> list of chosen site indices
    for elem, idxs in element_sites.items():
        for i in idxs:
            if i in defect_idxs:
                if elem not in defect_sites:
                    defect_sites[elem] = []
                defect_sites[elem].append(i)

    return defect_sites



def _combine_pair(v_tgt, v_src, v_init, possible_sites, prob_cross=0.5, prob_mutate=0.25):
    v_new = v_init.copy()
    
    # iterate over each element to be selected:
    for element_idxs in possible_sites.values():
        
        v_tgt_idxs = []
        v_src_idxs = []
        v_none_idxs = []

        # classify each index based on whether it appears in each parent:
        for i in element_idxs:
            x_tgt, x_src = v_tgt[i], v_src[i]
            if x_tgt > 0 and x_src > 0:
                assert(v_new[i] == 0.0)
                v_new[i] = 1.0
            elif x_src > 0:
                v_src_idxs.append(i)
            elif x_tgt > 0:
                v_tgt_idxs.append(i)
            else:
                v_none_idxs.append(i)

        python_random.shuffle(v_src_idxs)
        python_random.shuffle(v_none_idxs)
        
        for idx in v_tgt_idxs:
            # cross or mutate with some small probability:
            if np.random.uniform(0,1) < prob_cross and v_src_idxs:
                v_new[v_src_idxs.pop()] = 1.0
            elif np.random.uniform(0,1) < prob_mutate and v_none_idxs:
                v_new[v_none_idxs.pop()] = 1.0
            else:
                assert(v_new[idx] == 0.0)
                v_new[idx] = 1.0

    assert(np.sum(v_new) == np.sum(v_tgt))
    assert(np.sum(v_new) == np.sum(v_src))
    return v_new

def _crossover(pool, v_init, possible_sites, n_offspring):
    offspring = np.zeros((len(pool)*n_offspring, len(pool[0])))
    pairs = np.random.randint(0, len(pool), size=(len(offspring), 2))
    for i in range(len(offspring)):
        v1 = pool[pairs[i,0]]
        v2 = pool[pairs[i,1]]
        offspring[i] = _combine_pair(v1, v2, v_init, possible_sites)

    return offspring

def _ga_search(K, v_init, element_counts, element_sites, poolsize=128, n_generations = 80, n_offspring=4):

    # determine remaining substitution sites for each element:
    possible_sites = {
        elem : sorted([ i for i in element_sites[elem] if not v_init[i] ])
        for elem in element_counts
    }

    # initialize pool:
    pool = np.zeros((poolsize, len(v_init)))
    history = []
    for n in range(poolsize):

        # generate randomly sampled trait vector:
        sampled_sites = {
            elem : python_random.sample(element_sites[elem], int(count))
            for elem,count in element_counts.items()
        }
        v_sample = v_init.copy()
        for idxs in sampled_sites.values():
            for i in idxs:
                v_sample[i] = 1.0

        pool[n] = v_sample

    # iterate over each generation:
    for _ in range(n_generations):
        
        # generate next generation pool:
        offspring = _crossover(pool, v_init, possible_sites, n_offspring)
        
        # compute potential and select best candidates:
        generation_potentials = []
        for i in range(len(offspring)):
            potential = (np.dot(offspring[i], (K @ offspring[i].T)))
            generation_potentials.append((potential, i))

        # select only the best offspring:
        generation_potentials.sort()
        min_potential, min_idx = generation_potentials[0]
        mean_potential = np.mean(generation_potentials)
        history.append((mean_potential, min_potential, offspring[min_idx]))
        for i, (potential, idx) in enumerate(generation_potentials[:len(pool)]):
            pool[i] = offspring[idx]
        
    return history


def _select_interstitial_sites(supercell, interstitials,
                               rbf_potential='inverse_quadratic',
                               gamma_supercell=None,
                               gamma=None,
                               eta=0.5):
    positions = supercell.positions
    center = np.mean(positions, axis=0)

    # estimate the value of the potential gammas using an empirical rule of thumb:
    if gamma_supercell is None:
        total_n = np.ceil(len(supercell))
        gamma_supercell = 0.5 * (supercell.cell.volume / max(1,total_n))**(1/3)
    if gamma is None:
        total_n = np.ceil(sum(interstitials.values()))
        gamma = (supercell.cell.volume / max(1,total_n))**(1/3)

    # set initial pointpositions:
    elem_points = {}
    points = []
    for (elem,loc), n in interstitials.items():
        if elem not in elem_points:
            elem_points[elem] = 0
        elem_points[elem] += int(n)
        if isinstance(loc, str) and loc == 'auto':
            for _ in range(int(n)):
                points.append(python_random.choice(positions)+np.random.normal(0,eta))
            continue
        if isinstance(loc, str) and loc == 'center':
            for _ in range(int(n)):
                points.append(center + np.random.normal(0,eta))
        else:
            for _ in range(int(n)):
                points.append(supercell.cartesian_positions(loc) + np.random.normal(0,eta))
    points = np.array(points)

    # relax defect positions via gradient descent:
    relaxed_points = []
    if len(points) > 0:
        n_steps = 30
        _, relaxed_points = _grad_descent(supercell, points, 
                                                    rbf_potential,
                                                    gamma_supercell=gamma_supercell,
                                                    gamma_points=gamma,
                                                    eta=0.1*gamma_supercell,
                                                    n_steps=n_steps)

    relaxed_points = list(relaxed_points)
    python_random.shuffle(relaxed_points)
    interstitial_sites = {
        elem : [ relaxed_points.pop() for _ in range(n) ]
        for elem, n in elem_points.items()
    }
    
    return interstitial_sites


def _grad_descent(supercell, points,
                  rbf_potential='inverse_quadratic', 
                  gamma_supercell=1.0, 
                  gamma_points=1.0, 
                  eta=1.0,
                  damping_coeff=0.01,
                  n_steps=128):
    points = points.copy()
    potentials = np.zeros(n_steps)
    
    for i in range(n_steps//2):
        potentials[i] = np.sum(_compute_potentials(supercell, points, rbf_potential, 
                                                   damping_coeff*gamma_supercell, gamma_points))
        gradients = _compute_gradients(supercell, points, rbf_potential,
                                       damping_coeff*gamma_supercell, gamma_points)
        grad_norm_factor = np.maximum(1e-5,np.sqrt(np.sum(gradients**2, axis=1)))
        grad_norm_factor = np.repeat(np.expand_dims(grad_norm_factor,-1),3,axis=1)
        points += (eta*gradients/grad_norm_factor)
        points = wrap_positions(points, supercell.cell)
    for i in range(n_steps//2, n_steps):
        potentials[i] = np.sum(_compute_potentials(supercell, points, rbf_potential, 
                                                   gamma_supercell, damping_coeff*gamma_points))
        gradients = _compute_gradients(supercell, points, rbf_potential, 
                                       gamma_supercell, damping_coeff*gamma_points)
        grad_norm_factor = np.maximum(1e-5,np.sqrt(np.sum(gradients**2, axis=1)))
        grad_norm_factor = np.repeat(np.expand_dims(grad_norm_factor,-1),3,axis=1)
        points += (eta*gradients/grad_norm_factor)
        points = wrap_positions(points, supercell.cell)

    return potentials, points

def _compute_potentials(supercell, points, 
                        rbf_potential, 
                        gamma_supercell, 
                        gamma_points):
    V = np.zeros(len(points))
    for i, x in enumerate(points):
        for p in supercell.positions:
            dx = mic(p - x, supercell.cell)
            if rbf_potential == 'gaussian':
                V[i] += np.exp(-0.5*np.sum((dx/gamma_supercell)**2))
            elif rbf_potential == 'inverse_quadratic':
                V[i] += 1.0 / (1.0 + np.sum((dx/gamma_supercell)**2))**2
        for p in points:
            dx = mic(p - x, supercell.cell)
            if rbf_potential == 'gaussian':
                V[i] += np.exp(-0.5*np.sum((dx/gamma_points)**2))
            elif rbf_potential == 'inverse_quadratic':
                V[i] += 1.0/ (1.0 + np.sum((dx/gamma_points)**2))**2
    return V

def _compute_gradients(supercell, points, 
                       rbf_potential, 
                       gamma_supercell, 
                       gamma_points):

    V_grad = np.zeros_like(points)
    for i, x in enumerate(points):
        for p in supercell.positions:
            dx = mic(p - x, supercell.cell)
            if rbf_potential == 'gaussian':
                V_grad[i] += -dx*np.exp(-0.5*np.sum((dx/gamma_supercell)**2))/gamma_supercell**2
            elif rbf_potential == 'inverse_quadratic':
                V_grad[i] += -2.0*dx / (1.0 + np.sum((dx/gamma_supercell)**2))**2
        for p in points:
            dx = mic(p - x, supercell.cell)
            if rbf_potential == 'gaussian':
                V_grad[i] += -dx*np.exp(-0.5*np.sum((dx/gamma_points)**2))/gamma_points**2
            elif rbf_potential == 'inverse_quadratic':
                V_grad[i] += -2.0*dx / (1.0 + np.sum((dx/gamma_points)**2))**2
    return V_grad
    
def _get_rounded_material_defects(supercell_comp: dict, target_comp: dict, compatability_fn=None):

    for elem, count in target_comp.items():
        if count is None:
            raise Exception(f'Unable to determine count for element {elem} in target with composition {target_comp}.')

    # determine the differences in composition:
    union_elements = set(supercell_comp.keys()) | set(target_comp.keys())
    differences = {}
    for elem in union_elements:
        supercell_count = supercell_comp[elem] if elem in supercell_comp else 0.0
        target_count = target_comp[elem] if elem in target_comp else 0.0
        differences[elem] = target_count - supercell_count

    found_pairs = {}
    found_vacancies = {}
    found_interstitials = {}

    # atttempt to group pairs of differences that correspond to substitution doping:
    RESIDUE_CUTOFF = 1e-3
    pair_residues = [
        (diff1+diff2, e1, e2)
        for (e1, diff1) in differences.items()
        for (e2, diff2) in differences.items()
        if e1.atomic_number < e2.atomic_number and
        ( diff1 != 0 or diff2 != 0)
    ]

    # sort differences by floor absolute residue:
    pair_residues.sort(key=lambda x : abs(x[0]))
    matched_elements = set()

    # find stoichiometrically paired elements (greedy strategy):
    for (res, e1, e2) in pair_residues:
        if abs(res) < RESIDUE_CUTOFF and e1 not in matched_elements and \
            e2 not in matched_elements:

            if differences[e1] < 0.0:
                e1, e2 = e2, e1

            # ensure elements are not incompatible:
            if compatability_fn is not None and not compatability_fn(e1, e2):
                continue

            matched_elements.update([e1,e2])
            found_pairs[(e1, e2)] = (differences[e1], -differences[e1])
            
    # include the remaining values as defects/vacancies:
    for elem, diff in differences.items():
        if elem not in matched_elements and abs(diff) > (RESIDUE_CUTOFF/2.):
            if diff > 0.0:
                found_interstitials[elem] = abs(diff)
            else:
                found_vacancies[elem] = abs(diff)
    
    return found_pairs, found_vacancies, found_interstitials

def _find_optimal_ratio_match(a: Material, b: Material, tolerance: float = 4.0) -> Optional[Tuple[int,int]]:

    a_comp = a.get_composition()
    b_comp = b.get_composition()

    max_n = 2*int(sum(v for v in a_comp.values()) + \
                  sum(v for v in b_comp.values()))
    min_dist = tolerance + 1e-8
    nm = None
    
    for n, m in product(range(1,max_n), range(1,max_n)):
        if np.gcd(n,m) == 1:
            n_a_comp = { k: v*n for k,v in a_comp.items() }
            m_b_comp = { k: v*m for k,v in b_comp.items() }
            
            unnorm_dist = composition_distance(n_a_comp, m_b_comp, normalized=False)
            if unnorm_dist < min_dist:
                min_dist = unnorm_dist
                nm = (n,m)
    
    return nm, min_dist

def _split_into_similar_factors(N, n_factors):
    N = max(1,abs(N))
    found_factors = []
    for _ in range(n_factors):
        divisors = [ d for d in range(1,N+1) if N % d == 0 ]
        if divisors:
            n_factor = min(divisors, key = lambda x : np.abs(x - N**(1/n_factors)))
            found_factors.append(n_factor)
            N = N//n_factor
            n_factors -= 1

    return found_factors

