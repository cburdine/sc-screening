from typing import List, Union, Optional, Tuple, Dict, Callable
from itertools import chain, combinations

from ..materials.material import Material, Element, PeriodicTable
from ..notebook.nbconfig import using_notebook
from ..databases.database import Database

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from tqdm import tqdm
import tqdm.notebook as nbtqdm
from ase import Atoms

ElementList = List[Union[Element,str]]
RangeList = List[Tuple[Optional[float], Optional[float]]]
CompositionDict = Dict[Element,Union[float, int, None]]

PHASE_DIAGRAM_OVERLAYs = [
    'formation_energy',
    'total_energy'
]

def composition_distance(a: Union[Material,CompositionDict],
                         b: Union[Material,CompositionDict], normalized=True) -> float:
    distance = 0.0
    norm_a = norm_b = 1.0

    
    composition_a = a.get_composition() if isinstance(a, Material) else a
    composition_b = b.get_composition() if isinstance(b, Material) else b
    
    elements = set(composition_a.keys()) | set(composition_b.keys())

    if normalized:
        norm_a = sum(v if v else 0.0 for v in composition_a.values())
        norm_b = sum(v if v else 0.0 for v in composition_b.values())

    if norm_a == 0:
        raise Exception(f'Material "{a.get_formula_string(pretty=False)}" has non-normalizable composition.')
    if norm_b == 0:
        raise Exception(f'Material "{b.get_formula_string(pretty=False)}" has non-normalizable composition.')

    for elem in elements:
        count_a = composition_a[elem] if elem in composition_a else 0.0
        count_b = composition_b[elem] if elem in composition_b else 0.0
        if count_a is None:
            count_a = 0.0
        if count_b is None:
            count_b = 0.0
        
        distance += abs(count_a/norm_a - count_b/norm_b)

    return distance

def _nonempty_subsets(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

MATERIAL_DISTANCES = {
    'composition': lambda a, b : composition_distance(a,b,normalized=True),
    'formula':     lambda a, b : composition_distance(a,b,normalized=False)
}

class StructurePhaseDiagram():

    __PHASE_COORDS = [ np.array([-0.5,-np.sqrt(3)/6.]),
                       np.array([0.5, -np.sqrt(3)/6.]),
                       np.array([0.0, np.sqrt(3)/3]) ]

    __PHASE_ORIGIN = np.array([0.0, 0.0])

    def __init__(self, elements: Union[Material, str, ElementList]):
        
        # parse alternative elements input:
        if isinstance(elements, str):
            elements = Material(elements)
        if isinstance(elements, Material):
            elements = list(elements.get_composition().keys())

        # ensure elements are converted to Element objects:
        elements = [ PeriodicTable[e] for e in elements ]

        self.element_set = set(elements)
        self.material_dict = {}
        self.structure_dict = {}
        self.composition_dict = {}
        self.thermodynamics_dict = {}

    def add_structure(self, structure: Atoms, 
                      material : Union[str,Material,None], 
                      material_id: Optional[str] = None,
                      fe_per_atom: Optional[float] = None,
                      e_per_atom: Optional[float] = None,
                      ehull: Optional[float] = None):
        if material is None:
            material = Material(structure.get_chemical_formula())
        elif isinstance(material, str):
            material = Material(material)

        material_id = 'added-'+str(id(material))
        self.material_dict[material_id] = material
        self.structure_dict[material_id] = structure
        self.composition_dict[material_id] = material.get_composition()
        self.thermodynamics_dict[material_id] = {
            'formation_energy_per_atom': fe_per_atom,
            'energy_per_atom': e_per_atom,
            'energy_above_hull': ehull,
        }

    def add_database_structures(self, db : Database, pbar: bool = True):

        # configure progress bar:
        if not pbar:
            _tqdm = lambda x : x
        else:
            _tqdm = nbtqdm.tqdm if using_notebook() else tqdm
        
        # iterate over powerset of listed elements:
        subsets = list(_nonempty_subsets(self.element_set))
        for element_tuple in _tqdm(subsets):
            element_subset = list(element_tuple)

            # add materials containing elements of interest:
            db_material_dict = db.get_materials_with_elements(element_subset)
            db_structure_dict = db.get_structures_with_elements(element_subset)
            db_composition_dict = {}
            db_thermodynamics_dict = {}
            
            # add thermodynamics data (if supported):
            db_thermodynamics_fn = getattr(db, 'get_thermodynamic_data_with_elements', None)
            if callable(db_thermodynamics_fn):
                db_thermodynamics_dict |= db_thermodynamics_fn(element_subset)

            for id in db_structure_dict.keys():
                # only consider materials containing a subset of the atoms in the material of interest:
                id_composition = db_material_dict[id].get_composition()
                total_count = sum(id_composition.values())
                db_composition_dict[id] = {
                    elem : count/total_count 
                    for (elem,count) in id_composition.items()
                }

                # add blank thermodynamics data:
                if id not in db_thermodynamics_dict:
                    db_thermodynamics_dict[id] = {
                        'formation_energy_per_atom': None,
                        'energy_per_atom': None,
                        'energy_above_hull': None
                    }

            # merge database data with phase diagram data:
            self.material_dict |= db_material_dict
            self.structure_dict |= db_structure_dict
            self.composition_dict |= db_composition_dict
            self.thermodynamics_dict |= db_thermodynamics_dict

    def find_closest_structure(self, material: Union[Material,str],
                               metric: Union[str,Callable] = 'composition',
                               max_ehull: Optional[float] = None,
                               max_energy_per_atom: Optional[float] = None,
                               max_fe_per_atom: Optional[float] = None) -> Tuple[Optional[Material], Optional[Atoms]]:
        
        # convert to material:
        if isinstance(material, str):
            material = Material(material)
        
        # resolve distance metric
        if isinstance(metric, str):
            if metric not in MATERIAL_DISTANCES:
                raise Exception(f'Metric must be a callable or one of {MATERIAL_DISTANCES}.')
            metric = MATERIAL_DISTANCES[metric]

        # sort first by composition distance, second by thermodynamic quantities (if given)
        distances = sorted([
            (
                metric(material, m),
                self.thermodynamics_dict[m_id]['energy_above_hull'] \
                    if self.thermodynamics_dict[m_id]['energy_above_hull'] else np.inf,
                self.thermodynamics_dict[m_id]['energy_per_atom'] \
                    if self.thermodynamics_dict[m_id]['energy_per_atom'] else np.inf,
                self.thermodynamics_dict[m_id]['formation_energy_per_atom'] \
                    if self.thermodynamics_dict[m_id]['formation_energy_per_atom'] else np.inf,
                m_id
            )
            for m_id, m in self.material_dict.items()
        ])

        for _, _, _, _, m_id in distances:
            # check if thermodynamic constraints are satisfied:
            if m_id in self.thermodynamics_dict:
                thermodynamics = self.thermodynamics_dict[m_id]
                ehull = thermodynamics['energy_above_hull']
                energy_per_atom = thermodynamics['energy_per_atom']
                fe_per_atom = thermodynamics['formation_energy_per_atom']

                if max_ehull is not None and (ehull is None or ehull > max_ehull):
                    continue
                if max_energy_per_atom is not None and (energy_per_atom is None or energy_per_atom > max_energy_per_atom):
                    continue
                if max_fe_per_atom is not None and (fe_per_atom is None or fe_per_atom > max_fe_per_atom):
                    continue
            
            if m_id in self.material_dict and m_id in self.structure_dict:
                return self.material_dict[m_id], self.structure_dict[m_id]

        return None, None

    def _plot_triangle_interpolation(self, ax, x, y, z, vertices):
        triang = mtri.Triangulation(x,y)
        vertices = np.array(vertices)
        interp_cubic_geom = mtri.LinearTriInterpolator(triang, z)
        x_rng = (np.min(x), np.max(x))
        y_rng = (np.min(y), np.max(y))
        X, Y = np.meshgrid(np.linspace(*x_rng, 1000), np.linspace(*y_rng, 1000))
        Z = interp_cubic_geom(X,Y)
        ax.contourf(X, Y, Z)
        

    def plot(self, 
             element_subset: Optional[ElementList] = None, ax = None, 
             subset_only: bool = False, 
             show_ehull=False,
             overlay: Union[str,None] = None) -> Tuple[Optional[Material], Optional[Atoms]]:
        
        if not element_subset:
            element_subset = list(self.element_set)
        else:
            element_subset = [ PeriodicTable[e] for e in element_subset ]

        if not (2 <= len(element_subset) <= 3):
            raise Exception(f'Element subset must be of length 2 or 3 (got: {element_subset})')
        if not set(element_subset).issubset(self.element_set):
            raise Exception(f'Element subset {element_subset} must be a subset of element set ' + \
                            f'{list(self.element_set)}.')

        # compute coordinates of each element in the phase diagram:
        element_coords = {
            elem : x
            for elem,x in zip(element_subset, StructurePhaseDiagram.__PHASE_COORDS)
        }
        
        # compute the coordinates of each material (by composition) as a weighted average
        # of each element:
        shown_subset = set(element_subset) if subset_only else set(self.element_set)
        composition_coords = {
            id: sum(
                element_coords[e]*weight 
                for (e,weight) in comp.items()
                if e in element_coords
            ) + StructurePhaseDiagram.__PHASE_ORIGIN
            for id, comp in self.composition_dict.items()
            if set(comp.keys()).issubset(shown_subset)
        }

        # generate new ax if not given:
        if ax is None:
            ax = plt.subplot(111)

        # compute border and coordinates:
        border = StructurePhaseDiagram.__PHASE_COORDS[:len(element_subset)]
        border.append(border[0])
        border = np.array(border).T
        coords = np.array(list(composition_coords.values())).T
        coord_ids = list(composition_coords.keys())

        # reduce composition coordinates to minimal set:
        coord_counts = {
            coord : [] for coord in 
            set(tuple(v) for v in composition_coords.values())
        }
        for id, coord in composition_coords.items():
            coord_counts[tuple(coord)].append(id)

        # compute energies above hull (if available):
        ehulls = [ self.thermodynamics_dict[id]['energy_above_hull'] for id in coord_ids ]
        formation_energies = [ self.thermodynamics_dict[id]['formation_energy_per_atom'] for id in coord_ids ]
        max_ehull = max([0] + [ eh for eh in ehulls if eh is not None ])

        # compute overlay mesh:
        # overlay_pts = []
        # if overlay:
        #     if isinstance(overlay, str):
                

        # handle 1D-visualization:
        
        plot_1d = (len(element_subset) < 3)
        if plot_1d:
            coords[1] = ehulls if show_ehull else 0
            border[1] = 0
        else:
            # set ehulls to indicate the lowest ehull for each plotted coordinate:
            for coord, ids in coord_counts.items():
                id_rep = min(ids, 
                         key=lambda id : self.thermodynamics_dict[id]['energy_above_hull'] 
                         if self.thermodynamics_dict[id]['energy_above_hull'] else np.inf)
                ehull_rep = self.thermodynamics_dict[id_rep]['energy_above_hull']
                for id in ids:
                    ehulls[coord_ids.index(id)] = max_ehull if ehull_rep is None else ehull_rep
        
        # finilaize ehull array (account for unlisted ehull values)
        ehulls = np.array([ eh if eh is not None else max_ehull for eh in ehulls ]) if show_ehull else None


        # plot border and scatter plot of materials:
        ax.plot(border[0], border[1])
        self._plot_triangle_interpolation(ax, coords[0], coords[1], ehulls, list(element_coords.values()))
        sc = ax.scatter(coords[0], coords[1], c=ehulls, cmap='hot')
        ax.set_frame_on(False)

        # add colorbars:
        if show_ehull:
            plt.colorbar(sc, label='Energy Above Hull (eV/atom)')


        # plot internal coordinates of materials:
        for coord, ids in coord_counts.items():
            id_rep = min(ids, 
                         key=lambda id : self.thermodynamics_dict[id]['energy_above_hull']
                         if self.thermodynamics_dict[id]['energy_above_hull'] else np.inf)
            formula_str = self.material_dict[id_rep].get_formula_string(fmt='latex')
            if len(ids) > 1:
                formula_str += f' ({len(ids)})'
            x = coord[0]
            y = 0.01 if plot_1d else coord[1]
            ax.annotate(formula_str, (x,y), rotation=(90 if plot_1d else 0))
        
        # plot coordinates of base elements (Large text)
        for elem, coord in element_coords.items():
            x = coord[0] if plot_1d else coord[0]*1.2
            y = -max(0.008,0.08*max_ehull) if plot_1d else coord[1]*1.03
            ax.annotate(elem.symbol, (x,y), fontsize=20, ha='center')

        ax.set_xlim(-0.7,0.7)
        if not plot_1d:
            ax.set_ylim(-0.4, 0.6)

        return ax

        