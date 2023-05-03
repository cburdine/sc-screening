from requests import get
from typing import Optional, Dict, List, Union
from io import StringIO
import os
import tempfile
from contextlib import ExitStack

from ..materials.material import Material
from ..materials.element import Element

from mp_api.client import MPRester
from ase import Atoms
from ase.io.cif import read_cif

ElementList = List[Union[Element,str]]

class Database:
    
    info_url: str
    docs_url: str

    def __init__(self, info_url: str, docs_url: str):
        self.info_url = info_url
        self.docs_url = docs_url

    def get_info_url(self) -> str:
        return self.info_url

    def get_api_docs_url(self) -> str:
        return self.docs_url

    def get_structure_by_id(self, id: str) -> Optional[Atoms]:
        raise NotImplementedError()

    def get_structures(self, material: Material) -> Dict[str,Atoms]:
        raise NotImplementedError()

    def get_structure_ids(self, material: Material) -> List[str]:
        raise NotImplementedError()

    def get_structures_with_elements(self, elements: ElementList, contains_all: bool =True) -> Dict[str,Atoms]:
        raise NotImplementedError()

    def get_materials_with_elements(self, elements: ElementList, contains_all: bool =True) -> Dict[str,Material]:
        raise NotImplementedError()

    def get_structure_ids_with_elements(self, elements: ElementList, contains_all: bool = True) -> List[str]:
        raise NotImplementedError()

    def query_by_id(self, id: str) -> Optional[dict]:
        raise NotImplementedError()

    def query(self, params: Dict[str,str]) -> Dict[str,dict]:
        raise NotImplementedError()

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

class MaterialsProject(Database):

    _MP_INFO_URL: str = 'https://materialsproject.org/'
    _MP_DOCS_URL: str = 'https://materialsproject.org/api'

    _MAX_STRUCTURE_EHULL = None

    mp_connector: Optional[MPRester]
    _cache_dir: str
    _stack: Optional[ExitStack]

    def __init__(self, mp_api_key=None):
        super().__init__(MaterialsProject._MP_INFO_URL,
                         MaterialsProject._MP_DOCS_URL)
        if not mp_api_key:
            if len(os.environ['MP_API_KEY']):
                mp_api_key = os.environ['MP_API_KEY']
            else:
                raise Exception('Materials Project API key must be given or ' + \
                                'set in the environment variable $MP_API_KEY')
        
        self._cache_dir = tempfile.TemporaryDirectory()
        self.mp_connector = MPRester(mp_api_key)
        self._stack = ExitStack()

    def __enter__(self):
        self._stack.__enter__()
        self._stack.enter_context(self._cache_dir)
        self._stack.enter_context(self.mp_connector)
        return self

    def __exit__(self, type, value, traceback):
        self._stack.__exit__(type, value, traceback)

    def get_structure_by_id(self, id: str) -> Optional[Atoms]:
        
        id_file = os.path.join(self._cache_dir.name, f'{id}.cif')

        # check if file is cached, if not, fetch data from database:
        if not os.path.isfile(id_file):
            structure = self.mp_connector.get_structure_by_material_id(id)
            structure.to(id_file, fmt='cif')

        # return structure as atoms:
        atoms = next(read_cif(id_file, index=slice(None), store_tags=True))
        return atoms

    def get_structures(self, material: Material) -> Dict[str,Atoms]:
        docs = self.mp_connector.summary.search(
                                formula=material.get_formula_string(pretty=False), 
                                energy_above_hull = (0.0, MaterialsProject._MAX_STRUCTURE_EHULL),
                                fields=['material_id', 'structure'])
        
        structures = {}
        for doc in docs:
            # cache material structure:
            id_file = os.path.join(self._cache_dir.name, f'{doc.material_id}.cif')
            doc.structure.to(id_file, fmt='cif')

            # parse structure as atoms:
            atoms = next(read_cif(id_file, index=slice(None), store_tags=True))
            structures[doc.material_id] = atoms

        return structures

    def get_structure_ids(self, material: Material) -> List[str]:
        docs = self.mp_connector.summary.search(
                                formula=material.get_formula_string(pretty=False), 
                                energy_above_hull = (0.0, MaterialsProject._MAX_STRUCTURE_EHULL),
                                fields=['material_id'])
        
        return [ str(doc.material_id) for doc in docs ]

    def get_structures_with_elements(self, elements: ElementList, contains_all: bool =True) -> Dict[str,Atoms]:
        
        # parse list of elements:
        element_symbols = [
            e.symbol if isinstance(e, Element) else e
            for e in elements
        ]
        
        # search for materials containing elements
        docs = self.mp_connector.summary.search(
                                elements=element_symbols, 
                                energy_above_hull = (0.0, MaterialsProject._MAX_STRUCTURE_EHULL),
                                num_elements = len(elements) if contains_all else None,
                                fields=['material_id', 'structure'])
        
        structures = {}
        for doc in docs:
            # cache material structure:
            id_file = os.path.join(self._cache_dir.name, f'{doc.material_id}.cif')
            doc.structure.to(id_file, fmt='cif')

            # parse structure as atoms:
            atoms = next(read_cif(id_file, index=slice(None), store_tags=True))
            structures[str(doc.material_id)] = atoms

        return structures

    def get_materials_with_elements(self, elements: ElementList, contains_all: bool =True) -> Dict[str, Material]:
        # parse list of elements:
        element_symbols = [
            e.symbol if isinstance(e, Element) else e
            for e in elements
        ]

        
        # search for materials containing at most the given elements:
        docs = self.mp_connector.summary.search(
                                elements=element_symbols,
                                energy_above_hull = (0.0, MaterialsProject._MAX_STRUCTURE_EHULL),
                                num_elements = len(elements) if contains_all else None,
                                fields=['material_id', 'formula_pretty'])

        return {
            str(doc.material_id) :  Material(str(doc.formula_pretty)) 
            for doc in docs
        }

    def get_thermodynamic_data_with_elements(self, elements: ElementList, contains_all: bool =True) -> Dict[str,dict]:
        # parse list of elements:
        element_symbols = [
            e.symbol if isinstance(e, Element) else e
            for e in elements
        ]

        # search for materials containing at most the given elements:
        thermodynamic_fields = ['energy_above_hull', 
                                'formation_energy_per_atom', 
                                'energy_per_atom' ]
        docs = self.mp_connector.summary.search(
                                elements=element_symbols,
                                energy_above_hull = (0.0, MaterialsProject._MAX_STRUCTURE_EHULL),
                                num_elements = len(elements) if contains_all else None,
                                fields=['material_id'] + thermodynamic_fields)

        # parse thermodynamic fields:
        thermodynamic_results = {}
        for doc in docs:
            doc_dict = dict(doc)
            thermodynamic_results[str(doc.material_id)] = {
                field : doc_dict[field] for field in thermodynamic_fields
            }
        
        return thermodynamic_results

    def get_structure_ids_with_elements(self, elements: ElementList, contains_all: bool =True) -> List[str]:
        # parse list of elements:
        element_symbols = [
            e.symbol if isinstance(e, Element) else e
            for e in elements
        ]
        
        # search for materials containing elements
        docs = self.mp_connector.summary.search(
                                elements=element_symbols, 
                                energy_above_hull = (0.0, MaterialsProject._MAX_STRUCTURE_EHULL),
                                num_elements = len(elements) if contains_all else None,
                                fields=['material_id'])

        return [ str(doc.material_id) for doc in docs ]

    def query_by_id(self, id: str) -> Optional[dict]:
        docs = self.mp_connector.summary.search(material_ids=[id])
        return dict(docs[0]) if docs else None

    def query(self, params: Dict[str,str]) -> Dict[str,dict]:
        
        # ensure mp ids are retured in query:
        params = params.copy()
        if 'fields' in params and ( 'material_id' not in params['fields']):
            params['fields'] = list(params['fields'])
            params['fields'].append('material_id')
        
        # perform query and return raw results:
        docs = self.mp_connector.summary.search(**params)
        results = {
            doc.material_id : dict(doc) for doc in docs
            for doc in docs
        }

        for doc in docs:
            # cache material structures (if present):
            if 'structure' in doc:
                id_file = os.path.join(self._cache_dir.name, f'{doc.material_id}.cif')
                doc.structure.to(id_file, fmt='cif')

                # parse structure as atoms:
                atoms = next(read_cif(id_file, index=slice(None), store_tags=True))
                results[doc.material_id]['parsed_structure'] = atoms
        
        return results


class COD(Database):

    _COD_INFO_URL: str = 'http://www.crystallography.net/cod/index.php'
    _COD_DOCS_URL: str = 'https://wiki.crystallography.net/RESTful_API/'
    _COD_SEARCH_ENDPOINT: str = 'http://www.crystallography.net/cod/result'
    _COD_DATA_ENDPOINT: str = 'http://www.crystallography.net/cod/'

    def __init__(self):
        super().__init__(COD._COD_INFO_URL, 
                         COD._COD_DOCS_URL)

    def __enter__(self):
        return self

    def __exit__(self,type, value, traceback):
        pass
    
    def get_structure_by_id(self, id: str) -> Optional[Atoms]:

        # make HTTP request:
        response = get(COD._COD_DATA_ENDPOINT+f'{id}.cif')
        if response.status_code == 404:
            return None
        elif response.status_code != 200:
            raise Exception(f'HTTP Request failed with status code {response.status_code}.')
        else:
            cif_file = StringIO(response.text)
            atoms = next(read_cif(cif_file, index=slice(None), store_tags=True))
            return atoms

    def get_structures(self, material: Material) -> Dict[str,Atoms]:

        # retrieve structure ids:
        cod_ids = self.get_structure_ids(material)

        # retrieve all structures:
        structures = {}
        for cod_id in cod_ids:
            try:
                structure = self.get_structure_by_id(cod_id)
                structures[cod_id] = structure
            except:
                continue

        return structures

    def get_structure_ids(self, material: Material) -> List[str]:
        # handle invalid materials:
        if material.variable_elements:
            raise Exception(f'Unable to query material with variable elements {material.variable_elements}.')
        if material.variables:
            raise Exception(f'Unable to query material with variables {material.variables}.')

        # query data:
        cod_query_params = { 'formula': material.get_formula_string(fmt='cod') }
        results = self.query(cod_query_params).keys()

        return [ str(result) for result in results ]
    
    def get_structures_with_elements(self, elements: ElementList, contains_all: bool =True) -> Dict[str,Atoms]:
        
        # retrieve structure ids:
        cod_ids = self.get_structure_ids_with_elements(elements, contains_all)

        # retrieve all structures:
        structures = {}
        for cod_id in cod_ids:
            try:
                structure = self.get_structure_by_id(cod_id)
                structures[cod_id] = structure
            except:
                continue
        
        return structures

    def get_materials_with_elements(self, elements: ElementList, contains_all: bool =True) -> Dict[str, Material]:
        # parse list of elements:
        element_symbols = [
            e.symbol if isinstance(e, Element) else e
            for e in elements
        ]

        # validate and trim element list (COD only supports 8 distinct elements):
        if not element_symbols:
            raise Exception('Elements list cannot be empty')
        elif len(element_symbols) > 8:
            element_symbols = element_symbols[:8]
        element_symbols.sort()
        
        # construct query parameters:
        params = { f'el{i+1}' : sym for i, sym in enumerate(element_symbols) }
        if contains_all:
            params['strictmax'] = str(len(element_symbols))

        # perform query:
        results = self.query(params)

        # return results:
        materials = {}
        for cod_id, fields in results.items():
            mat = Material(fields['formula'].strip(' -'))
            materials[cod_id] = mat

        return materials

    def get_structure_ids_with_elements(self, elements: ElementList, contains_all: bool =True) -> List[str]:
        
        # parse list of elements:
        element_symbols = [
            e.symbol if isinstance(e, Element) else e
            for e in elements
        ]

        # validate and trim element list (COD only supports 8 distinct elements):
        if not element_symbols:
            raise Exception('Elements list cannot be empty')
        elif len(element_symbols) > 8:
            element_symbols = element_symbols[:8]
        element_symbols.sort()
        
        # construct query parameters:
        params = { f'el{i+1}' : sym for i, sym in enumerate(element_symbols) }
        if contains_all:
            params['strictmax'] = str(len(element_symbols))

        # perform query:
        results = self.query(params)

        # return results:
        return [ str(cod_id) for cod_id in results.keys() ]

    def query_by_id(self, id: str) -> Optional[dict]:
        id_params = { 'id' : id }
        results = self.query(id_params).values()
        return results[0] if results else None
    
    def query(self, params: Dict[str, str]) -> Dict[str,dict]:
        
        if not params:
            raise Exception('Parameter dict cannot be empty.')

        fmt_params = { 'format': 'json' }
        response = get(COD._COD_SEARCH_ENDPOINT, params=(params|fmt_params))
        if response.status_code == 404:
            return dict()
        elif response.status_code != 200:
            raise Exception(f'HTTP Request failed with status code {response.status_code}.')
        else:
            results = response.json()
            return { res['file'] : res for res in results }
            
