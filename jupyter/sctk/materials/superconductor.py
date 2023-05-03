from typing import Union, List, Dict
import re

from .material import Material, Formula, squeeze_formula, parse_formula
from .element import Element


class Superconductor(Material):
    
    _SUPERCONDUCTOR_FORMULA_MAP = {
        'LBCO' : 'Cu Ba 0.15 La 1.85 O 4',
        'LSCO' : 'La 1.85 Sr 0.15 Cu O 4',
        'YBCO' : 'Y Ba 2 Cu 3 O 7-δ',
        r'Y[ -]?(1?[0-9])-(1?[0-9])-(1?[0-9])' : r'Y a Ba b Cu c O 7-δ',
        r'Y[ -]?([0-9])[-]?([0-9])[-]?([0-9])' : r'Y a Ba b Cu c O 7-δ',
        'PrBCO' : 'PrBa2Cu3O7',
        r'Pr[ -]?([0-9])[-]?([0-9])[-]?([0-9])' : r'Pr a Ba b Cu c O 7-δ',
        'BSCCO' : 'Bi 2 Sr 2 Ca 1 Cu 2 O 8+δ',
        r'Bi[ -]?([0-9])[-]?([0-9])[-]?([0-9])[-]?([0-9])' : r'Bi a Sr b Ca c Cu d O 4+2d+δ',
        'TBCCO' : 'Tl 2 Ba 2 Ca 2 Cu 3 O 10',
        r'Tl[ -]?([0-9])[-]?([0-9])[-]?([0-9])[-]?([0-9])' : r'Tl a Ba b Ca c Cu d O 4+2d+δ',
        'H[g]?BCCO' : 'HgBa 2 Ca 2 Cu 3 O 8+δ',
        r'Hg[ -]?([0-9])[-]?([0-9])[-]?([0-9])[-]?([0-9])' : r'Hg a Ba b Ca c Cu d O 4+2d+δ',
        'EuBCO' : 'Eu Ba 2 Cu 3 O 6',
        r'Eu[ -]?([0-9])[-]?([0-9])[-]?([0-9])': r'Eu a Ba b Cu d O 6',
        'GdBCO' : 'Gd Ba 2 Cu 3 O 7',
        r'Gd[ -]?([0-9])[-]?([0-9])[-]?([0-9])': r'Gd a Ba b Cu d O 7',
        r'V[ -]?([0-9])[-]?([0-9])[-]?([0-9])[-]?([0-9])[-]?([0-9])' : r'Sr a Sc b Fe c As d O e',
        r'Sm[ -]?([0-9])[-]?([0-9])[-]?([0-9])[-]?([0-9])' : r'Sm a Fe b As c O d',
        r'Ca[ -]?(1?[0-9])-(1?[0-9])-(1?[0-9])': r'Ca a Ir b As c (Fe 2 As 2) 5'
    }
    
    def __init__(self, formula: Union[str, Formula], variable_elements=None, canonical_variables=False):
        # screen for canonical superconductor formulas:
        substitutions = None

        for key, key_form in self._SUPERCONDUCTOR_FORMULA_MAP.items():
            if match := re.match(key,formula):
                substitutions = {
                    chr(ord('a')+i) : int(val)
                    for i, val in enumerate(match.groups())
                }
                if not canonical_variables:
                    substitutions['δ'] = 0

                formula = squeeze_formula(parse_formula(key_form))
                break
        
        # initialize self:
        super().__init__(formula, variable_elements)
        
        # substitute canonical variables (if any)
        if substitutions:
            self.substitute(substitutions)