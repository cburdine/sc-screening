from typing import List, Union, Optional, Tuple, Dict, Callable
import re
from copy import deepcopy

from .element import Element, PeriodicTable, _PeriodicTableElements

import numpy as np

ElementList = List[Union[Element,str]]
Formula = List[Tuple[Union[list,Element],Union[float,str]]]
FormulaValue = Union[int,float,str]
ElementSubstitutionDict = Dict[Union[str,Element],Union[str,Element]]

__FORMULA_VALUE_RE = r'([-+−])?([0-9]*\.[0-9]+|[0-9]+)?([a-zαβγδ])?'
__FORMULA_SPACE_RE = r'(-|[_ ]?)'

def _make_formula_regex(elements: ElementList):
    
    # compile list of symbols (with larger symbols matched first):
    element_symbols = [
        e.symbol if isinstance(e,Element) else e 
        for e in elements
    ]
    match_order_symbols = sorted(element_symbols, key=lambda s : -len(s))
    
    # construct formula regex:
    count_re = f'(({__FORMULA_VALUE_RE})*)'
    space_re = __FORMULA_SPACE_RE
    elem_re = r'([\(\[\)\]]|' + '|'.join(match_order_symbols) + ')'
    return re.compile(elem_re + space_re + count_re)

def _parse_formula_value(val: str) -> FormulaValue:
    if val.isdigit():
        return int(val)
    if val.replace('.','',1).isdigit():
        return float(val)
    return val

def _parse_formula_variables(formula: Formula) -> Tuple[List[str],List[Element]]:
    
    count_term_re = re.compile(__FORMULA_VALUE_RE)
    variables = set()
    variable_elements = set()
    for (elem, count) in formula:

        # parse Element part of tuple
        if isinstance(elem,list):
            vs, ves = _parse_formula_variables(elem)
            variables.update(vs)
            variable_elements.update(ves)
        elif isinstance(elem,Element):
            if elem.symbol not in PeriodicTable:
               variable_elements.add(elem)
        else:
            raise Exception(f'Element in formula must be of type Element, not {type(elem)}.')

        # parse count part of tuple:
        if isinstance(count,str):
            last_end = 0
            for match in count_term_re.finditer(count):

                # validate count term:
                unmatched_str = match.string[last_end:match.start()].strip()
                if unmatched_str:
                    raise Exception(f'Error parsing token "{unmatched_str}" in formula "{formula}".')
                last_end = match.end()

                # record seen variables:
                if var := match.group(3):
                    variables.add(var)
    
    return list(variables), list(variable_elements)

def parse_formula(formula: str, 
                  variable_elements: Optional[ElementList] = None,
                  strict: bool = True) -> Formula:

    if not formula:
        raise Exception('Formula cannot be empty.')

    element_dict = { e.symbol : e for e in PeriodicTable }
    if variable_elements:
        # generate new formula regex:
        parsed_var_elems = [
            Element(e) if isinstance(e,str) else e 
            for e in variable_elements
        ]
        regex = _make_formula_regex(_PeriodicTableElements + parsed_var_elems)

        # add variable elements to element dict:
        element_dict |= { e.symbol : e for e in parsed_var_elems }           
    else:
        # load default regex (create if not initialized)
        try:
            regex = parse_formula._default_formula_regex
        except AttributeError:
            regex = _make_formula_regex(_PeriodicTableElements)
            parse_formula._default_formula_regex = regex

    # parse formula sequentially, iterating over matched elements: 
    tokens = []
    tokens_stack = []
    last_end = 0
    for re_match in regex.finditer(formula):
        
        # if strict parsing, check for invalid tokens:
        unmatched_str = re_match.string[last_end:re_match.start()].strip('- ~+,')
        if unmatched_str and strict:
            raise Exception(f'Error parsing token "{unmatched_str}" in formula "{formula}".')
        last_end = re_match.end()

        elem, count = None, None
        if re_match.group(1) in '([':
            tokens_stack.append(tokens)
            tokens = []
            continue
        
        if re_match.group(1) in ')]':
            if tokens_stack:
                elem = tokens
                tokens = tokens_stack.pop()
            elif strict:
                # if closing parentheses not matched, raise exception (if strict)
                raise Exception('Unmatched ")" or "]" in formula.')
            else:
                continue
        else:
            elem = re_match.group(1)
            if elem not in element_dict:
                raise Exception(f'Unknown element: "{elem}" in formula "{formula}".')
            elem = element_dict[elem]

        if elem:
            # parse count as a float or int, depending on type:
            count = re_match.group(3) if re_match.group(3) else '1'
            if count in ['+', '-', '~']:
                raise Exception('Sign token {count} must be followed by value.')
            if count.lstrip('+-').isdigit():
                count = int(count)
            elif count.lstrip('+-').replace('.','',1).isdigit():
                count = float(count)
            
            tokens.append((elem, count))
    
    # ensure there is no unprocessed trailing text:
    if strict and last_end != len(formula):
        unmatched_str = formula[last_end:]
        raise Exception(f'Trailing text "{unmatched_str}" in formula "{formula}')
    
    # if parentheses not closed, raise an exception (if strict)
    if strict and tokens_stack:
        raise Exception('Unmatched "(" or "[" in formula.')
    while tokens_stack:
        parent_tokens = tokens_stack.pop()
        parent_tokens.append((tokens, 1))
        tokens = parent_tokens
    
    return tokens 
    

def squeeze_formula(formula:Formula) -> Formula:
    new_formula = []
    for (elem,count) in formula:
        if count == 0:
            continue

        if isinstance(elem, list):
            if count == 1:
                new_formula.extend(squeeze_formula(elem))
            else:
                new_formula.append((squeeze_formula(elem), count))
        else:
            new_formula.append((elem, count))

    return new_formula

def _substitute_formula_variable(formula: Formula, 
                                 x: str, 
                                 y: FormulaValue):
    count_term_re = re.compile(__FORMULA_VALUE_RE)

    if len(x) != 1:
        raise Exception(f'Variable "{x}" must be a single character.')
    if isinstance(y,str) and len(y) != 1:
        raise Exception(f'Substituted variable "{y}" can only be a single character.')
    
    remove_elem_idxs = []
    for i, (elem, count) in enumerate(formula):
        if isinstance(elem, list):
            _substitute_formula_variable(elem, x, y)
        if isinstance(count, str) and x in count:
            count_sum = 0
            remaining_terms = []

            for match in count_term_re.finditer(count):
                if not match.group(0):
                    continue

                sign = match.group(1) if match.group(1) else '+'
                coeff = match.group(2)
                var = match.group(3)
                n = _parse_formula_value(coeff) if coeff else 1
                if sign == '-':
                    n *= -1

                # add together all terms without a variable:
                if var is None:
                    count_sum += n
                elif var == x:
                    # perform variable substitution (numeric or symbolic)
                    if isinstance(y, str):
                        new_term = match.group(0).replace(x,y)
                        if not count_term_re.match(new_term):
                            raise Exception('Substitution of "{y}" for "{x}" produces invalid term "{new_term}".')
                        remaining_terms.append(new_term)
                    else:
                        count_sum += n*y
                elif match.group(0):
                    # keep all terms not containing the variable as they are:
                    rem_term = match.group(0)
                    if rem_term[0] not in '+-':
                        rem_term = ('+' + rem_term)
                    remaining_terms.append(rem_term)

            # if all variables are resolved, substitute the numeric value:
            if not remaining_terms:
                # check if element needs to be removed:
                if count_sum == 0:
                    remove_elem_idxs.append(i)
                else:
                    formula[i] = (elem,count_sum)
            else:
                # compile string form of new count:
                sum_term = str(count_sum) if count_sum != 0 else ''
                count_str = sum_term + ''.join(remaining_terms)
                if count_str and count_str[0] == '+':
                    count_str = count_str[1:]
                formula[i] = (elem, count_str)

    # remove all elements with count 0:
    for idx in sorted(remove_elem_idxs, reverse=True):
        if idx < len(formula):
            formula.pop(idx)


def _substitute_formula_element(formula: Formula,
                                x: Union[str,Element],
                                y: Union[str,Element]):
    
    # resolve x and y, if a known Element:
    if isinstance(x,str):
        x = PeriodicTable[x] if x in PeriodicTable else Element(x)
    if isinstance(y,str):
        y = PeriodicTable[y] if y in PeriodicTable else Element(y)
    
    # recursively perform element substitution:
    for i, (elem,count) in enumerate(formula):
        if isinstance(elem, list):
            _substitute_formula_element(elem,x,y)
        if elem == x:
            formula[i] = (y,count)

def _to_subscript(x: str) -> str:
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()αβγδ"
    sub_s =  "ABCDEFGHIJKLMNOPQRSTUVWXYZₐbcᑯₑf₉ₕᵢⱼₖₗₘₙₒₚqᵣₛₜᵤᵥwₓᵧz₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎𝛼ᵦᵧ𝛿"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)

def _formula_to_string(formula: Formula,
                       pretty: bool=False,
                       subscript_begin: Optional[str] = None,
                       subscript_end: Optional[str] = None,
                       elem_spacing: Optional[str] = None,
                       sort_key: Optional[Callable] = None) -> str:
    str_parts = []
    
    # set default spacing in formula:
    if not subscript_begin:
        subscript_begin = ''
    if not subscript_end:
        subscript_end = ''
    if not elem_spacing:
        elem_spacing = ''
    
    for (elem, count) in formula:
        
        # build count string:
        if isinstance(count, str):
            count_str = count
        else:
            count_str = '' if count == 1.0 else str(round(count,3))
        if pretty:
            count_str = _to_subscript(count_str)

        if len(count_str) > 0:
            count_str = ''.join([subscript_begin,count_str,subscript_end])

        # build element component:
        if isinstance(elem, list):
            elem_str = _formula_to_string(elem,
                                          pretty=pretty,
                                          subscript_begin=subscript_begin,
                                          subscript_end=subscript_end,
                                          elem_spacing=elem_spacing)
            str_parts.append(f'({elem_str})'+count_str)
        else:
            str_parts.append(elem.symbol+count_str)
    
    if sort_key:
        str_parts.sort(key=sort_key)

    # return joined string components
    return elem_spacing.join(str_parts)

def _get_element_composition(formula: Formula) -> Dict[Element, Union[int,float,None]]:
    comp = {}
    for (elem, count) in formula:
        if isinstance(elem, list):
            elem_comp = _get_element_composition(elem)
            for k, v in elem_comp.items():
                if isinstance(count,str) or v is None:
                    comp[k] = None
                elif k not in comp:
                    comp[k] = v*count
                elif comp[k] is not None:
                    comp[k] += v*count
        else:
            if isinstance(count, str):
                comp[elem] = None
            elif elem not in comp:
                comp[elem] = count
            elif comp[elem] is not None:
                comp[elem] += count

    return comp

def _to_pristine_formula(formula: Formula, method: str='round', 
                         with_variables: bool =False, 
                         max_denom: int = 120,
                         used_variables: Optional[List[str]] = None):
    if used_variables is None:
        used_variables = []

    fractions = {}
    for i, (elem, count) in enumerate(formula):
        if isinstance(elem, list):
            _to_pristine_formula(elem, method, with_variables, max_denom, used_variables)
        if isinstance(count, str):
            continue #TODO: handle rounding of fractions appearing in the counts
        elif isinstance(count, float) and (count - int(count)) > 1e-5:
            fractions[i] = count
    
    unused_variables = sorted(list(set(
            chr(ord('a')+i) for i in range(26)) - set(used_variables)
        ),reverse=True)

    if method == 'round':
        # atttempt to group pairs of the fractions that add up to integers:
        RESIDUE_CUTOFF = 0.09
        pair_residues = [
            ((v1 + v2) - int(v1 + v2), i1, i2)
            for (i1, v1) in fractions.items()
            for (i2, v2) in fractions.items()
            if i1 < i2
        ]

        # sort fraction counts by floor residue, then by proximity in formula:
        pair_residues.sort(key=lambda x : (x[0],abs(x[2]-x[1])))
        matched_indices = set()

        # find the stoichiometrically paired elements:
        for (res, i1, i2) in pair_residues:
            if res < RESIDUE_CUTOFF and i1 not in matched_indices and \
               i2 not in matched_indices:

                matched_indices.update([i1,i2])
                i_min, i_max = i1, i2
                if fractions[i2] < fractions[i1]:
                    i_min, i_max = i2, i1

                elem_min, count_min = formula[i_min]
                elem_max, count_max = formula[i_max]
                count_max_ceil = int(np.ceil(count_max))
                count_min_floor = int(np.floor(count_min))
                if with_variables:
                    new_variable = unused_variables.pop()
                    str_count_min = f'{count_min_floor}+{new_variable}' \
                                        if count_min_floor != 0 else new_variable
                    str_count_max = f'{count_max_ceil}-{new_variable}' \
                                        if count_max_ceil != 0 else f'-{new_variable}'
                    formula[i_min] = (elem_min, str_count_min)
                    formula[i_max] = (elem_max, str_count_max)
                else:
                    formula[i_min] = (elem_min, count_min_floor)
                    formula[i_max] = (elem_max, count_max_ceil)
                
        # round the remaining values:
        for i, fract in fractions.items():
            if i not in matched_indices:
                elem, count = formula[i]
                if with_variables:
                    new_variable = unused_variables.pop()
                    if fract <= 0.5:
                        count_floor = int(np.floor(count))
                        new_count = f'{count_floor}+{new_variable}' if count_floor != 0 \
                                     else new_variable
                    else:
                        count_ceil = int(np.ceil(count))
                        new_count = f'{count_ceil}-{new_variable}' if count_ceil != 0 \
                                     else new_variable

                    formula[i] = (elem, new_count)
                else:
                    formula[i] = (elem, int(np.round(count)))

class Material:
    
    formula: Formula
    variables: List[str]
    variable_elements: List[Element]

    def __init__(self,
                 formula: Union[str,Formula],
                 variable_elements: Union[ElementList,None]=None):
        if isinstance(formula, str):
            self.formula = parse_formula(formula, variable_elements)
        else:
            self.formula = formula
        self.variables, self.variable_elements = _parse_formula_variables(self.formula)

    def copy(self):
        return Material(deepcopy(self.formula))

    def substitute(self, variable: Union[str, Dict[str,FormulaValue]], value: Optional[FormulaValue] = None) -> 'Material':

        # perform substitutions:
        if isinstance(variable, str):
             _substitute_formula_variable(self.formula, variable,value)
        elif isinstance(variable, dict):
            for k,v in variable.items():
                _substitute_formula_variable(self.formula, k,v)

        # parse free variables and elements again:
        self.variables, self.variable_elements = _parse_formula_variables(self.formula)

        return self

    def squeeze(self) -> 'Material':

        # squeeze formula:
        self.formula = squeeze_formula(self.formula)

        # parse free variables and elements again:
        self.variables, self.variable_elements = _parse_formula_variables(self.formula)

        return self

    def substitute_element(self,
                           variable: Union[str,Element,ElementSubstitutionDict], 
                           value: Union[str,Element]) -> 'Material':
        
        # perform substitutions:
        if isinstance(variable, str):
             _substitute_formula_element(self.formula, variable,value)
        elif isinstance(variable, dict):
            for k,v in variable.items():
                _substitute_formula_element(self.formula, k,v)
        
        # parse free variables and elements again:
        self.variables, self.variable_elements = _parse_formula_variables(self.formula)

        return self


    def str(self, squeezed: bool=False, pretty: bool=True, 
            fmt: Optional[str]=None) -> str:
        return self.get_formula_string(squeezed, pretty, fmt)

    def get_formula_string(self, squeezed: bool=False, pretty: bool=True, 
                           fmt: Optional[str]=None) -> str:
        f_squeezed = squeeze_formula(self.formula) if squeezed else self.formula
        sub_start = sub_end = None
        spacing = None
        sort_key = None

        if fmt:
            pretty=False
            if fmt.lower() == 'latex':
                # use Latex format:
                sub_start = r'$_{'
                sub_end = r'}$'
                spacing = None
                sort_key = None
            elif fmt.lower() == 'cod':
                sub_start = ' '
                sub_end = None
                spacing = ' '
                sort_key =lambda e : e.symbol if isinstance(e,Element) else '~'
        
        return _formula_to_string(f_squeezed,
                                  pretty=pretty,
                                  subscript_begin=sub_start,
                                  subscript_end=sub_end,
                                  elem_spacing=spacing,
                                  sort_key=sort_key)
    
    def get_composition(self) -> Dict[Element, Union[int,float,None]]:
        return _get_element_composition(self.formula)

    def matches_composition(self, other: Union[str,'Material']) -> bool:

        # ensure other is a material and get its composition:
        if isinstance(other, str):
            other = Material(other)
        other_comp = other.get_composition()
        self_comp = self.get_composition()
        union_elements = set([ e.symbol for e in self_comp.keys()  ] + \
                             [ e.symbol for e in other_comp.keys() ])
        
        for elem in union_elements:
            if elem not in self_comp or elem not in other_comp:
                return False
            if self_comp[elem] and other_comp[elem] \
                and self_comp[elem] != other_comp[elem]:
                return False
        return True

    def get_pristine_material(self, method: str='round', with_variables: bool =False, max_denom: int = 120) \
            -> Optional['Material']:
        
        # generate new formula with all variables substituted to zero (optional):
        new_formula = deepcopy(self.formula)
        vars, var_elems = _parse_formula_variables(new_formula)
        if not with_variables:
            for v in vars:
                _substitute_formula_variable(new_formula, v, 0)

        # determine pristine material based on method ('round' or 'lcm')
        _to_pristine_formula(new_formula, method, with_variables, max_denom)
        new_formula = squeeze_formula(new_formula)

        return Material(new_formula) if new_formula else None

    def _repr_latex_(self):
        return self.get_formula_string(self,pretty=False, fmt='latex')