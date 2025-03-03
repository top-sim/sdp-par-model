"""This file contains methods for programmatically evaluating the SKA SDP
parametric model.
"""

from __future__ import print_function  # Makes Python-3 style print() function available in Python 2.x

import warnings

import numpy as np
from scipy import optimize as opt
from sympy import simplify, lambdify, Max, Lambda, FiniteSet, Function, Expr, Symbol
import math
import traceback
import itertools

from .parameters.definitions import Telescopes, Pipelines, Bands, Constants as c
from .parameters.container import ParameterContainer, BLDep

def is_literal(expression):
    """
    Returns true iff the expression is already a literal (e.g. float
    or integer) value that cannot be substituted or evaluated
    further. Used to halt attempts at further evaluating symbolic
    expressions
    """
    return isinstance(expression, (str, float, int, np.ndarray, list))


def evaluate_expression(expression, tp):
    """
    Evaluate an expression by substituting the telescopec parameters
    into them. Depending on the type of expression, the result might
    be a value, a list of values (in case it is baseline-dependent) or
    a string (if evaluation failed).

    :param expression: the expression, expressed as a function of the telescope parameters, Tsnap and Nfacet
    :param tp: the telescope parameters (ParameterContainer object containing all relevant parameters)
    :return:
    """

    # Already a plain value?
    if is_literal(expression):  # Only evaluate symbolically when required
        return expression

    # Dictionary? Recurse
    if isinstance(expression, dict):
        return { k: evaluate_expression(e, tp)
                 for k, e in expression.items() }

    # Otherwise try to evaluate using sympy
    try:

        # Baseline dependent?
        if isinstance(expression, BLDep):
            return [ float(expression(**subs)) for subs in tp.baseline_bins ]
        else:
            # Otherwise just evaluate directly
            return float(expression)

    except Exception as e:
        traceback.print_exc()
        warnings.warn("Failed to evaluate (%s): %s" % (e, str(expression)))
        return None

def optimize_lambdified_expr(lam, bound_lower, bound_upper):

    # Lower bound cannot be higher than the uppper bound.
    if bound_lower < bound_upper:
        result = opt.minimize_scalar(lam, bounds=(float(bound_lower), float(bound_upper)), method='bounded')
        if not result.success:
            warnings.warn('WARNING! : Was unable to optimize free variable. Using a value of: %f' % result.x)
        # else:
        #     print ('Optimized free variable = %f' % result.x)
        #     pass
        return result.x
    elif bound_lower > bound_upper:
        warnings.warn('Unable to optimize free variable as upper bound %g is lower than lower bound %g.'
                      'Adhering to lower bound.' % (bound_upper, bound_lower))

        return bound_lower
    elif bound_lower == bound_upper:
        return bound_lower
    else:
        raise Exception("Computer says no.")  # This should be impossible

def cheap_lambdify_curry(free_vars, expression):
    """
    Translate sympy expression to an actual Python expression that can
    be evaluated quickly. This is roughly the same as sympy's
    lambdify, with a number of differences:

    1. We only support a subset of functions. Note that sympy's
       list is incomplete as well, and actually has a wrong
       translation rule for "Max".

    2. The return is curried, so for multiple "free_vars" (x, y)
       you will have to call the result as "f(x)(y)" instead of
       "f(x,y)". This means we can easily obtain a function that
       is specialised for a certain value of the outer variable.

    """

    # Do "quick & dirty" translation. This map might need updating
    # when new functions get used in equations.py
    module = {
        'Max': 'max',
        'Min': 'min',
        'ln': 'math.log',
        'log': 'math.log',
        'sqrt': 'math.sqrt',
        'sign': 'np.sign', # No sign in math, apparently
        'Abs': 'abs',
        'sin': 'math.sin',
        'cos': 'math.cos',
        'acos': 'math.acos',
        'acosh': 'math.acosh',
        'arg': 'np.angle',
        'asin': 'math.asin',
        'asinh': 'math.asinh',
        'atan': 'math.atan',
        'atan2': 'math.atan2',
        'atanh': 'math.atanh',
        'ceiling': 'math.ceil',
        'floor': 'math.floor'
    }
    expr_body = str(expression)
    for (sympy_name, numpy_name) in module.items():
        expr_body = expr_body.replace(sympy_name + '(', numpy_name + '(')

    # Create head of lambda expression
    expr_head = ''
    for free_var in free_vars:
        expr_head += 'lambda ' + str(free_var) + ':'

    # Evaluate in order to build lambda
    return eval(expr_head + expr_body)


def minimise_parameters(telescope_parameters, expression_string = 'Rflop', expression = None,
                        lower_bound = {}, upper_bound = {}, only_one_minimum = ['Nfacet'], verbose = True):
    """Computes the optimal value for free variables in telescope parameters
    such  that it minimizes the value of an expression (typically
    Rflop). Returns result as a dictionary.

    :param telescope_parameters: Contains the definition of the
      expression that needs to be minimzed. This should be a
      symbolic expression that involves Deltaw_Stack and/or Nfacet.
    :param expression_string: The expression that should be minimized.
            This is typically assumed to be the computational load, but may also be, for example, buffer size.
    :param expression: TODO define
    :param lower_bound: Lower bound to use for symbols
    :param upper_bound: Upper bound to use for symbols
    :param only_one_minimum: Assume that the given (integer) symbol
      only has one minimum, so we can assume that any local optimum we
      find is the global optimum.
    :param verbose: Be more chatty about what's going on.
    """
    assert isinstance(telescope_parameters, ParameterContainer)

    # Find symbols to optimise
    if expression is None:
        expression = telescope_parameters.get(expression_string)
        assert expression is not None, "Telescope parameters do not define %s!" % expression_string
    else:
        expression_string = str(expression)
    free_symbols = expression.free_symbols
    free_symbol_names = [ str(sym) for sym in free_symbols ]

    # Default bounds
    lower_bound = dict(lower_bound)
    upper_bound = dict(upper_bound)
    if 'Nfacet' in free_symbol_names:
        if 'Nfacet' not in lower_bound: lower_bound['Nfacet'] = 1
        if 'Nfacet' not in upper_bound: upper_bound['Nfacet'] = 20
    if 'Tsnap' in free_symbol_names:
        if 'Tsnap' not in lower_bound:
            lower_bound['Tsnap'] = telescope_parameters.get('Tsnap_min', 0.1, warn=False)
        if 'Tsnap' not in upper_bound:
            upper_bound['Tsnap'] = max(telescope_parameters.get('Tsnap_min', 0.1, warn=False),
                                       0.5 * telescope_parameters.get('Tobs', 21600, warn=False))
    if 'DeltaW_stack' in free_symbol_names:
        if 'DeltaW_stack' not in lower_bound:
            lower_bound['DeltaW_stack'] = 0.1
        if 'DeltaW_stack' not in upper_bound:
            upper_bound['DeltaW_stack'] = telescope_parameters.DeltaW_max(telescope_parameters.Bmax)

    # We can only optimise one floating-point variable, and every symbol must have bounds
    float_symbol = None
    int_symbols = []
    int_ranges = []
    for sym in free_symbols:
        assert str(sym) in lower_bound, "Symbol %s must have a lower bound!" % sym
        assert str(sym) in upper_bound, "Symbol %s must have an upper bound!" % sym
        if not sym.is_integer:
            assert float_symbol is None, "Can only optimise for one float symbol at a time!"
            float_symbol = sym
            float_lower_bound = lower_bound[str(float_symbol)]
            float_upper_bound = upper_bound[str(float_symbol)]
        else:
            rnge = range(lower_bound[str(sym)], upper_bound[str(sym)]+1)
            # Sort symbols with only one minimum to the back to the list
            if str(sym) in only_one_minimum:
                int_symbols.append(sym)
                int_ranges.append(rnge)
            else:
                int_symbols.insert(0, sym)
                int_ranges.insert(0, rnge)

    # Construct lambda from integer parameters, then float parameter
    params = list(int_symbols)
    if float_symbol is not None:
        params.append(float_symbol)
    expression_lam = cheap_lambdify_curry(params, expression)

    # Loop over the different integer values
    results = []
    skip_until = tuple()
    last_result = None
    result_got_worse = 0
    for int_vals in itertools.product(*int_ranges):
        if int_vals < skip_until:
            continue
        if verbose:
            print('Evaluating', ', '.join(["%s=%d" % vi for vi in zip(int_symbols, int_vals)]), end='')

        # Set integer symbols
        expr = expression_lam
        #TODO Peter: what does the forloop below accomplish?
        for i in int_vals:
            expr = expr(i)

        # Optimise float symbol
        if float_symbol is None:
            opt_val = 0
            result = float(expr)
            if verbose:
                print(' -> %s = %g' % (expression_string, result))
        else:
            opt_val = optimize_lambdified_expr(expr, float_lower_bound, float_upper_bound)
            result = float(expr(opt_val))
            if verbose:
                print(' -> %s=%g, %s=%g' % (float_symbol, opt_val, expression_string, result))

        results.append((result,) + int_vals + (opt_val,))

        # Check whether we can start skipping values (naive, can likely do better)
        if last_result is None:
            pass
        elif last_result >= result:
            result_got_worse = 0
        elif str(int_symbols[-1]) in only_one_minimum:

            # Allow this a few times before we actually stop
            result_got_worse += 1
            if result_got_worse >= 2:
                if len(int_vals) == 1:
                    break

                # I.e. if we are at (1,2,3), skip until (1,3)
                skip_until = int_vals[:-1]
                skip_until[-1] += 1
        last_result = result

    # Return parameters with lowest value
    result, *vals = results[np.argmin(np.array(results)[:,0])]
    if verbose:
        print(', '.join(["%s=%g" % vi for vi in zip(params, vals)]),
              ' yielded the lowest value of %s=%g' % (expression_string, result))
    return dict(zip([str(p) for p in params],vals))

def evaluate_expressions(expressions, tp):
    """
    Evaluate a sequence of expressions by substituting the telescope_parameters into them.

    :param expressions: An array of expressions to be evaluated
    :param tp: The set of telescope parameters that should be used to evaluate each expression
    """
    return [ evaluate_expression(expr, tp) for expr in expressions ]

def collect_free_symbols(formulas):
    """
    Returns the names of all free symbol in the given formulas. We
    always count all functions as free.

    :param formulas: Formulas to search for free symbols.
    """

    def free_f(expr):
        if isinstance(expr, BLDep):
            expr = expr.term
        if not isinstance(expr, Expr):
            return set()
        functions = set(map(lambda f: str(f.func), expr.atoms(Function)))
        frees = set(map(lambda s: str(s), expr.free_symbols))
        return set(frees).union(functions)
    return set().union(*list(map(free_f, formulas)))
