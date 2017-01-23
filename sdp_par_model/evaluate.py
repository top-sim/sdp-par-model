"""This file contains methods for programmatically evaluating the SKA SDP
parametric model.
"""

from __future__ import print_function

from .parameters.definitions import Telescopes, Pipelines, Bands
from .parameters.container import BLDep
import numpy as np
from sympy import Lambda, FiniteSet, Function, Expr, Symbol
import warnings


def is_literal(expression):
    """
    Returns true iff the expression is already a literal (e.g. float or integer) value that cannot be substituted
    or evaluated further. Used to halt attempts at further evaluating symbolic expressions
    """
    return isinstance(expression, (str, float, int, np.ndarray, list))


def evaluate_expression(expression, tp, tsnap, nfacet, key=None):
    # TODO check all calls ans see whether "key" can be added, or whether it should be removed altogether
    """Evaluate an expression by substituting the telescopec parameters
    into them. Depending on the type of expression, the result
    might be a value, a list of values (in case it is baseline-dependent) or a string (if evaluation failed).

    @param expression: the expression, expressed as a function of the telescope parameters, Tsnap and Nfacet
    @param tp: the telescope parameters (ParameterContainer object containing all relevant parameters)
    @param tsnap: The snapshot time to use
    @param nfacet: The number of facets to use
    @param key: (optional) the string name of the expression that is being evaluated. Used in error reporting.
    @return:

    """

    # Already a plain value?
    if is_literal(expression):  # Only evaluate symbolically when required
        return expression

    # Dictionary? Recurse
    if isinstance(expression, dict):
        return { k: evaluate_expression(e, tp, tsnap, nfacet)
                 for k, e in expression.items() }

    # Otherwise try to evaluate using sympy
    try:

        # First substitute parameters
        expression_subst = expression.subs({tp.Tsnap: tsnap, tp.Nfacet: nfacet})

        # Baseline dependent?
        if isinstance(expression_subst, BLDep):
            result = [ float(expression_subst(b=bmax, bcount=tp.Nbl_full*frac_val))
                       for frac_val,bmax in zip(tp.frac_bins, tp.Bmax_bins) ]
        else:
            # Otherwise just evaluate directly
            result = float(expression_subst)

    except Exception as e:
        result = "Failed to evaluate (%s): %s" % (e, str(expression))

    return result

def optimize_lambdified_expr(lam, bound_lower, bound_upper):

    # Lower bound cannot be higher than the uppper bound.
    if bound_lower < bound_upper:
        result = opt.minimize_scalar(lam,
                                     bounds=(bound_lower, bound_upper),
                                     method='bounded')
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
    """Translate sympy expression to an actual Python expression that can
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


def find_optimal_Tsnap_Nfacet(telescope_parameters, expr_to_minimize_string='Rflop',
                              max_number_nfacets=20, min_number_nfacets=1,
                              verbose=False):
    """Computes the optimal value for Tsnap and Nfacet that minimizes the
    value of an expression (typically Rflop). Returns result as a
    2-tuple (Tsnap_opt, Nfacet_opt)

    @param telescope_parameters: Contains the definition of the
      expression that needs to be minimzed. This should be a
      symbolic expression that involves Tsnap and/or Nfacet.
    @param expr_to_minimize_string: The expression that should be
      minimized. This is typically assumed to be the computational
      load, but may also be, for example, buffer size.
    @param max_number_nfacets: Provides an upper limit to
      Nfacet. Because we currently do a linear search for the
      minimum value, using a for loop, we need to know when to
      quit. Max should never be reached unless in pathological
      cases
    @param verbose:

    """
    assert isinstance(telescope_parameters, ParameterContainer)
    assert hasattr(telescope_parameters, expr_to_minimize_string)

    if telescope_parameters.pipeline not in Pipelines.imaging: # Not imaging, return defaults
        if verbose:
            print(telescope_parameters.pipeline, "not imaging - no need to optimise Tsnap and Nfacet")
        return (telescope_parameters.Tobs, 1)

    # Construct lambda from our two parameters (facet number and
    # snapshot time) to the expression to minimise
    expression_original = eval('telescope_parameters.%s' % expr_to_minimize_string)
    params = []
    if isinstance(telescope_parameters.Nfacet, Symbol):
        params.append(telescope_parameters.Nfacet)
        nfacet_range = range(min_number_nfacets, max_number_nfacets+1)
    else:
        nfacet_range = [telescope_parameters.Nfacet]
    if isinstance(telescope_parameters.Tsnap, Symbol):
        params.append(telescope_parameters.Tsnap)
    expression_lam = cheap_lambdify_curry(params, expression_original)

    # Loop over the different integer values of NFacet
    results = []
    warned = False
    for nfacets in nfacet_range:
        # Warn if large values of nfacets are reached, as it may indicate an error and take long!
        if (nfacets > 20) and not warned:
            warnings.warn('Searching minimum value by incrementing Nfacet; value of 20 exceeded... this is odd '
                          '(search may take a long time; will self-terminate at Nfacet = %d' % max_number_nfacets)
            warned = True

        if verbose:
            print ('Evaluating Nfacets = %d' % nfacets)

        # Find optimal Tsnap for this number of facets, obtaining result in "result"
        if isinstance(telescope_parameters.Nfacet, Symbol):
            expr = expression_lam(nfacets)
        else:
            expr = expression_lam
        if isinstance(telescope_parameters.Tsnap, Symbol):
            result = minimize_by_Tsnap_lambdified(expr,
                                                  telescope_parameters,
                                                  verbose=verbose)
        else:
            result = (telescope_parameters.Tsnap, float(expr))
        results.append((nfacets, result[0], result[1]))

        # Continue to at least Nfacet==3 as there can be a local
        # increase between nfacet=1 and 2
        if len(results) >= 3:
            if results[-1][2] >= results[-2][2]:
                if verbose:
                    print ('\nExpression increasing with number of facets; aborting exploration of Nfacets > %d' \
                          % nfacets)
                break

    # Return parameters with lowest value
    nfacets, tsnap, val = results[np.argmin(np.array(results)[:,2])]
    if verbose:
        print ('\n(Nfacet, Tsnap) = (%d, %.2f) yielded the lowest value of %s = %g'
               % (nfacets, tsnap, expr_to_minimize_string, val))
    return (tsnap, nfacets)


def minimize_by_Tsnap_lambdified(lam, telescope_parameters, verbose=False):
    """
    The supplied lambda expression (a function of Tnsap) is minimized.
    @param lam: The lambda expression (a function of Tsnap)
    @param telescope_parameters: The telescope parameters
    @param verbose:
    @return: The optimal Tsnap value, along with the optimal value (as a pair)
    """
    assert telescope_parameters.pipeline in Pipelines.imaging

    # Compute lower & upper bounds
    bound_lower = telescope_parameters.Tsnap_min
    bound_upper = max(bound_lower, 0.5 * telescope_parameters.Tobs)

    # Do optimisation
    Tsnap_optimal = optimize_lambdified_expr(lam, bound_lower, bound_upper)
    value_optimal = lam(Tsnap_optimal)
    if verbose:
        print ("Tsnap has been optimized as : %f. (Cost function = %f)" % \
              (Tsnap_optimal, value_optimal / c.peta))
    return (Tsnap_optimal, value_optimal)  # Replace Tsnap with optimal value


def evaluate_expressions(expressions, tp, tsnap, nfacet):
    """
    Evaluate a sequence of expressions by substituting the telescope_parameters into them. Returns the result
    @param expressions: An array of expressions to be evaluated
    @param tp: The set of telescope parameters that should be used to evaluate each expression
    @param tsnap: The relevant (typically optimal) snapshot time
    @param nfacet: The relevant (typically optimal) number of facets to use
    """
    results = []
    for i in range(len(expressions)):
        expression = expressions[i]
        result = evaluate_expression(expression, tp, tsnap, nfacet)
        results.append(result)
    return results


def eval_products_symbolic(pipelineConfig, expression='Rflop', symbolify='product'):
    """
    Returns formulas for the given product property.
    @param pipelineConfig: Pipeline configuration to use.
    @param expression: Product property to query. FLOP rate by default.
    @param symbolify: How aggressively sub-formulas should be replaced by symbols.
    """

    # Create symbol-ified telescope model
    tp = pipelineConfig.calc_tel_params(symbolify=symbolify)

    # Collect equations and free variables
    eqs = {}
    for product in tp.products:
        eqs[product] = tp.products[product].get(expression, 0)
    return eqs


def eval_symbols(pipelineConfig, symbols,
                 recursive=False, symbolify='', optimize_expression=None):
    """Returns formulas for the given symbol names. This can be used to
    look up the definitions behind sympy Symbols returned by
    eval_products_symbolic or this function.

    The returned dictionary will contain an entry for all symbols
    that we could look up sucessfully - this excludes symbols that
    are not defined or have only a tautological definition ("sym =
    sym").

    @param pipelineConfig: Pipeline configuration to use.
    @param symbols: Symbols to query
    @param recursive: Look up free symbols in symbol definitions?
    @param symbolify: How aggressively sub-formulas should be replaced by symbols.
    """

    # Create possibly symbol-ified telescope model
    tp = pipelineConfig.calc_tel_params(symbolify=symbolify)

    # Optimise to settle Tsnap and Nfacet
    if not optimize_expression is None:
        assert(symbolify == '') # Will likely fail otherwise
        (tsnap_opt, nfacet_opt) = find_optimal_Tsnap_Nfacet(tp, expr_to_minimize_string=optimize_expression)
        tp = pipelineConfig.calc_tel_params(adjusts={'Tsnap': tsnap_opt, 'Nfacet': nfacet_opt})

    # Create lookup map for symbols
    symMap = {}
    for name, v in tp.__dict__.items():
        symMap[tp.make_symbol_name(name)] = v

    # Start collecting equations
    eqs = {}
    while len(symbols) > 0:
        new_symbols = set()
        for sym in symbols:
            if sym in eqs: continue

            # Look up
            if not sym in symMap: continue
            v = symMap[str(sym)]

            # If the equation is "name = name", it is not defined at this level. Push back to next level
            if isinstance(v, Symbol) and str(v) == sym:
                continue
            eqs[str(sym)] = v
            if isinstance(v, Expr) or isinstance(v, BLDep):
                new_symbols = new_symbols.union(collect_free_symbols([v]))
        symbols = new_symbols
    return eqs


def collect_free_symbols(formulas):
    """
    Returns the names of all free symbol in the given formulas. We
    always count all functions as free.

    @param formulas: Formulas to search for free symbols.
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
