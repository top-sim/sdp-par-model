"""
This file contains methods for programmatically interacting with the SKA SDP Parametric Model using Python.
"""

from __future__ import print_function

from parameter_definitions import Telescopes, Pipelines, Bands, ParameterDefinitions, BLDep
from equations import Equations
from implementation import Implementation as imp
import numpy as np
from sympy import Lambda, FiniteSet, Function, Expr, Symbol
import warnings

class SkaPythonAPI:
    """
    This class (SKA API) represents an API by which the SKA Parametric Model can be called programmatically, without
    making use of the iPython Notebook infrastructure.
    """

    def __init__(self):
        pass

    @staticmethod
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
        if imp.is_literal(expression):  # Only evaluate symbolically when required
            return expression

        # Dictionary? Recurse
        if isinstance(expression, dict):
            return { k: SkaPythonAPI.evaluate_expression(e, tp, tsnap, nfacet)
                     for k, e in expression.items() }

        # Otherwise try to evaluate using sympy
        try:

            # First substitute parameters
            expression_subst = expression.subs({tp.Tsnap: tsnap, tp.Nfacet: nfacet})

            # Baseline dependent?
            if isinstance(expression_subst, BLDep):
                result = [ float(expression_subst(bmax, tp.Nbl_full*frac_val))
                           for frac_val,bmax in zip(tp.frac_bins, tp.Bmax_bins) ]
            else:
                # Otherwise just evaluate directly
                result = float(expression_subst)

        except Exception as e:
            result = "Failed to evaluate (%s): %s" % (e, str(expression))

        return result

    @staticmethod
    def eval_expression_default(pipelineConfig, expression_string='Rflop', verbose=False):
        """
        Evaluating a parameter for its default parameter value
        @param pipelineConfig:
        @param expression_string:
        @param verbose:
        """

        result = 0
        for pipeline in pipelineConfig.relevant_pipelines:
            pipelineConfig.pipeline = pipeline
            tp = imp.calc_tel_params(pipelineConfig, verbose)

            result_expression = tp.__dict__[expression_string]
            (tsnap_opt, nfacet_opt) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            result += SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap_opt, nfacet_opt)

        return result

    @staticmethod
    def eval_product_default(pipelineConfig, product, expression='Rflop', verbose=False):
        """
        Evaluating a product parameter for its default parameter value
        @param pipelineConfig:
        @param expression:
        @param verbose:
        """

        result = 0
        for pipeline in pipelineConfig.relevant_pipelines:
            pipelineConfig.pipeline = pipeline
            tp = imp.calc_tel_params(pipelineConfig, verbose)

            result_expression = tp.products.get(product, {}).get(expression, 0)
            (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            result += SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap, nfacet)

        return result

    @staticmethod
    def eval_expression_default_products(pipelineConfig, expression='Rflop', verbose=False):
        """
        Evaluating a parameter for its default parameter value
        @param pipelineConfig:
        @param expression:
        @param verbose:
        """

        values={}
        for pipeline in pipelineConfig.relevant_pipelines:
            pipelineConfig.pipeline = pipeline
            tp = imp.calc_tel_params(pipelineConfig, verbose)
            (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)

            # Loop through defined products, add to result
            for name, product in tp.products.items():
                if expression in product:
                    values[name] = values.get(name, 0) + \
                        SkaPythonAPI.evaluate_expression(product[expression], tp, tsnap, nfacet)

        return values

    @staticmethod
    def eval_param_sweep_1d(pipelineConfig, expression_string='Rflop',
                            parameter_string='Rccf', param_val_min=10,
                            param_val_max=10, number_steps=1,
                            verbose=False):
        """
        Evaluates an expression for a range of different parameter values, by varying the parameter linearly in
        a specified range in a number of steps

        @param pipelineConfig:
        @param expression_string: The expression that needs to be evaluated, as string (e.g. "Rflop")
        @param parameter_string: the parameter that will be swept - written as text (e.g. "Bmax")
        @param param_val_min: minimum value for the parameter's value sweep
        @param param_val_max: maximum value for the parameter's value sweep
        @param number_steps: the number of *intervals* that will be used to sweep the parameter from min to max

        @param verbose:
        @return: @raise AssertionError:
        """
        assert param_val_max > param_val_min

        print("Starting sweep of parameter %s, evaluating expression %s over range (%s, %s) in %d steps "
              "(i.e. %d data points)" %
              (parameter_string, expression_string, str(param_val_min), str(param_val_max), number_steps, number_steps + 1))

        param_values = np.linspace(param_val_min, param_val_max, num=number_steps + 1)

        results = []
        for i in range(len(param_values)):
            # Calculate telescope parameter with adjusted parameter
            adjusts = {parameter_string: param_values[i]}
            tp = imp.calc_tel_params(pipelineConfig, verbose, adjusts=adjusts)

            percentage_done = i * 100.0 / len(param_values)
            print("> %.1f%% done: Evaluating %s for %s = %g" % (percentage_done, expression_string,
                                                                parameter_string, param_values[i]))

            # Perform a check to see that the value of the assigned parameter wasn't changed by the imaging equations,
            # otherwise the assigned value would have been lost (i.e. not a free parameter)
            parameter_final_value = tp.__dict__[parameter_string]
            eta = 1e-10
            if abs((parameter_final_value - param_values[i])/param_values[i]) > eta:
                raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                     'by the method compute_derived_parameters(). (%g -> %g). '
                                     'Cannot peform parameter sweep.'
                                     % (parameter_string, param_values[i], parameter_final_value))

            result_expression = tp.__dict__[expression_string]
            (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
            results.append(SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap, nfacet))

        print('done with parameter sweep!')
        return (param_values, results)

    @staticmethod
    def eval_param_sweep_2d(pipelineConfig, expression_string='Rflop', parameters=None, params_ranges=None,
                            number_steps=2, verbose=False):
        """
        Evaluates an expression for a 2D grid of different values for two parameters, by varying each parameter
        linearly in a specified range in a number of steps. Similar to eval_param_sweep_1d, except that it sweeps
        a 2D parameter space, returning a matrix of values.

        @param pipelineConfig
        @param expression_string: The expression that needs to be evalued, as string (e.g. "Rflop")
        @param parameters:
        @param params_ranges:
        @param number_steps:
        @param verbose:
        @return:
        """
        assert (parameters is not None) and (len(parameters) == 2)
        assert (params_ranges is not None) and (len(params_ranges) == 2)
        for prange in params_ranges:
            assert len(prange) == 2
            assert prange[1] > prange[0]

        n_param_x_values = number_steps + 1
        n_param_y_values = number_steps + 1
        nr_evaluations = n_param_x_values * n_param_y_values  # The number of function evaluations that will be required

        print("Evaluating expression %s while\nsweeping parameters %s and %s over 2D domain [%s, %s] x [%s, %s] in %d "
              "steps each,\nfor a total of %d data evaluation points" %
              (expression_string, parameters[0], parameters[1], str(params_ranges[0][0]), str(params_ranges[0][1]),
               str(params_ranges[1][0]), str(params_ranges[1][1]), number_steps, nr_evaluations))

        param_x_values = np.linspace(params_ranges[0][0], params_ranges[0][1], num=n_param_x_values)
        param_y_values = np.linspace(params_ranges[1][0], params_ranges[1][1], num=n_param_y_values)
        results = np.zeros((n_param_x_values, n_param_y_values))  # Create an empty numpy matrix to hold results

        # Nested 2D loop over all values for param1 and param2. Indexes iterate over y (inner loop), then x (outer loop)
        for ix in range(n_param_x_values):
            param_x_value = param_x_values[ix]
            for iy in range(n_param_y_values):
                param_y_value = param_y_values[iy]

                # Overwrite the corresponding fields of tp with the to-be-evaluated values
                adjusts = {
                    parameters[0]: param_x_value,
                    parameters[1]: param_y_value,
                }
                tp = imp.calc_tel_params(pipelineConfig, verbose, adjusts=adjusts)

                percentage_done = (ix * n_param_y_values + iy) * 100.0 / nr_evaluations
                print("> %.1f%% done: Evaluating %s for (%s, %s) = (%s, %s)" % (percentage_done, expression_string,
                                                                                parameters[0], parameters[1],
                                                                                str(param_x_value), str(param_y_value)))

                # Perform a check to see that the value of the assigned parameters weren't changed by the imaging
                # equations, otherwise the assigned values would have been lost (i.e. not free parameters)
                parameter1_final_value = eval('tp.%s' % parameters[0])
                parameter2_final_value = eval('tp.%s' % parameters[1])
                eta = 1e-10
                if abs((parameter1_final_value - param_x_value) / param_x_value) > eta:
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[0])
                if abs((parameter2_final_value - param_y_value) / param_y_value) > eta:
                    print(parameter2_final_value)
                    print(param_y_value)
                    raise AssertionError('Value assigned to %s seems to be overwritten after assignment '
                                         'by the method compute_derived_parameters(). Cannot peform parameter sweep.'
                                         % parameters[1])

                result_expression = tp.__dict__[expression_string]
                (tsnap, nfacet) = imp.find_optimal_Tsnap_Nfacet(tp, verbose=verbose)
                results[iy, ix] = SkaPythonAPI.evaluate_expression(result_expression, tp, tsnap, nfacet)

        print('done with parameter sweep!')
        return (param_x_values, param_y_values, results)

    @staticmethod
    def eval_param_sweep_2d_noopt(pipelineConfig, expression='Rflop',
                                  tsnaps=[100.0], nfacets=[1], verbose=False):
        """Evaluates an expression for a 2D grid of different values for
        snapshot time and facet number, by varying each parameter
        linearly in a specified range in a number of steps.

        """
        n_param_y_values = len(tsnaps)
        n_param_x_values = len(nfacets)
        nr_evaluations = n_param_x_values * n_param_y_values  # The number of function evaluations that will be required

        print("Evaluating expression %s while\nsweeping parameters tsnap and nfacet over 2D domain %s x %s " %
              (expression, str(tsnaps), str(nfacets)))

        # Generate telescope parameters, lambdify target expression
        telescope_params = imp.calc_tel_params(pipelineConfig, verbose=verbose)
        result_expression = telescope_params.__dict__[expression]
        expression_lam = imp.cheap_lambdify_curry((telescope_params.Nfacet,
                                                   telescope_params.Tsnap),
                                                  result_expression)

        # Create an empty numpy matrix to hold results
        results = np.zeros((n_param_y_values, n_param_x_values))


        # Nested 2D loop over all values for param1 and
        # param2. Indexes iterate over y (inner loop), then x (outer
        # loop)
        for iy in range(len(tsnaps)):
            tsnap = tsnaps[iy]
            for ix in range(len(nfacets)):
                nfacet = nfacets[ix]

                if verbose:
                    percentage_done = (ix + iy * n_param_x_values) * 100.0 / nr_evaluations
                    print("> %.1f%% done: Evaluating %s for (tsnap, nfacet) = (%s, %s)" % (percentage_done, expression,
                                                                                 str(tsnap), str(nfacet)))

                results[iy, ix] = expression_lam(nfacet)(tsnap)

        print('Done with parameter sweep!')
        return (tsnaps, nfacets, results)

    @staticmethod
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
            result = SkaPythonAPI.evaluate_expression(expression, tp, tsnap, nfacet)
            results.append(result)
        return results

    @staticmethod
    def eval_products_symbolic(pipelineConfig, expression='Rflop', symbolify='product'):
        """
        Returns formulas for the given product property.
        @param pipelineConfig: Pipeline configuration to use.
        @param expression: Product property to query. FLOP rate by default.
        @param symbolify: How aggressively sub-formulas should be replaced by symbols.
        """

        # Create symbol-ified telescope model
        tp = imp.calc_tel_params(pipelineConfig, symbolify=symbolify)

        # Collect equations and free variables
        eqs = {}
        for product in tp.products:
            eqs[product] = tp.products[product].get(expression, 0)
        return eqs

    @staticmethod
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
        tp = imp.calc_tel_params(pipelineConfig, symbolify=symbolify)

        # Optimise to settle Tsnap and Nfacet
        if not optimize_expression is None:
            assert(symbolify == '') # Will likely fail otherwise
            (tsnap_opt, nfacet_opt) = imp.find_optimal_Tsnap_Nfacet(tp, expr_to_minimize_string=optimize_expression)
            tp = imp.calc_tel_params(pipelineConfig, adjusts={'Tsnap': tsnap_opt, 'Nfacet': nfacet_opt})

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
                    new_symbols = new_symbols.union(SkaPythonAPI.collect_free_symbols([v]))
            symbols = new_symbols
        return eqs

    @staticmethod
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
