"""
This class contains design equations that are used to estimate the SDP's performance, cost and size.
Values and equations are derived from PDR05 (version 1.85).
"""
from parameter_definitions import ParameterContainer
from sympy import symbols

class DesignEquations:
    def __init__(self):
        pass

    @staticmethod
    def define_symbolic_variables(o):
        """
        This method defines the *symbolic* variables that we will use during computations
        and that may need to be kept symbolic during evaluation. One reason to do this would be to allow
        the output formula to be optimized by varying these variables
        @param o: A supplied ParameterContainer object, to which the symbolic variables are appended (in-place)
        @rtype : ParameterContainer
        """
        assert isinstance(o, ParameterContainer)

        o.NfOut = symbols("N_f\,out", integer=True, positive=True)
        o.Nmajor = symbols("N_major", integer=True, positive=True)
        o.Nfacet = symbols("N_facet", integer=True, positive=True)
        o.Ncu = symbols("N_cu", integer=True, positive=True)  # number of compute units.
        o.RcuFLOP = symbols("R_cu\,FLOP", positive=True)  # peak FLOP capability of the compute unit
        o.RcuInter = symbols("R_cu\,inter", positive=True)  # maximum bandwidth of interconnect per Compute Unit
        o.RcuIo = symbols("R_cu\,io", positive=True)  # maximum I/O bandwidth of each compute unit to buffer
        o.Rfft = symbols("R_FFT", positive=True)
        o.Rinterfacet = symbols("R_interfacet", positive=True)
        o.Rrp = symbols("R_RP", positive=True)
        o.MuvGrid = symbols("M_uv\,grid", positive=True)
        o.McuWork = symbols("M_cu\,work", positive=True)  # Size of main working memory of the compute unit
        o.McuPool = symbols("M_cu\,pool", positive=True)  # Size of slower (swap) working memory of the compute unit
        o.McuBuf = symbols("M_cu\,buf", positive=True)  # Size of buffer (or share of data-island local buffer)

        o.RspecFLOP = symbols("R^spec_FLOP", positive=True)
        o.RcontFLOP = symbols("R^cont_FLOP", positive=True)
        o.RfastFLOP = symbols("R^fast_FLOP", positive=True)

        o.RspecIo = symbols("R^spec_io", positive=True)
        o.RcontIo = symbols("R^cont_io", positive=True)

        o.Fci = symbols("F_ci", positive=True)
        o.MspecBufVis = symbols("M^spec_buf\,vis", positive=True)
        o.McontBufVis = symbols("M^cont_buf\,vis", positive=True)
        return o