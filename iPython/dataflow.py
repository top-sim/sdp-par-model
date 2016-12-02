
from __future__ import print_function

from abc import ABCMeta, abstractmethod

from parameter_definitions import Constants

from sympy import Max, Expr, Lambda, Symbol, Sum, Mul
from graphviz import Digraph
from math import floor
from functools import total_ordering

import unittest

# Special region properties
INDEX_PROP = 'index'
COUNT_PROP = 'count'
SIZE_PROP = 'size'
OFFSET_PROP = 'offset'

# Internal properties. Might not be transparent with enumeration!
SPLIT_PROP = '_split'

def mk_lambda(v):
    if callable(v) and not isinstance(v, Expr):
        return v
    else:
        return (lambda rbox: v)

@total_ordering
class Domain:
    """A domain is an axis that a computation can be distributed over -
    such as time or frequency.
    """

    def __init__(self, name, unit = '', priority=0):
        self.name = name
        self.unit = unit
        self.priority = priority

    def __repr__(self):
        return '<Domain %s>' % self.name

    def __lt__(self, other):
        if self.priority == other.priority:
            return self.name < other.name
        return self.priority > other.priority

    def regions(self, size, props = {}):
        """Make a new unsplit region set for the domain."""
        return Regions(self, size, 1, props)

    def __call__(self, prop):
        return DomainExpr(lambda rb: rb.eval(self, prop))

class DomainExpr:
    """ An expression depending on domain properties. """

    def __init__(self, expr):
        if isinstance(expr, DomainExpr):
            self.expr = expr.expr
        elif callable(expr) and not isinstance(expr, Expr):
            self.expr = expr
        else:
            self.expr = lambda rb: expr

    def eval(self, rb):
        return self.expr(rb)

    # DomainExpr objects can be combined using simple arithmetic
    def __mul__(self, other):
        return DomainExpr(lambda rb: self.eval(rb) * DomainExpr(other).eval(rb))
    def __rmul__(self, other):
        return DomainExpr(lambda rb: DomainExpr(other).eval(rb) * self.eval(rb))
    def __truediv__(self, other):
        return DomainExpr(lambda rb: self.eval(rb) / DomainExpr(other).eval(rb))
    def __rtruediv__(self, other):
        return DomainExpr(lambda rb: DomainExpr(other).eval(rb) / self.eval(rb))
    def __add__(self, other):
        return DomainExpr(lambda rb: self.eval(rb) + DomainExpr(other).eval(rb))
    def __radd__(self, other):
        return DomainExpr(lambda rb: DomainExpr(other).eval(rb) + self.eval(rb))
    def __neg__(self):
        return DomainExpr(lambda rb: -self.eval(rb))

class Regions:
    """A set of regions of a domain."""

    def __init__(self, domain, size, count, props={}):
        """Makes a new named domain that has given total size and is split up
        into the given number of equal-sized pieces."""

        self.domain = domain
        self.size = size
        self._count = DomainExpr(count)
        self.props = props

    def __repr__(self):
        return '<Regions %d x %s>' % (self.size, self.domain.name)

    def split(self, split, props = {}):
        return Regions(self.domain, self.size,
                       self._count * DomainExpr(split),
                       props)

    def count(self, rb):
        return self._count.eval(rb)

    def constantSpacing(self):
        """Returns whether the regions have constant spacing. This allows us
        to reason about these regions cheaper."""

        # No size property? Then we assume constant spacing
        if not SIZE_PROP in self.props:
            return True

        # Otherwise, our size property must convert to a plain number
        # of some sort.
        try:
            float(self.props[SIZE_PROP])
            return True
        except ValueError:
            return False

    def region(self, rb):

        # Properties that do not depend on the region, or only
        # need to know it symbolically (sympy Lambda) can be
        # worked with even if the region stays anonymous.
        reg = {
            k: v
            for k, v in self.props.items()
            if not (callable(v) or isinstance(v, DomainExpr)) or isinstance(v, Lambda)
        }

        # The split degree is one of the few things we know about
        # regions no matter what.
        reg[COUNT_PROP] = self.count(rb)

        # Split degree - starts at zero index and offset, not enumerated
        reg[SPLIT_PROP] = (0, 0, reg[COUNT_PROP])

        # If size does not get defined in self.props, we provide a
        # default implementation.
        if not SIZE_PROP in reg and \
           not SIZE_PROP in self.props and \
           reg[COUNT_PROP] != 0:
            reg[SIZE_PROP] = self.size / reg[COUNT_PROP]

        return reg

    def region_enum(self, rb, index, offset):
        """Get properties of a region of this region set"""

        # Only copy over detailed properties if our domain is
        # enumerated - they are allowed to depend on the concrete
        # region..
        reg = {
            k: DomainExpr(mk_lambda(v)(index)).eval(rb)
            for k, v in self.props.items()
        }

        # Regions are unary now, with an index and offset
        reg[INDEX_PROP] = index
        reg[OFFSET_PROP] = offset
        reg[COUNT_PROP] = self.count(rb)

        # Set split degree accordingly
        reg[SPLIT_PROP] = (index, offset, 1)

        # If size does not get defined in self.props, we provide a
        # default implementation.
        if not SIZE_PROP in reg and \
           not SIZE_PROP in self.props:
           reg[SIZE_PROP] = self.size / self.count(rb)

        return reg

    def regions(self, rbox):
        """Get all regions of this region set."""

        off = 0 # TODO: regions start offset?
        for i in range(int(round(self.count(rbox)))):
            props = self.region_enum(rbox, i, off)
            off += props[SIZE_PROP]
            yield props

    def bounds(self, rbox):
        """Cheaper version of regions that only returns the region bounds."""

        # Constant spacing?
        if self.constantSpacing():
            size = float(self.props.get(SIZE_PROP, self.size / self.count(rbox)))
            for i in range(int(round(self.count(rbox)))):
                yield (floor(i * size), floor(size))
        # Otherwise fall back to enumerating
        else:
            for reg in self.regions(rbox):
                yield (reg[OFFSET_PROP], reg[SIZE_PROP])

    def regionsEqual(self, rbox, regs2, rbox2):
        """Checks whether all regions are equal to another set of regions."""

        # Custom size? Need to compare in detail then...
        if not self.constantSpacing() or not regs2.constantSpacing():
            return list(self.regions(rbox)) == list(regs2.regions(rbox2))

        # Otherwise (try to) compare count + size
        reg = self.region(rbox)
        reg2 = regs2.region(rbox2)
        return reg[COUNT_PROP] == reg2[COUNT_PROP] and reg[SIZE_PROP] == reg2[SIZE_PROP]

    def the(self, prop):
        # Defer to RegionBoxes, which will care about enumerating the
        # regions if required.
        return RegionBoxes([self]).the(prop)

    def sum(self, prop):
        # See above
        return RegionBoxes([self]).sum(prop)

    def max(self, prop):
        # See above
        return RegionBoxes([self]).max(prop)

class DomainPrecendenceException(BaseException):
    """Thrown when the properties of a region are accessed 'before' they
    are defined. This should only happen while RegionBoxes._getBoxes()
    builds up all region boxes, because that is when region boxes
    might not be fully populated yet. This exception means that we
    need to re-order domains in order to be able to determine their
    properties."""
    def __init__(self, rboxes, domain):
        self.regionBoxes = rboxes
        self.domain = domain

class NeedRegionEnumerationException(BaseException):

    """Thrown when we detect that we need concrete region information
    about a given domain. This exception gets thrown by RegionBox
    when that information is requested, and when caught should
    lead to the domain getting added to the RegionBox's
    enumDomains.
    """
    def __init__(self, rboxes, domain):
        self.regionBoxes = rboxes
        self.domain = domain

class RegionBoxBase:

    __metaclass__ = ABCMeta

    @abstractmethod
    def domains(self):
        """Returns the domains this box has regions for."""

    @abstractmethod
    def enumDomains(self):
        """Returns currently enumerated domains. For internal use."""

    @abstractmethod
    def eval(self, domain, prop):
        """
        Queries a domain property of this box. Might raise an
        "NeedRegionEnumerationException" if the requested property is
        not available unless the regions of this domain have to be
        enumerated. Use _withEnums to automatically handle these
        exceptions.
        """

    @abstractmethod
    def regions(self, dom):
        """
        Generates the properties of all regions of the given domain that
        are contained in this region box.
        """

    @abstractmethod
    def bounds(self, dom):
        """
        Returns offset-size pairs for the regions of a domain. Like
        regions(), but cheaper if we are only interested in the bounds.
        """

    @abstractmethod
    def regionsEqual(self, rbox, dom):
        """Check whether we have the same regions for the given domain. This
        is a cheap heuristic: We might return False even though the
        regions are actually equal.
        """

    def the(self, prop):
        """Return the value of the given property for this region box"""
        return DomainExpr(prop).eval(self)

    def count(self):
        count = 1
        # Multiply out all non-enumerated (!) regions
        for dom in self.domains():
            if not dom in self.enumDomains():
                count *= self.eval(dom,COUNT_PROP)
        return count

    def mults(lbox, rbox, verbose):
        """Determines multiplicity of region overlaps between two boxes. This
        will return for every domain and either side:

        * the index of the first overlapping region
        * the total number of overlapping regions

        If there is no overlap, this returns None. If a region does
        not exist on the right side, we return None in the dictionary
        instead of the index/count pair.
        """

        # Check whether we have any overlap
        dims = {}
        for dom in lbox.domains():

            # Get left box properties
            lsize = lbox.eval(dom, SIZE_PROP)
            lix, loff, lcount = lbox.eval(dom, SPLIT_PROP)

            # Domain not existent on the right? Then we assume that
            # all from the left apply.
            if not dom in rbox.domains():
                dims[dom] = ((int(lix), int(lcount)), None)
                continue

            # Get box properties
            rsize = rbox.eval(dom, SIZE_PROP)
            rix, roff, rcount = rbox.eval(dom, SPLIT_PROP)

            if verbose:
                print('%s Left off=%d size=%d index=%d count=%d' %
                       (dom.name, loff, lsize, lix, lcount))
                print('%s Right off=%d size=%d index=%d count=%d' %
                       (dom.name, roff, rsize, rix, rcount))

            # Determine overlap region
            off = max(loff, roff)
            size = min(loff + lsize*lcount, roff + rsize*rcount) - off

            # Size substantially smaller than a region on either side?
            # Ignore it. There's a bunch of rounding issues there, the
            # limit of half a region size is unlikely to trigger them.
            # print (" => off=%d size=%d" % (off, size))
            if size < lsize / 2 and size < rsize / 2:
                return None

            # Identify indices
            from math import ceil
            lix0 = lix + int((off - loff) / lsize)
            lix1 = lix + int(min(lcount, ceil((off + size - loff) / lsize)))
            rix0 = rix + int((off - roff) / rsize)
            rix1 = rix + int(min(rcount, ceil((off + size - roff) / rsize)))

            # Bail out if no overlap. Otherwise add input dimension.
            if rix1 <= rix0:
                return None
            if verbose:
                print (" => Left ", (lix0, lix1-lix0))
                print (" => Right ", (rix0, rix1-rix0))
            dims[dom] = ((lix0, lix1-lix0), (rix0, rix1-rix0))

        return dims

    def sum(self, prop):
        """Return the sum of the given property for this region box"""

        # Apply property to this region box, collect symbols
        self.symbols = {}
        expr = DomainExpr(prop).eval(self)

        # Now multiply out regions where we don't care about the
        # index, but generate a sympy Sum term for when we are
        # referring to the index symbolically.
        #
        # Note that this is a rough solution - there are probably
        # situations with multiple indices where this goes
        # wrong. Especially concerning the order in which we do all
        # this - the "count" property can depend on other domains, so
        # there's no guarantee that we don't end up using undefined
        # symbols here...
        for dom in self.domains():
            if not dom in self.symbols and not dom in self.enumDomains():
                expr = self.eval(dom, COUNT_PROP) * expr
        sums = []
        sumSymbols = []
        for dom in self.domains():
            if dom in self.symbols:
                sumSymbols.append(self.symbols[dom])
                sums.append((self.symbols[dom],
                             0, self.eval(dom, COUNT_PROP) - 1))

        # Do we need to formulate a sum? If not (common case), we can
        # just return "expr" here.
        if len(sums) == 0:
            return expr

        # Factor out independent product terms before formulating
        # the sum. Not sure why sympy doens't do this by default?
        # Probably overlooking something...
        if isinstance(expr, Mul):
            def indep(e): return len(set(sumSymbols).intersection(e.free_symbols)) == 0
            indepFactors = list(filter(indep, expr.as_ordered_factors()))
            if len(indepFactors) > 0:
                def not_indep(e): return not indep(e)
                restFactors = list(filter(not_indep, expr.as_ordered_factors()))
                if len(restFactors) == 0:
                    m = 1
                    for _, low, high in sums:
                        m *= (high - low + 1)
                    return Mul(m, *indepFactors)
                else:
                    return Mul(*indepFactors) * Sum(Mul(*restFactors), *sums)

        return Sum(expr, *sums)

    def allBounds(self):
        return [ self.bounds(dom)
                 for dom in self.domains() ]

    def _edgeMult(lbox, rbox, dom):
        """Returns the edge multiplier between two region boxes concerning a
        domain. This is basically answering the question - if we have
        one domain that is split in two ways, how many edges would we
        have between two sub-sets of these splits?
        """

        # Domain enumerated on both sides? Then there's either one
        # edge or there is none. Check match.
        if dom in lbox.enumDomains() and dom in rbox.enumDomains():
            # Must match so we don't zip both sides
            # all-to-all. If we find that the two boxes don't
            # match, we conclude that this box combination is
            # invalid and bail out completely.
            loff = lbox.eval(dom, OFFSET_PROP)
            roff = rbox.eval(dom, OFFSET_PROP)
            if loff < roff:
                if roff >= loff + lbox.eval(dom, SIZE_PROP):
                    return 0
            else:
                if loff >= roff + rbox.eval(dom, SIZE_PROP):
                    return 0
            return 1
        # Enumerated on just one side? Assuming equal spacing
        # on the other side, calculate how many region starts
        # on the other side fall into the current region.
        if dom in lbox.enumDomains():
            loff = lbox.eval(dom, OFFSET_PROP)
            lsize = lbox.eval(dom, SIZE_PROP)
            rsize = rbox.eval(dom, SIZE_PROP)
            rcount = rbox.eval(dom, COUNT_PROP)
            if loff >= rsize * rcount: return 0
            def bound(x): return min(rcount-1, max(0, int(x)))
            return 1 + bound((loff + lsize - 1) / rsize) - bound(loff / rsize)
        if dom in rbox.enumDomains():
            lsize = lbox.eval(dom, SIZE_PROP)
            lcount = lbox.eval(dom, COUNT_PROP)
            roff = rbox.eval(dom, OFFSET_PROP)
            rsize = rbox.eval(dom, SIZE_PROP)
            if roff >= lsize * lcount: return 0
            def bound(x): return min(lcount-1, max(0, int(x)))
            return 1 + bound((roff + rsize - 1) / lsize) - bound(roff / lsize)
        # Otherwise, the higher granularity gives the number
        # of edges.
        lcount = lbox.eval(dom, COUNT_PROP)
        rcount = rbox.eval(dom, COUNT_PROP)
        lsize = lbox.eval(dom, SIZE_PROP)
        rsize = rbox.eval(dom, SIZE_PROP)
        import math
        if lcount * lsize > rcount * rsize:
            lcount = int((rcount * rsize + lsize - 1) / lsize)
        elif lcount * lsize < rcount * rsize:
            rcount = int((lcount * lsize + rsize - 1) / rsize)
        if lcount < rcount:
            return lcount * int((rcount + lcount - 1) / lcount)
        else:
            return rcount * int((lcount + rcount - 1) / rcount)

    def _zipSum(lbox, rbox, commonDoms, leftDoms, rightDoms, f, verbose):
        """Return sum of function, applied to all pairs of edges between
        individual region boxes between lbox and rbox.
        """

        # Match common domains
        mult = 1
        for dom in commonDoms:
            m = lbox._edgeMult(rbox, dom)
            if verbose: print("common", dom, m)
            mult *= m
            if mult == 0: return 0

        # Domains on only one side: Simply multiply out (where
        # un-enumerated)
        for dom in leftDoms:
            if not dom in lbox.enumDomains():
                mult *= lbox.eval(dom, COUNT_PROP)
                if verbose: print("left", dom, lbox.eval(dom, COUNT_PROP))
        for dom in rightDoms:
            if not dom in rbox.enumDomains():
                mult *= rbox.eval(dom, COUNT_PROP)
                if verbose: print("right", dom, rbox.eval(dom, COUNT_PROP))

        if verbose: print("-> mult:", mult)

        # Okay, now call and multiply
        return mult * f(lbox, rbox)

    def _edgeCrossMult(lbox, rbox, cbox, dom, a = False):
        """Returns the edge cross multiplier between two region boxes
        concerning a domain. This means that we count the number of
        edges between left and right that do not start in the same
        cross region that they end in. An edge that starts within a
        cross region and ends outside of any other cross region
        counts for our purposes.
        """

        # Left and right regions the same?
        if lbox.regionsEqual(rbox, dom):

            # This always means 0 crosses, because all edges are
            # horizontal.
            if not a:
                return 0

            # If cross domain is not enumerated, this is simply the
            # edge count
            if not dom in cbox.enumDomains():
                return lbox._edgeMult(rbox, dom)
        # In contrast to _edgeMult, we must go through all individual
        # regions separately. Slightly less efficient and more work,
        # but on the other hand this means that we do not need to care
        # about enumeration in the first place.

        # Set up region iterators
        lregs = lbox.bounds(dom)
        rregs = rbox.bounds(dom)
        cregs = cbox.bounds(dom)
        try:
            (loff, lsize) = next(lregs)
            (roff, rsize) = next(rregs)
            (coff, csize) = next(cregs)
        except StopIteration:
            return 0

        count = 0
        while True:

            # Advance iterators until there is at least some overlap
            try:
                while loff + lsize < roff:
                    (loff, lsize) = next(lregs)
                while roff + rsize < loff:
                    (roff, rsize) = next(rregs)
                while coff + csize <= loff or coff + csize <= roff:
                    (coff, csize) = next(cregs)
            except StopIteration:
                break

            # Is our current edge crossing?
            if loff < roff:
                if roff < loff + lsize and (a or loff < coff) and roff >= coff:
                    count += 1
            else:
                if loff < roff + rsize and (a or roff < coff) and loff >= coff:
                    count += 1

            # Advance either left or right iterator
            try:
                if loff + lsize < roff + rsize:
                    (loff, lsize) = next(lregs)
                else:
                    (roff, rsize) = next(rregs)
            except StopIteration:
                break

        return count

    def _edgeCrossMultOneSided(lbox, cbox, dom):
        """Count how many edges cross 'out' of the cross boxes. This is used
        when there is no valid end point for the edges, so all edges
        cross. However, we still only want to count edges that come
        from inside "cbox" in case it is enumerated.
        """

        # Set up region iterators
        lregs = lbox.regions(dom)
        cregs = cbox.regions(dom)
        try:
            lreg = next(lregs)
            creg = next(cregs)
        except StopIteration:
            return 0

        count = 0
        loff = lreg[OFFSET_PROP]; lsize = lreg[SIZE_PROP]
        coff = creg[OFFSET_PROP]; csize = creg[SIZE_PROP]
        while True:

            # Advance cross iterator
            try:
                while coff + csize <= loff:
                    creg = next(cregs)
                    coff = creg[OFFSET_PROP]; csize = creg[SIZE_PROP]
            except StopIteration:
                break

            # Is our current edge crossing?
            if loff >= coff and loff < coff + csize:
                count += 1

            # Advance iterator
            try:
                lreg = next(lregs)
                loff = lreg[OFFSET_PROP]; lsize = lreg[SIZE_PROP]
            except StopIteration:
                break

        return count

    def _zipCrossSum(lbox, rbox, cbox, commonDoms, leftDoms, rightDoms, f, verbose):
        """Return sum of function, applied to all pairs of edges between
        individual region boxes between lbox and rbox that start in a
        different region of the cross domain than they end in.
        """

        # Match common domains
        mults = []
        cross_doms = cbox.domains()
        for dom in commonDoms:

            # Determine edge multiplier. This is about determining how
            # many edges will cross if *another* domain has a
            # cross. E.g. if we have two domains with region counts 12
            # and 7, and no crosses in domain 1 but 1 cross in domain
            # 2, we get 12 * 1 crossings total.
            #
            # Note that if the domain itself is a cross domain, we
            # might have to restrict the domain space we count edges
            # in so we are transparent to enumeration. So e.g. if in
            # the above case we have an enumeration of 2 regions of
            # cross boxes in domain 1, _zipCrossSum will get called
            # twice and we want to report 6*1 crossings each.
            if dom in cross_doms:
                m = lbox._edgeCrossMult(rbox, cbox, dom, a=True)
            else:
                m = lbox._edgeMult(rbox, dom)
            # As usual, zero means zero edges, and therefore zero
            # crossing edges.
            if m == 0: return 0

            # Get number of crosses for this domain. If this doesn't
            # appear in the cross domains, it means no edges cross.
            if dom in cross_doms:
                cm = lbox._edgeCrossMult(rbox, cbox, dom)
                if verbose: print("Common cross", dom, ":", (m, cm))
            else:
                cm = 0
                if verbose: print("Common non-cross", dom, ":", (m, cm))
            mults.append((m,cm))

        # Domains on only one side: Edge multiplier is given by region
        # count. Edge cross multiplier depends - if the domain in
        # question is not one we track edge crossings for, we
        # obviously have no such crossings. Otherwise, all edges
        # cross.
        for dom in leftDoms:
            if dom in cross_doms:
                m = cm = lbox._edgeCrossMultOneSided(cbox, dom)
                if verbose: print("Left cross ", dom, ":", (m, cm))
            else:
                if dom in lbox.regionBoxes.enumDomains:
                    m = 1
                else:
                    m = lbox.eval(dom, COUNT_PROP)
                cm = 0
                if verbose: print("Left non-cross ", dom, ":", (m, cm))
            mults.append((m,cm))
        for dom in rightDoms:
            if dom in cross_doms:
                m = cm = rbox._edgeCrossMultOneSided(cbox, dom)
                if verbose: print("Right cross ", dom, ":", (m, cm))
            else:
                if dom in rbox.regionBoxes.enumDomains:
                    m = 1
                else:
                    m = rbox.eval(dom, COUNT_PROP)
                cm = 0
                if verbose: print("Right non-cross ", dom, ":", (m, cm))
            mults.append((m,cm))

        # Multiply out. It is easiest to achieve this by determining
        # the number of edges that do *not* cross, then remove that
        # from the total number of edges. So basically, with A,B,C the
        # edge multipliers and a,b,c the cross multipliers, we calculate:
        #
        #   ABC - (A-a)(B-b)(C-c)
        mult = 1
        cmult = 1
        for (m, cm) in mults:
            mult *= m
            cmult *= (m - cm)

        # Okay, now call and multiply
        return (mult - cmult) * f(lbox, rbox)

    def product(lbox, rbox):
        return RegionBoxProduct(lbox, rbox)

class RegionBox(RegionBoxBase):

    def __init__(self, regionBoxes, domainProps):
        self.regionBoxes = regionBoxes
        self.domainProps = domainProps
        self.symbols = {}

    def domains(self):
        return self.domainProps.keys()

    def enumDomains(self):
        """Returns currently enumerated domains."""
        return self.regionBoxes.enumDomains

    def __repr__(self):
        return '<RegionBox [%s] %s>' % (
            ','.join(map(lambda d:d.name, self.regionBoxes.regionsMap.keys())),
            repr(self.domainProps))

    def eval(self, domain, prop):

        # Must be a domain of our regions box
        assert domain in self.regionBoxes.domains()

        # No properties known yet? This should only happen while we're
        # inside RegionBoxes._getBoxes(). Then it means that this
        # domain needs to be defined before whatever we're doing here.
        if not domain in self.domainProps:
            raise DomainPrecendenceException(self.regionBoxes, domain)

        # If we don't have the property, this might also point towards
        # the need for enumeration.
        if not prop in self.domainProps[domain] and \
           not domain in self.regionBoxes.enumDomains:
            raise NeedRegionEnumerationException(self.regionBoxes, domain)

        # Not there?
        if not prop in self.domainProps[domain]:
            raise Exception('Property %s is undefined for this region set of domain %s!'
                             % (prop, domain.name))

        # Look it up
        val = self.domainProps[domain][prop]

        # Symbolised? Pass region index as symbol. Otherwise just return as-is.
        if isinstance(val, Lambda):
            if not domain in self.symbols:
                self.symbols[domain] = Symbol("i_" + domain.name)
            return val(self.symbols[domain])
        return val

    def regions(self, dom):
        """Generates the properties of all regions of the given domain that
        are contained in this region box."""
        assert dom in self.domains()
        if dom in self.enumDomains():
            return iter([self.domainProps[dom]])
        else:
            return self.regionBoxes.regionsMap[dom].regions(self)

    def bounds(self, dom):
        if dom in self.regionBoxes.enumDomains:
            return iter([(self.domainProps[dom][OFFSET_PROP],
                          self.domainProps[dom][SIZE_PROP])])
        else:
            return self.regionBoxes.regionsMap[dom].bounds(self)

    def regionsEqual(self, rbox, dom):
        """Check whether we have the same regions for the given domain. This
        is a cheap heuristic: We might return False even though the
        regions are actually equal.
        """
        assert dom in self.domainProps and dom in rbox.domainProps
        if dom in self.regionBoxes.enumDomains:
            # Only one to compare - fall back to direct comparison
            return [self.domainProps[dom]] == list(rbox.regions(dom))
        else:
            return self.regionBoxes.regionsMap[dom].regionsEqual(
                self, rbox.regionBoxes.regionsMap[dom], rbox)


class RegionBoxProduct(RegionBoxBase):

    def __init__(self, rbox1, rbox2):
        assert len(set(rbox1.domains()).intersection(rbox2.domains())) == 0
        self.rbox1 = rbox1
        self.rbox2 = rbox2

    def __repr__(self):
        return '<RegionBoxProduct %s x %s>' % (self.rbox1, self.rbox2)

    def __call__(self, domain, prop):

        # Must be a domain of either region box
        assert domain in self.domains()

        # Forward
        if domain in self.rbox1.domains():
            return self.rbox1(domain, prop)
        else:
            return self.rbox2(domain, prop)

    def domains(self):
        return set(self.rbox1.domains()).union(self.rbox2.domains())

    def enumDomains(self):
        return set(self.rbox1.enumDomains()).union(self.rbox2.enumDomains())

    def regions(self):
        import itertools
        return itertools.product(self.rbox1.regions(), self.rbox2.regions())

    def bounds(self, dom):
        assert dom in self.domains()
        if dom in self.rbox1.domains():
            return self.rbox1.bounds(dom)
        else:
            return self.rbox2.bounds(dom)

    def regionsEqual(self, rbox, dom):
        assert dom in self.domains()
        if dom in self.rbox1.regionsEqual(rbox, dom):
            return self.rbox1.regionsEqual(rbox, dom)
        else:
            return self.rbox2.regionsEqual(rbox, dom)

class RegionBoxes:
    """A set of region boxes. Conceptually, this is the cartesian product
    of all involved regions, the full list of which might be very,
    very large. To get around this, we assume that most regions are
    split into equal pieces, which we can reason about without
    enumerating every single one. This generally works as long as we
    are only interested in their count or size, but never ask for
    anything more specific.
    """

    def __init__(self, regionsBox):
        self.regionsBox = regionsBox

        # Set up regions map. Every domain is only allowed to appear once
        self.regionsMap = {}
        for regions in regionsBox:
            if regions.domain in self.regionsMap:
                raise Exception('Domain %s has two regions in the same '
                                'box, this is not allowed!' % regions.domain.name)
            self.regionsMap[regions.domain] = regions

        # Try to construct a list with all region boxes we need to
        # enumerate. This will add enumDomains as a side-effect
        self.enumDomains = []
        self.priorityDomains = []
        self._genBoxes()

    def _genBoxes(self):
        self.boxes = self._withEnums(lambda: self._getBoxes(),
                                     regenerateBoxes=False)

    def _withEnums(self, code, regenerateBoxes=True):
        """Runs some code, catching NeedRegionEnumerationException and
        enumerating appropriately. """

        # Might need to start over because additional regions need to
        # be enumerated.
        try:
            return code()

        except DomainPrecendenceException as e:

            # Rethrow if this is for a different region boxes
            if e.regionBoxes != self:
                raise e

            # Sanity-check
            if e.domain in self.priorityDomains:
                raise Exception('Domain %s already prioritised - domain dependency cycle?' % e.domain.name)

            # Add domain to be prioritised
            self.priorityDomains.insert(0,e.domain)
            return self._withEnums(code)

        except NeedRegionEnumerationException as e:

            # Rethrow if this is for a different region boxes
            if e.regionBoxes != self:
                raise e

            # Sanity-check the domain we are supposed to enumerate
            if not e.domain in self.regionsMap:
                raise Exception('Domain %s not in region box - can\'t depend on it.' % e.domain.name)
            if e.domain in self.enumDomains:
                raise Exception('Domain %s already enumerated - domain dependency cycle?' % e.domain.name)

            # Add domain to be enumerated.
            self.enumDomains.insert(0,e.domain)
            if regenerateBoxes: self._genBoxes()

        # Start over
        return self._withEnums(code)

    def _getBoxes(self):
        """Generates all region boxes, determining all their properties as we
        go along. NeedRegionEnumerationException might get thrown if
        not all properties can be determined without enumerating the
        given domain.
        """

        # Start with just the single global region box, with
        # no known region properties.
        rboxProps = [{}]
        for edom in self.enumDomains:

            # Multiply out all region boxes with all regions
            # for this domain.
            rboxPropsNew = []
            regs = self.regionsMap[edom]
            for regProps in rboxProps:

                # Make incomplete region box for evaluation count and
                # regions() to determine region properties. Both of
                # these could raise an NeedRegionEnumerationException.
                rbox = RegionBox(self, regProps)
                for props in regs.regions(rbox):
                    regProps = dict(regProps)
                    regProps[edom] = props
                    rboxPropsNew.append(regProps)
            rboxProps = rboxPropsNew

        # Done. Now we take care of all non-enumerated
        # domains. Ideally this should just means calling region() to
        # get the properties for each. But again this might raise
        # NeedRegionEnumerationException, which sends us back to
        # square one.
        rboxes = []
        for props in rboxProps:
            # Make yet another temporary region box
            rbox = RegionBox(self, props)
            for prio in [True, False]:
                for regs in self.regionsBox:
                    if not regs.domain in self.enumDomains and \
                       (prio == (regs.domain in self.priorityDomains)):
                        props[regs.domain] = regs.region(rbox)
            # Props have changed, construct the (finally)
            # completed region box
            rboxes.append(RegionBox(self, props))
        return rboxes

    def __repr__(self):
        return '<RegionBoxes %s>' % (repr(self.boxes))

    def domains(self):
        return self.regionsMap.keys()

    def regions(self, dom):
        """Returns all regions we see for the given domain. Note that regions
        might be duplicated due to enumeration. This is the right
        behaviour, as different region boxes might have different
        regions for this domain.
        """
        assert dom in self.regionsMap
        return [ list(box.regions(dom)) for box in self.boxes ]

    def bounds(self, dom):
        assert dom in self.regionsMap
        from itertools import chain
        return [ list(box.bounds(dom)) for box in self.boxes ]

    def allBounds(self):
        assert dom in self.regionsMap
        from itertools import chain
        return [ box.allBounds() for box in self.boxes ]

    def count(self):
        """Return the total number of boxes in this box set."""
        return self._withEnums(lambda: self._count())
    def _count(self):
        count = 0
        for box in self.boxes:
            count += box.count()
        return count

    def the(self, prop):
        """Return the value of the given property for this box set. If it
        cannot proven to be constant for all boxes, an exception will
        be raised."""
        return self._withEnums(lambda: self._the(prop))
    def _the(self, prop):
        s = None
        for box in self.boxes:
            if s is None:
                s = box.the(prop)
            else:
                s2 = box.the(prop)
                if s != s2:
                    raise Exception("Property is not constant: %s != %s!" % (s, s2))
        return s

    def sum(self, prop):
        return self._withEnums(lambda: self._sum(prop))
    def _sum(self, prop):
        s = 0
        for box in self.boxes:
            s += box.sum(prop)
        return s

    def max(self, prop):
        return self._withEnums(lambda: self._max(prop))
    def _max(self, prop):
        s = None
        for box in self.boxes:
            if s is None:
                s = box.the(prop)
            else:
                s = Max(s, box.the(prop))
        return s

    def zipSum(left, right, f, verbose=False):
        return left._withEnums(lambda: right._withEnums(
            lambda: left._zipSum(right, f, verbose)))
    def _zipSum(left, right, f, verbose):
        """
        Return sum of function, applied to all edges from one region box
        set to another. An edge exists between two region boxes if all
        regions overlap. Domains that exist on only one side are
        ignored for this purpose - so e.g. region boxes involving
        distinct domains always have an edge, but region boxes that
        have a non-overlapping region in some domain never have.
        """

        # Classify domains
        leftDoms = []
        rightDoms = []
        commonDoms = []
        for dom in left.regionsMap.keys():
            if dom in right.regionsMap:
                commonDoms.append(dom)
            else:
                leftDoms.append(dom)
        for dom in right.regionsMap.keys():
            if not dom in left.regionsMap:
                rightDoms.append(dom)

        # Sum up result of zip
        result = 0
        for lbox in left.boxes:
            for rbox in right.boxes:
                result += lbox._zipSum(rbox, commonDoms, leftDoms, rightDoms, f, verbose)
        return result

    def zipCrossSum(left, right, cross, f, verbose=False):
        return left._withEnums(lambda: right._withEnums(
            lambda: left._zipCrossSum(right, cross, f, verbose)))
    def _zipCrossSum(left, right, cross, f, verbose):

        # Classify domains
        leftDoms = []
        rightDoms = []
        commonDoms = []
        for dom in left.regionsMap.keys():
            if dom in right.regionsMap:
                commonDoms.append(dom)
            else:
                leftDoms.append(dom)
        for dom in right.regionsMap.keys():
            if not dom in left.regionsMap:
                rightDoms.append(dom)

        if verbose:
            print("Left doms:", leftDoms)
            print("Right doms:", rightDoms)
            print("Common doms:", commonDoms)
            print("Cross doms:", cross.regionsMap.keys())

        # Sum up result of zip
        result = 0
        for cbox in cross.boxes:
            for lbox in left.boxes:
                for rbox in right.boxes:
                    result += lbox._zipCrossSum(rbox, cbox, commonDoms, leftDoms, rightDoms, f, verbose)
        return result


class Flow:
    """A flow is a set of results, produced for the cartesian product of a
    number of region sets. Producing these results will have costs
    associated with it - e.g. 'compute' costs to produce the result in
    the first place or 'transfer' cost for the size of the result when
    transferred to other nodes.
    """

    def __init__(self, name, regss = [], costs = {}, attrs = {}, deps = [], weights = [],
                 cluster=''):
        self.name = name
        self.boxes = RegionBoxes(regss)
        self.costs = costs
        self.attrs = attrs
        self.deps = deps
        self.weights = weights + [1] * (len(deps) - len(weights))
        self.cluster = cluster

    def domains(self):
        return self.boxes.domains()

    def depend(self, flow, weight=1):
        self.deps.append(flow)
        self.weights.append(weight)

    def removeDepend(self, flow):
        try:
            ix = self.deps.index(flow)
            del self.deps[ix]
            del self.weights[ix]
        except ValueError:
            pass

    def output(self, name, costs, attrs={}):
        """Make a new Flow for an output of this Flow. Useful when a Flow has
        multiple outputs that have different associated costs."""
        return Flow(name, self.regionsBox, deps=[self], costs=costs, attrs=attrs)

    def count(self):
        return self.boxes.count()

    def the(self, prop):
        return self.boxes.the(prop)

    def sum(self, prop):
        return self.boxes.sum(prop)

    def max(self, prop):
        return self.boxes.max(prop)

    def edges(self, dep):
        """Calculates number of edges coming from a dependency."""
        return self.edgeSum(dep, 1)

    def edgeSum(self, dep, prop):
        """
        Returns the sum of an edge property over all edges coming from the
        given dependency.
        """
        if not dep in self.deps:
            return 0
        return dep.boxes.zipSum(self.boxes, lambda l, r: DomainExpr(prop).eval(l))

    def crossEdges(self, dep, crossBoxes):
        """
        Calculates number of edges from a dependency that end in a
        different region box than they start in (for a given granularity).
        """
        return self.crossEdgeSum(dep, crossBoxes, 1)

    def crossEdgeSum(self, dep, crossBoxes, prop):
        """
        Sums up a property over all crossing edges from a dependency.
        :param dep: Dependency to formulate edge cross sum for
        :param crossBoxes: Granularity for cross check. An edge counts
          as crossing if it starts in a different box than it ends in.
        :param prop: Property to sum up. Can depend on properties of
          the dependency's region box (the edge "start").
        """
        if not dep in self.deps:
            return 0
        return dep.boxes.zipCrossSum(self.boxes, crossBoxes, lambda l, r: DomainExpr(prop).eval(l))

    def cost(self, name):
        if name in self.costs:
            return self.sum(DomainExpr(self.costs[name]))
        else:
            return 0

    def regions(self):
        return self.boxes.regionsBox

    def recursiveDeps(self):
        """ Returns all direct and indirect dependencies of this node."""

        active = [self]
        recDeps = [self]

        while len(active) > 0:
            node = active.pop()
            for dep in node.deps:
                if not dep in recDeps:
                    active.append(dep)
                    recDeps.append(dep)

        return list(recDeps)

    def getDep(self, name):
        """Returns a flow with the given name from all (possibly indirect)
        dependencies of this node."""

        active = [self]
        recDeps = [self]
        while len(active) > 0:
            node = active.pop()
            if node is None:
                continue
            if node.name == name:
                return node
            for dep in node.deps:
                if not dep in recDeps:
                    active.append(dep)
                    recDeps.append(dep)

        return None

    def getDeps(self, names):
        """Look up multiple dependencies."""

        return list(filter(lambda f: not f is None,
                           map(lambda n: self.getDep(n), names)))

    def recursiveCost(self, name):
        """
        Returns the sum of the given cost over this flow and all its
        dependencies.
        """

        cost = 0
        for dep in self.recursiveDeps():
            cost += dep.cost(name)
        return cost

    def recursiveEdgeCost(self, name):
        """
        Returns the sum of the given cost over all edges in the flow network
        """

        cost = 0
        for flow in self.recursiveDeps():
            for dep in flow.deps:
                if name in dep.costs:
                    cost += flow.edgeSum(dep, dep.costs[name])
        return cost

def mergeFlows(root, group, regs, reevalTransfer=False):
    """Group given flows into a common Flow with the desired
    granularity.

    This is only allowed if there is one designated output Flow for
    the group. Or put another way: Only one Flow is allowed to have
    outside Flows depend on it.

    Normally the cost is going to be redistributed "proportionally"
    among the tasks of the new granularity. This is only the right
    thing to do if:

    - the merged tasks have the same computational and transfer cost
      as the original tasks (output tasks) put together, and
    - both output and transfer cost are distributed equally for the
      merged tasks

    So basically, we can only merge if the end state is a bunch of
    tasks that all look exactly the same. The one way to get around
    this is to recalculate transfer costs: This makes sense if the
    transfer formula of the output flow can simply be re-evaluated for
    the new granularity.

    :param root: Root flow of the flow network
    :param group: Flows to group together
    :param regs: Granularity to use for merged flow
    :param reevalTransfer: Recaluluate output amount for the new
      granularity.
    """

    boxes = RegionBoxes(regs)
    count = boxes.count()
    allFlows = root.recursiveDeps()

    flops = 0
    in_transfer = 0
    in_count = 0
    out_transfer = None
    cluster = None
    deps = set()
    revDeps = set()

    for flow in group:
        assert flow in allFlows

        # Determine cluster
        if cluster is None:
            cluster = flow.cluster
        elif cluster != flow.cluster:
            cluster = ''

        # Count flops
        flops += flow.cost('compute')

        # Determine input transfer
        for d in flow.deps:
            if not d in group:
                deps.add(d)

        # Only count transfer that leaves group
        outside_dep = False
        for f in root.recursiveDeps():
            if f in group:
                continue
            for d in f.deps:
                if d == flow:
                    outside_dep = True
                    revDeps.add(f)
        if outside_dep and 'transfer' in flow.costs:

            # Only one output flow allowed
            assert out_transfer is None, "Cannot merge group %s with two outgoing flows!" % group
            domains = { reg.domain for reg in regs }

            # Re-evaluate transfer amount?
            if reevalTransfer:

                # Determine whether any domains got removed
                out_transfer = flow.costs['transfer']
                removedDoms = set(flow.domains()).difference(domains)
                if len(removedDoms) > 0:

                    # Make new regions box for the removed regions
                    remRegs = RegionBoxes([ flow.boxes.regionsMap[dom]
                                            for dom in removedDoms ])

                    # Replace transfer property with a sum over the product
                    out_transfer_old = out_transfer # Prevent recursion
                    out_transfer = lambda rb: \
                    remRegs.sum(lambda rb2:
                                out_transfer_old(rb.product(rb2)))

            else:

                # Evaluate original transfer amount, split "equally"
                # amongst merged task
                out_transfer = flow.cost('transfer') / count

    # Create new Flow
    groupFlow = Flow(' + '.join(map(lambda f: f.name, group)), regs,
                     cluster=cluster,
                     costs = { 'compute': flops / count,
                               'transfer': out_transfer },
                     deps = list(deps)
    )

    # Change dependencies of existing flows
    for f in revDeps:
        for mf in group:
            f.removeDepend(mf)
        f.depend(groupFlow)

    return groupFlow

def makeSplitFlow(root, flow, regs):
    """
    Make a new Flow that redistributes the given flow's output. This
    makes sense if downstream flows have a finer granularity, and we
    do not want to send the full task result to each of them. This
    cleanly splits the 'compute' and 'transfer' costs: The new split
    flow will have zero computation cost, and the old flow will have
    zero transfer cost.

    Note that the effect of this chiefly depends on how the given
    flow's 'transfer' property scales with the changed granularity.

    :param root: Root flow of the flow network
    :param flow: Flow we want to split the output for
    :param regs: Granularity to use for split flow
    """

    allFlows = root.recursiveDeps()
    assert flow in allFlows

    # Set transfer cost of old flow to zero (by copying it)
    copyCosts = { name: cost for (name, cost) in flow.costs.items()
                  if name != 'transfer' }
    copy = Flow(flow.name, flow.boxes.regionsBox,
                cluster=flow.cluster,
                costs=copyCosts,
                deps=flow.deps[:],
                weights=flow.weights[:])

    # Create the new Flow
    splitFlow = Flow(flow.name + " (split)", regs,
                     cluster=flow.cluster,
                     costs = { 'transfer': flow.costs.get('transfer') },
                     deps = [copy])

    # Rewrite dependencies
    for f in allFlows:
        if flow in f.deps:
            f.removeDepend(flow)
            f.depend(splitFlow)

    return splitFlow

def flowsToDot(root, t, computeSpeed=None,
               graph_attr={}, node_attr={'shape':'box'}, edge_attr={},
               showRegions=True, showRates=True, showTaskRates=False,
               showGranularity=True, showDegrees=True, cross_regs=None):

    # Get root flow dependencies
    flows = root.recursiveDeps()

    # Make digraph
    dot = Digraph(graph_attr=graph_attr,node_attr=node_attr,edge_attr=edge_attr)

    # Assign IDs to flows, collect clusters
    flowIds = {}
    clusters = {}
    for i, flow in enumerate(flows):
        flowIds[flow] = "node.%d" % i
        if flow.cluster in clusters:
            clusters[flow.cluster].append(flow)
        else:
            clusters[flow.cluster] = [flow]

    # Make boxes for cross regions
    crossBoxes = None
    if not cross_regs is None:
        crossBoxes = RegionBoxes(cross_regs)

    # Helper for formatting sizes
    def format_bytes(n):
        if n < 10**9: return '%.1f MB' % (n/10**6)
        return '%.1f GB' % (n/10**9)

    # Make nodes per cluster
    for cluster, cflows in clusters.items():

        if cluster == '':
            graph = dot
        else:
            graph = Digraph(name="cluster_"+cluster,
                            graph_attr={'label':cluster})

        for flow in cflows:

            # Start with flow name
            text = flow.name

            # Add relevant regions
            def regName(r): return r.domain.name
            if showRegions:
                for region in sorted(flow.regions(), key=regName):
                    count = flow.max(region.domain('count'))
                    size = flow.max(region.domain('size'))
                    if count == 1 and size == 1:
                        continue
                    text += "\n%s: %d x %g %s" % \
                            (region.domain.name,
                             flow.max(region.domain('count')),
                             flow.max(region.domain('size')),
                             region.domain.unit)

            # Add compute count
            count = flow.count()
            try:
                compute = flow.cost('compute')
                transfer = flow.cost('transfer')
            except:
                print()
                print("Exception raised while determining cost for '" + flow.name + "':")
                raise
            if showGranularity:
                if count != 1:
                    text += "\nTasks: %.3g" % (count)
                    if count > t:
                        text += " (%d/s)" % (count/t)
                if compute > 0 and computeSpeed is not None:
                    text += "\nRuntime: %.3g s/task" % (compute/count/computeSpeed)

            # Show overall rates
            if showRates:
                if compute > 0:
                    text += "\nFLOPs: %.2f TOP/s" % (compute/t/Constants.tera)
                if transfer > 0:
                    text += "\nOutput: %.2f TB/s" % (transfer/t/Constants.tera)

            # Show per-task data statistic
            if showTaskRates:

                # Determine input data
                in_count = 0
                in_transfer = 0
                for dep in flow.deps:
                    in_count += dep.boxes.zipSum(flow.boxes, lambda l,r: 1)
                    if 'transfer' in dep.costs:
                        def transfer_prop(l,r): return DomainExpr(dep.costs['transfer']).eval(l)
                        in_transfer += dep.boxes.zipSum(flow.boxes, transfer_prop)

                if compute > 0:
                    if in_transfer > 0:
                        text += "\nTask Input: %s (%s/s)" % \
                                (format_bytes(in_transfer / flow.boxes.count()),
                                 format_bytes(in_transfer / flow.boxes.count() / compute*count*computeSpeed))
                    if transfer > 0:
                        text += "\nTask Output: %s (%s/s)" % \
                                (format_bytes(transfer / flow.boxes.count()),
                                 format_bytes(transfer / flow.boxes.count() / compute*count*computeSpeed))
                else:
                    if in_transfer > 0:
                        text += "\nTask Input: %s" % \
                                format_bytes(in_transfer / flow.boxes.count())
                    if transfer > 0:
                        text += "\nTask Output: %s" % \
                                format_bytes(transfer / flow.boxes.count())

            attrs = flow.attrs
            graph.node(flowIds[flow], text, attrs)

            # Add dependencies
            for dep, weight in zip(flow.deps, flow.weights):
                if dep in flowIds:

                    # Calculate number of edges, and use node counts
                    # on both sides to calculate (average!) in and out
                    # degrees.
                    edges = flow.edges(dep)
                    label = ''
                    if 'transfer' in dep.costs:
                        transfer = flow.edgeSum(dep, dep.costs['transfer'])
                        label = '%.1g x %s' % \
                                (edges, format_bytes(transfer/edges))

                    # Do in-/out-degree analysis
                    if showDegrees:
                        depcount = dep.count()
                        flowcount = flow.count()
                        if label != '':
                            label += '\n'
                        if 'transfer' in dep.costs and transfer > 0:
                            label += 'out: %.1f (%s)\nin: %.1f (%s)' % \
                                     (edges/depcount, format_bytes(transfer/depcount),
                                      edges/flowcount, format_bytes(transfer/flowcount))
                        else:
                            label += 'out: %d\nin: %d' % (edges/depcount, edges/flowcount)

                    # Check for box crossings
                    if crossBoxes:
                        cross = flow.crossEdges(dep, crossBoxes)
                        label += '\ncrossing: %.1f%%' % (100 * cross / edges)
                        if 'transfer' in dep.costs and transfer > 0:
                            crossTransfer = flow.crossEdgeSum(dep, crossBoxes, dep.costs['transfer'])
                            label += ' (%.2f TB/s)' % (transfer * cross / edges / t / Constants.tera)

                    dot.edge(flowIds[dep], flowIds[flow], label, weight=str(weight))


        if cluster != '':
            dot.subgraph(graph)

    return dot

class DataFlowTests(unittest.TestCase):

    def setUp(self):

        # Set up a domain with four different splits - all, each,
        # split into 4 regions and split into 3 regions. All sizes
        # given as floating point numbers, as we should be able to
        # somewhat consistently work with that.
        self.size1 = 12.
        self.dom1 = Domain("Domain 1")
        props = {'dummy': lambda rb: 0 }
        self.reg1 = self.dom1.regions(self.size1, props=props)
        self.split1 = self.reg1.split(self.size1, props=props)
        self.size1a = 4.
        self.split1a = self.reg1.split(self.size1a, props=props)
        self.size1b = 3.
        self.split1b = self.reg1.split(self.size1b, props=props)

        # Assumed by tests
        assert self.size1a > self.size1b and self.size1a < 2*self.size1b

        # Second domain, with splits "all" and "each".
        self.size2 = 7.
        self.dom2 = Domain("Domain 2")
        self.reg2 = self.dom2.regions(self.size2, props=props)
        self.split2 = self.reg2.split(self.size2, props=props)

        # Smaller region for the second domain, with splits "all" and "each".
        self.size3 = 5.
        self.dom3 = self.dom2
        self.reg3 = self.dom3.regions(self.size3, props=props)
        self.split3 = self.reg3.split(self.size3, props=props)

    def test_size(self):

        # Expected region properties
        self.assertEqual(self.reg1.size, self.size1)
        self.assertEqual(self.split1.size, self.size1)
        self.assertEqual(self.split1a.size, self.size1)
        self.assertEqual(self.split1b.size, self.size1)
        countProp = self.dom1(COUNT_PROP)
        self.assertEqual(self.reg1.the(countProp), 1)
        self.assertEqual(self.split1.the(countProp), self.size1)
        self.assertEqual(self.split1a.the(countProp), self.size1a)
        self.assertEqual(self.split1b.the(countProp), self.size1b)

    def test_rboxes_1(self):
        self._test_rboxes_1(self.reg1, 1, self.size1)
        self._test_rboxes_1(self.split1, self.size1, 1)
        self._test_rboxes_1(self.split1a, self.size1a, self.size1 / self.size1a)
        self._test_rboxes_1(self.split1b, self.size1b, self.size1 / self.size1b)

    def _test_rboxes_1(self, regs, count, boxSize):
        rboxes = RegionBoxes([regs])
        self.assertEqual(rboxes.count(), count)

        # Summing up a value of one per box should yield the box count
        self.assertEqual(rboxes.sum(lambda rb: 1), count)
        countProp = self.dom1(COUNT_PROP)
        self.assertEqual(rboxes.the(countProp), count)

        # Summing up the box sizes should give us the region size
        sizeProp = self.dom1(SIZE_PROP)
        self.assertEqual(rboxes.the(sizeProp), boxSize)
        self.assertEqual(rboxes.sum(sizeProp), self.size1)

        # Should all still work when we force the boxes to be
        # enumerated by depending on the dummy property
        dummySizeProp = self.dom1('dummy') + self.dom1(SIZE_PROP)
        self.assertFalse(self.dom1 in rboxes.enumDomains)
        self.assertEqual(rboxes.the(dummySizeProp), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp), self.size1)
        self.assertTrue(self.dom1 in rboxes.enumDomains)
        self.assertEqual(rboxes.the(countProp), count)

        # Summing up indices should work as expected
        indexProp = self.dom1(INDEX_PROP)
        self.assertEqual(rboxes.max(indexProp), count-1)
        self.assertEqual(rboxes.sum(indexProp), count * (count-1) // 2)
        offsetProp = self.dom1(OFFSET_PROP)
        self.assertEqual(rboxes.max(offsetProp), (count-1) * boxSize)
        self.assertEqual(rboxes.sum(offsetProp), (count * (count-1) // 2) * boxSize)

    def test_rboxes_2(self):

        # Same tests as rboxes_1, but with two nested domains. We
        # especially check that domain order does not make a difference.
        self._test_rboxes_2([self.reg1, self.reg2], 1, self.size1 * self.size2)
        self._test_rboxes_2([self.reg1, self.split2], self.size2, self.size1)
        self._test_rboxes_2([self.split1, self.reg2], self.size1, self.size2)
        self._test_rboxes_2([self.split1, self.split2], self.size1 * self.size2, 1)
        self._test_rboxes_2([self.split1a, self.reg2], self.size1a, self.size2 * self.size1 / self.size1a)
        self._test_rboxes_2([self.split1a, self.split2], self.size1a * self.size2, self.size1 / self.size1a)
        self._test_rboxes_2([self.split1b, self.reg2], self.size1b, self.size2 * self.size1 / self.size1b)
        self._test_rboxes_2([self.split1b, self.split2], self.size1b * self.size2, self.size1 / self.size1b)
        self._test_rboxes_2([self.reg2, self.reg1], 1, self.size1 * self.size2)
        self._test_rboxes_2([self.split2, self.reg1], self.size2, self.size1)
        self._test_rboxes_2([self.reg2, self.split1], self.size1, self.size2)
        self._test_rboxes_2([self.split2, self.split1], self.size1 * self.size2, 1)
        self._test_rboxes_2([self.reg2, self.split1a], self.size1a, self.size2 * self.size1 / self.size1a)
        self._test_rboxes_2([self.split2, self.split1a], self.size1a * self.size2, self.size1 / self.size1a)
        self._test_rboxes_2([self.reg2, self.split1b], self.size1b, self.size2 * self.size1 / self.size1b)
        self._test_rboxes_2([self.split2, self.split1b], self.size1b * self.size2, self.size1 / self.size1b)

    def _test_rboxes_2(self, regss, count, boxSize):
        rboxes = RegionBoxes(regss)
        self.assertEqual(rboxes.count(), count)
        self.assertEqual(rboxes.sum(lambda rb: 1), count)
        sizeProp = self.dom1(SIZE_PROP) * self.dom2(SIZE_PROP)
        self.assertEqual(rboxes.the(sizeProp), boxSize)
        self.assertEqual(rboxes.sum(sizeProp), self.size1*self.size2)
        dummySizeProp1 = self.dom1('dummy') + sizeProp
        self.assertEqual(rboxes.the(dummySizeProp1), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp1), self.size1*self.size2)
        rboxes = RegionBoxes(regss)
        dummySizeProp2 = self.dom2('dummy') + sizeProp
        self.assertEqual(rboxes.the(dummySizeProp2), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp2), self.size1*self.size2)
        rboxes = RegionBoxes(regss)
        dummySizeProp12 = self.dom1('dummy') + self.dom2('dummy') + sizeProp
        self.assertEqual(rboxes.the(dummySizeProp12), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp12), self.size1*self.size2)
        self.assertTrue(self.dom1 in rboxes.enumDomains)
        self.assertTrue(self.dom2 in rboxes.enumDomains)

    def test_rboxes_zip(self):

        # Edges between regions of the same domain grow linearly
        self._test_rboxes_zip([self.reg1],   [self.reg1],    self.dom1, self.dom1, 1)
        self._test_rboxes_zip([self.reg1],   [self.split1],  self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.reg1],   [self.split1a], self.dom1, self.dom1, self.size1a)
        self._test_rboxes_zip([self.reg1],   [self.split1b], self.dom1, self.dom1, self.size1b)
        self._test_rboxes_zip([self.split1], [self.reg1],    self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1], [self.split1],  self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1], [self.split1a], self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1], [self.split1b], self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1a], [self.split1b], self.dom1, self.dom1, 2 * self.size1b)
        self._test_rboxes_zip([self.split1b], [self.split1a], self.dom1, self.dom1, 2 * self.size1b)

        # Edges between regions of different domains grow quadradically
        self._test_rboxes_zip([self.reg1],   [self.reg2],   self.dom1, self.dom2, 1)
        self._test_rboxes_zip([self.reg1],   [self.split2], self.dom1, self.dom2, self.size2)
        self._test_rboxes_zip([self.split1], [self.reg2],   self.dom1, self.dom2, self.size1)
        self._test_rboxes_zip([self.split1], [self.split2], self.dom1, self.dom2, self.size1 * self.size2)
        self._test_rboxes_zip([self.split1a],[self.split2], self.dom1, self.dom2, self.size1a * self.size2)
        self._test_rboxes_zip([self.split2], [self.split1a],self.dom2, self.dom1, self.size1a * self.size2)
        self._test_rboxes_zip([self.reg2],   [self.split1a],self.dom2, self.dom1, self.size1a)
        self._test_rboxes_zip([self.split1a],[self.reg2],   self.dom1, self.dom2, self.size1a)
        self._test_rboxes_zip([self.reg2],   [self.split1b],self.dom2, self.dom1, self.size1b)
        self._test_rboxes_zip([self.split1b],[self.reg2],   self.dom1, self.dom2, self.size1b)

        # Multidimensional boxes make edge count go up where you would
        # expect. The number of possible cases explodes here, we just
        # test a few representative ones.
        self._test_rboxes_zip([self.split2, self.reg1],   [self.reg2],   self.dom1, self.dom2,
                              self.size2)
        self._test_rboxes_zip([self.split2, self.reg1],   [self.reg2],   self.dom2, self.dom2,
                              self.size2)
        self._test_rboxes_zip([self.split2, self.reg1],   [self.split2], self.dom1, self.dom2,
                              self.size2)
        self._test_rboxes_zip([self.split2, self.reg1],   [self.split2], self.dom2, self.dom2,
                              self.size2)
        self._test_rboxes_zip([self.split2, self.split1], [self.reg2],   self.dom1, self.dom2,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split2, self.split1], [self.reg2],   self.dom2, self.dom2,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split2, self.split1], [self.split2], self.dom1, self.dom2,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split2, self.split1], [self.split2], self.dom2, self.dom2,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split2, self.split1a],[self.split2], self.dom1, self.dom2,
                              self.size1a * self.size2)
        self._test_rboxes_zip([self.split2, self.split1a],[self.split2], self.dom2, self.dom2,
                              self.size1a * self.size2)
        self._test_rboxes_zip([self.split1, self.split2], [self.split1a],self.dom1, self.dom1,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split1, self.split2], [self.split1b],self.dom2, self.dom1,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split1, self.reg2],   [self.split1a],self.dom1, self.dom1,
                              self.size1)
        self._test_rboxes_zip([self.split1, self.reg2],   [self.split1b],self.dom2, self.dom1,
                              self.size1)
        self._test_rboxes_zip([self.split1a, self.reg2],  [self.split1b],self.dom2, self.dom1,
                              2 * self.size1b)
        self._test_rboxes_zip([self.split1b, self.reg2],  [self.split1a],self.dom1, self.dom1,
                              2 * self.size1b)
        self._test_rboxes_zip([self.split2, self.split1a],[self.reg2],   self.dom1, self.dom2,
                              self.size1a * self.size2)
        self._test_rboxes_zip([self.split2, self.split1a],[self.reg2],   self.dom2, self.dom2,
                              self.size1a * self.size2)

    def test_rboxes_zip_different_root(self):

        # Domain 3 is the same as domain 2, however has a different
        # root region size. This means that a few regions split from
        # the larger root region might end up without edges.
        self._test_rboxes_zip([self.reg2],   [self.reg3],   self.dom2, self.dom2, 1)
        self._test_rboxes_zip([self.reg2],   [self.split3], self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.split2], [self.reg3],   self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.split2], [self.split3], self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.reg3],   [self.reg2],   self.dom2, self.dom2, 1)
        self._test_rboxes_zip([self.split3], [self.reg2],   self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.reg3],   [self.split2], self.dom2, self.dom2, self.size3)
        self._test_rboxes_zip([self.split3], [self.split2], self.dom2, self.dom2, self.size3)

    def _test_rboxes_zip(self, regsA, regsB, domA, domB, edges):
        rboxesA = RegionBoxes(regsA)
        rboxesB = RegionBoxes(regsB)

        # Summing one per edge should give us the number of
        # edges. Enumerating shouldn't make a difference, as before.
        def oneProp(rbA, rbB): return 1
        self.assertEqual(rboxesA.zipSum(rboxesB, oneProp), edges)
        def onePropA(rbA, rbB): return 1 + rbA.eval(domA, 'dummy')
        self.assertEqual(rboxesA.zipSum(rboxesB, onePropA), edges)
        self.assertTrue(domA in rboxesA.enumDomains)
        self.assertFalse(domA in rboxesB.enumDomains)
        rboxesA = RegionBoxes(regsA)
        rboxesB = RegionBoxes(regsB)
        def onePropB(rbA, rbB): return 1 + rbB.eval(domB, 'dummy')
        self.assertEqual(rboxesA.zipSum(rboxesB, onePropB), edges)
        self.assertFalse(domB in rboxesA.enumDomains)
        self.assertTrue(domB in rboxesB.enumDomains)
        def onePropAB(rbA, rbB): return 1 + rbA.eval(domA, 'dummy') + rbB.eval(domB, 'dummy')
        self.assertEqual(rboxesA.zipSum(rboxesB, onePropAB), edges)

    def test_rboxes_cross_zip_simple(self):

        reg1 = self.reg1; split1 = self.split1; split1a = self.split1a; split1b = self.split1b
        reg2 = self.reg2; split2 = self.split2

        # No edges cross if the cross region is coarser than the
        # region with the finest split.
        self._test_rboxes_cross_zip([reg1],   [reg1],    [reg1], 0)
        self._test_rboxes_cross_zip([reg1],   [split1],  [reg1], 0)
        self._test_rboxes_cross_zip([split1], [reg1],    [reg1], 0)
        self._test_rboxes_cross_zip([split1], [split1],  [reg1], 0)

        # All of them cross if we go in or out of the cross domain
        self._test_rboxes_cross_zip([reg2],   [reg1],    [reg1], 1)
        self._test_rboxes_cross_zip([reg2],   [reg1],    [reg2], 1)
        self._test_rboxes_cross_zip([reg2],   [reg1],    [split1], 1)
        self._test_rboxes_cross_zip([reg2],   [reg1],    [split2], 1)
        self._test_rboxes_cross_zip([reg2],   [reg2],    [reg1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [reg2], 0)
        self._test_rboxes_cross_zip([reg2],   [reg2],    [split1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [split2], 0)
        self._test_rboxes_cross_zip([reg2],   [reg2],    [reg1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [reg2], 0)
        self._test_rboxes_cross_zip([reg2],   [reg2],    [split1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [split2], 0)
        self._test_rboxes_cross_zip([reg2],   [split1],  [reg1], self.size1)
        self._test_rboxes_cross_zip([reg2],   [split1],  [reg2], self.size1)
        self._test_rboxes_cross_zip([split2], [split1],  [reg1], self.size1 * self.size2)
        self._test_rboxes_cross_zip([split2], [split1],  [reg2], self.size1 * self.size2)

        # However, all edges but one cross if we make it finer
        self._test_rboxes_cross_zip([split1], [reg1],    [split1], self.size1 - 1)
        self._test_rboxes_cross_zip([reg1],   [split1],  [split1], self.size1 - 1)

        # Yet there is no crossing unless we have diagonal edges
        self._test_rboxes_cross_zip([reg1],   [reg1],    [split1], 0)

        # For coarser splits, a few more edges end up non-crossing
        self._test_rboxes_cross_zip([split1], [reg1],    [split1a], self.size1 - self.size1 / self.size1a)
        self._test_rboxes_cross_zip([split1], [reg1],    [split1b], self.size1 - self.size1 / self.size1b)

        # Furthermore, a N-to-1 split will produce (N-1) crossing
        # edges - basically all but the horizontal edges.
        self._test_rboxes_cross_zip([split1], [split1a], [split1], self.size1a * (self.size1 / self.size1a - 1))
        self._test_rboxes_cross_zip([split1], [split1b], [split1], self.size1b * (self.size1 / self.size1b - 1))

        # Things get more complicated if we use a disagreeing split
        # number for detecting crossings. Not 100% sure whether the
        # formulas match reality, but I am pretty sure that the
        # (numerical) results of 3 and 6 respectively are correct.
        self._test_rboxes_cross_zip([split1], [split1a], [split1b],
                                    (self.size1 / self.size1a) * (self.size1 / self.size1a - 1) / 2)
        self._test_rboxes_cross_zip([split1], [split1b], [split1a],
                                    (self.size1 / self.size1b) * (self.size1 / self.size1b - 1) / 2)

        # Adding extra domains into the mix multiplies things up as should be expected
        self._test_rboxes_cross_zip([split1, reg2], [reg1], [split1a],
                                    self.size1 - self.size1 / self.size1a)
        self._test_rboxes_cross_zip([split1, reg2], [reg1], [split1b],
                                    self.size1 - self.size1 / self.size1b)
        self._test_rboxes_cross_zip([split1, split2], [reg1], [split1a],
                                    self.size2 * (self.size1 - self.size1 / self.size1a))
        self._test_rboxes_cross_zip([split1, split2], [reg1], [split1b],
                                    self.size2 * (self.size1 - self.size1 / self.size1b))

    def test_rboxes_cross_zip_multiple1(self):
        reg1 = self.reg1; split1 = self.split1; split1a = self.split1a; split1b = self.split1b
        reg2 = self.reg2; split2 = self.split2

        # Simple cases for splitting by multiple domains
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1, reg2], [reg1, reg2], 0)
        self._test_rboxes_cross_zip([split1, reg2], [reg1, reg2], [reg1, reg2], 0)
        self._test_rboxes_cross_zip([reg1, split2], [reg1, reg2], [reg1, reg2], 0)
        self._test_rboxes_cross_zip([split1,split2],[reg1, reg2], [reg1, reg2], 0)
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1],       [reg1, reg2], 1)
        self._test_rboxes_cross_zip([split1, reg2], [reg1],       [reg1, reg2], self.size1)
        self._test_rboxes_cross_zip([reg1, split2], [reg1],       [reg1, reg2], self.size2)
        self._test_rboxes_cross_zip([split1,split2],[reg1],       [reg1, reg2], self.size1 * self.size2)

    def test_rboxes_cross_zip_multiple2(self):
        reg1 = self.reg1; split1 = self.split1; split1a = self.split1a; split1b = self.split1b
        reg2 = self.reg2; split2 = self.split2

        # If a split domain doesn't exist on one side, all existing edges cross.
        self._test_rboxes_cross_zip([reg1,split2],  [reg1],       [reg1, split2], self.size2)
        self._test_rboxes_cross_zip([split1,reg2],  [reg1],       [reg1, split2], self.size1)
        self._test_rboxes_cross_zip([split1,reg2],  [reg1],       [split1, reg2], self.size1)
        self._test_rboxes_cross_zip([split1,split2],[reg1],       [reg1, split2], self.size1*self.size2)
        self._test_rboxes_cross_zip([split1,split2],[reg1],       [split1,split2],self.size1*self.size2)
        self._test_rboxes_cross_zip([split1,split2],[reg1],       [split1, reg2], self.size1*self.size2)
        self._test_rboxes_cross_zip([split1,split2],[split1],     [split1, reg2], self.size1*self.size2)

    def test_rboxes_cross_zip_multiple3(self):
        reg1 = self.reg1; split1 = self.split1; split1a = self.split1a; split1b = self.split1b
        reg2 = self.reg2; split2 = self.split2

        # If they both exist, things multiply out
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1, reg2], [split1, reg2], 0)
        self._test_rboxes_cross_zip([split1, reg2], [reg1, reg2], [split1, reg2], self.size1 - 1)
        self._test_rboxes_cross_zip([reg1, split2], [reg1, reg2], [split1, reg2], 0)
        self._test_rboxes_cross_zip([split1,split2],[reg1, reg2], [split1, reg2], (self.size1 - 1) * self.size2)
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1, reg2], [reg1, split2], 0)
        self._test_rboxes_cross_zip([split1, reg2], [reg1, reg2], [reg1, split2], 0)
        self._test_rboxes_cross_zip([reg1, split2], [reg1, reg2], [reg1, split2], self.size2 - 1)
        self._test_rboxes_cross_zip([split1,split2],[reg1, reg2], [reg1, split2], self.size1 * (self.size2 - 1))
        self._test_rboxes_cross_zip([reg1, reg2],   [reg1, reg2], [split1,split2],0)
        self._test_rboxes_cross_zip([split1, reg2], [reg1, reg2], [split1,split2],self.size1 - 1)
        self._test_rboxes_cross_zip([reg1, split2], [reg1, reg2], [split1,split2],self.size2 - 1)
        self._test_rboxes_cross_zip([split1,split2],[reg1, reg2], [split1,split2],self.size1 * self.size2 - 1)

    def _test_rboxes_cross_zip(self, regsA, regsB, regsC, edges):
        import itertools
        def enums(regs):
            doms = map(lambda rs: rs.domain, regs)
            return itertools.product(doms, [False, True])
        for (domA, enumA), (domB, enumB), (domC, enumC) in \
                itertools.product(enums(regsA), enums(regsB), enums(regsC)):

            # Construct region boxes
            rboxesA = RegionBoxes(regsA)
            rboxesB = RegionBoxes(regsB)
            rboxesC = RegionBoxes(regsC)

            # Make sure the right domains end up enumerated
            if enumA: rboxesA.sum(1 + domA('dummy'))
            if enumB: rboxesB.sum(1 + domB('dummy'))
            if enumC: rboxesC.sum(1 + domC('dummy'))

            # Check that bounds is consistent, as
            # zipCrossSum will depend on it heavily
            def toOffSize(reg): return (reg[OFFSET_PROP], reg[SIZE_PROP])
            def toOffSizes(regs): return list(map(toOffSize, regs))
            self.assertEqual(list(map(toOffSizes, rboxesA.regions(domA))), list(rboxesA.bounds(domA)))
            self.assertEqual(list(map(toOffSizes, rboxesB.regions(domB))), list(rboxesB.bounds(domB)))
            self.assertEqual(list(map(toOffSizes, rboxesC.regions(domC))), list(rboxesC.bounds(domC)))

            # Summing one per edge should give us the number of
            # edges, no matter whether we ask for the dummy
            # property or not.
            msg = "cross sum mismatch for regions %s vs %s, enumerating [%s %s %s]" % \
                (regsA, regsB,
                 domA.name if enumA else '',
                 domB.name if enumB else '',
                 domC.name if enumC else '')
            def onePropEdge(rbA, rbB): return 1
            self.assertEqual(rboxesA.zipCrossSum(rboxesB, rboxesC, onePropEdge),edges,msg=msg)
            self.assertEqual(rboxesB.zipCrossSum(rboxesA, rboxesC, onePropEdge),edges,msg='reversed ' + msg)

class DataFlowTestsSymbol(unittest.TestCase):

    def test_symbols(self):

        # Make a region with entirely symbolised properties
        dom = Domain('Symbolized')
        size = Symbol('size')
        regs = dom.regions(size)
        i = Symbol("i")
        split = regs.split(size, props = {
            'p1': 1,
            'p2': Lambda(i, 1),
            'p3': Lambda(i, i) })

        # Check size
        rbox = RegionBoxes([split])
        self.assertEqual(rbox.count(), size)
        self.assertEqual(rbox.sum(dom('p1')), size)
        self.assertEqual(rbox.sum(dom('p2')).doit(), size)
        self.assertEqual(rbox.sum(dom('p3')).doit(), size**2/2-size/2)

class DataFlowTestsFlow(unittest.TestCase):

    def setUp(self):

        # Again, set up some domains and splits so we can build a
        # meaningful flow network.
        self.size1 = 12.
        self.dom1 = Domain("Domain 1")
        self.reg1 = self.dom1.regions(self.size1)
        self.split1 = self.reg1.split(self.size1)
        self.size2 = 7.
        self.dom2 = Domain("Domain 2")
        self.reg2 = self.dom2.regions(self.size2, props={'testCost': 100 })
        self.split2 = self.reg2.split(self.size2, props={'testCost': 100 })
        self.size3 = 5.
        self.dom3 = self.dom2
        self.reg3 = self.dom3.regions(self.size3)
        self.split3 = self.reg3.split(self.size3)

    def _makeFlows(self):

        # Make some input nodes without dependencies
        input1 = Flow('Global Input', [], cluster='input', costs = { 'transfer': 12 })
        input2 = Flow('Input Domain 1',
                      [self.split1], cluster='input',
                      costs = { 'transfer': self.dom1('size') })
        input3 = Flow('Input Domain 1 + 2',
                      [self.reg1, self.split2], cluster='input',
                      costs = { 'transfer': self.dom2('size') })

        # Make some intermediate nodes
        interg = Flow('Intermediate Global', [],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [input1, input3]
        )
        self.inter1_name = 'Intermediate Domain 1'
        inter1 = Flow(self.inter1_name, [self.split1],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [interg, input2, input3]
        )
        self.inter2_name = 'Intermediate Domain 2'
        inter2 = Flow(self.inter2_name, [self.split2],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [interg, inter1, input3]
        )
        self.inter3_name = 'Intermediate Domain 3'
        inter3 = Flow(self.inter3_name, [self.split3],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [inter1, inter2, input3]
        )
        inter4 = Flow('Intermediate Domain 1+3', [self.split1, self.split3],
                      costs = { 'compute': 1000, 'transfer': 1000 },
                      deps = [inter1, inter2, input3]
        )

        # Tie them together
        self.gather_name = 'Gather'
        gather = Flow(self.gather_name, [self.reg2],
                      costs = {
                          'compute': 1000,
                          'transfer': self.dom2('testCost') * self.dom2('size')
                      },
                      deps = [inter3, inter4]
        )

        # Do a loop
        self.loop_name = 'Loop'
        loop = Flow(self.loop_name, [self.split3],
                    costs = { 'compute': 1000, 'transfer': 1000},
                    deps = [gather]
        )
        inter2.depend(loop)

        # Add an output node
        output = Flow('Output', [self.split2],
                      costs = {}, deps = [gather])
        return output

    def testFlows(self):

        root = self._makeFlows()

        # Make sure we have all flows in the recursive dependencies
        self.assertEqual(len(root.recursiveDeps()), 11)

        # Make sure all vanish if we remove the main dependency
        dep = root.deps[0]
        root.removeDepend(dep)
        self.assertEqual(len(root.recursiveDeps()), 1)

        # Make sure weights are consistent
        self.assertEqual(len(root.deps), len(root.weights))

    def testDot(self):

        # Run GraphViz generation with all optional features
        root = self._makeFlows()
        dot = flowsToDot(root, t=1.0, computeSpeed=Constants.tera, showTaskRates=True)
        self.assertGreater(len(dot.source), 3000)

        # Analyse crossings for split2
        dot = flowsToDot(root, t=1.0, cross_regs=[self.split2])
        self.assertGreater(len(dot.source), 3000)

        # We would expect a number of different crossing degrees:
        #  '0.0'  - we have edges that stay fully inside
        #  '80.0' - for split3->reg2 (keep in mind that split3 is the same domain)
        #  '85.7' - for reg2->split2
        #  '100'  - for all cases where domain 2 is not present
        # (this check might be a bit fragile...)
        import re
        self.assertEqual(len(set(re.findall("crossing: ([0-9.]+)%", dot.source))), 4)

    def testCost(self):

        # Make sure the cost sum is what we expect
        root = self._makeFlows()
        self.assertEqual(root.recursiveCost('compute'), 91000)
        self.assertEqual(root.recursiveCost('transfer'), 90731)

    def testMerge(self):

        root = self._makeFlows()
        gather = root.getDep(self.gather_name)
        loop = root.getDep(self.loop_name)
        inter1 = root.getDep(self.inter1_name)
        inter2 = root.getDep(self.inter2_name)
        inter3 = root.getDep(self.inter3_name)
        compute = root.recursiveCost('compute')
        produce = root.recursiveCost('transfer')
        transfer = root.recursiveEdgeCost('transfer')

        # First test a No-Op (replace with a Flow of the same granularity)
        gather_i = mergeFlows(root, [gather], [self.reg2])
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

        # If we change to coarser granularity produced data as well as
        # transfered data should reduce - we have removed (size3-1)
        # tasks and their input edges from "Gather".
        mergeFlows(root, [loop], [self.reg3], reevalTransfer=True)
        produce -= (self.size3 - 1) * 1000
        transfer -= (self.size3 - 1) * (100 * self.size2)
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

        # Try merging something into the gather. This will remove some
        # transfer from the overall network (working out the details
        # takes a bit of work with pen and paper, but it checks out.)
        gather_m = mergeFlows(root, [gather_i, inter3], [self.reg2])
        produce -= 1000 * self.size3
        transfer -= 1000 * (self.size1 * self.size3 + 2 * self.size3) + self.size3
        transfer += 1000 * (self.size1 + self.size2) + self.size2
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

        # This should be the same as if we remove the domain, in this
        # case.
        gather_m2 = mergeFlows(root, [gather_m], [])
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

    def testSplit(self):

        root = self._makeFlows()
        gather = root.getDep(self.gather_name)
        compute = root.recursiveCost('compute')
        produce = root.recursiveCost('transfer')
        transfer = root.recursiveEdgeCost('transfer')

        # Make a splitter node after the gather flow. Total data
        # produced will stay the same, but transfer goes down because
        # we have stopped sending the whole task result to the "loop"
        # and "output" nodes.
        makeSplitFlow(root, gather, [self.split2])
        transfer -= (self.size2 + self.size3) * self.size2 * 100
        transfer += (self.size2 + self.size3) * 100
        self.assertEqual(root.recursiveCost('compute'), compute)
        self.assertEqual(root.recursiveCost('transfer'), produce)
        self.assertEqual(root.recursiveEdgeCost('transfer'), transfer)

if __name__ == '__main__':
    unittest.main()
