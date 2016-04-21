
from parameter_definitions import Constants

from sympy import Max, Expr, Lambda, Symbol, Sum
from graphviz import Digraph
from math import floor

import unittest

INDEX_PROP = 'index'
OFFSET_PROP = 'offset'
COUNT_PROP = 'count'
SIZE_PROP = 'size'

DEBUG = False

def mk_lambda(v):
    if callable(v) and not isinstance(v, Expr):
        return v
    else:
        return (lambda rbox: v)

class Domain:
    """A domain is an axis that a computation can be distributed over -
    such as time or frequency.
    """

    def __init__(self, name, unit = ''):
        self.name = name
        self.unit = unit

    def __str__(self):
        return '<Domain %s>' % self.name

    def regions(self, size, props = {}):
        """Make a new unsplit region set for the domain."""
        return Regions(self, size, 1, props)

class Regions:
    """A set of regions of a domain."""

    def __init__(self, domain, size, count, props={}):
        """Makes a new named domain that has given total size and is split up
        into the given number of equal-sized pieces."""

        self.domain = domain
        self.size = size
        self.count = mk_lambda(count)
        self.props = props

    def split(self, split, props = {}):
        return Regions(self.domain, self.size,
                       lambda rb: self.count(rb) * mk_lambda(split)(rb),
                       props)

    def constantSpacing(self):
        """Returns whether the regions have constant spacing. This allows us
        to reason about these regions cheaper."""

        # No size property? Then we assume constant spacing
        if not self.props.has_key(SIZE_PROP):
            return True

        # Otherwise, our size property must convert to a plain number
        # of some sort.
        try:
            float(self.props[SIZE_PROP])
            return True
        except ValueError:
            return False

    def region(self, rb, i = None, off = None):
        """Get properties of a region of this region set"""

        # Enumerated?
        if i is None:

            # Properties that do not depend on the region, or only
            # need to know it symbolically (sympy Lambda) can be
            # worked with even if the region stays anonymous.
            reg = {
                k: v
                for k, v in self.props.iteritems()
                if not callable(v) or isinstance(v, Lambda)
            }

        else:

            # Only copy over detailed properties if our domain is
            # enumerated - they are allowed to depend on the concrete
            # region..
            reg = {
                k: mk_lambda(v)(i)
                for k, v in self.props.iteritems()
            }
            reg[INDEX_PROP] = i
            reg[OFFSET_PROP] = off

        # The split degree is one of the few things we know about
        # regions no matter what.
        reg[COUNT_PROP] = self.count(rb)

        # If size does not get defined in self.props, we provide a
        # default implementation.
        if not reg.has_key(SIZE_PROP) and \
           not self.props.has_key(SIZE_PROP) and \
           reg[COUNT_PROP] != 0:
            reg[SIZE_PROP] = self.size / reg[COUNT_PROP]

        return reg

    def regions(self, rbox):
        """Get all regions of this region set."""

        off = 0 # TODO: regions start offset?
        for i in range(self.count(rbox)):
            props = self.region(rbox, i, off)
            off += props[SIZE_PROP]
            yield props

    def _cheapRegionBounds(self, rbox):
        """Cheaper version of regions that only returns the region bounds."""

        # Constant spacing?
        if self.constantSpacing():
            size = float(self.props.get(SIZE_PROP, self.size / self.count(rbox)))
            for i in range(self.count(rbox)):
                yield (i * size, size)
        # Otherwise fall back to enumerating
        else:
            print "non-constant spacing..."
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

class DomainPrecendenceException:
    """Thrown when the properties of a region are accessed 'before' they
    are defined. This should only happen while RegionBoxes._getBoxes()
    builds up all region boxes, because that is when region boxes
    might not be fully populated yet. This exception means that we
    need to re-order domains in order to be able to determine their
    properties."""
    def __init__(self, rboxes, domain):
        self.regionBoxes = rboxes
        self.domain = domain

class NeedRegionEnumerationException:

    """Thrown when we detect that we need concrete region information
    about a given domain. This exception gets thrown by RegionBox
    when that information is requested, and when caught should
    lead to the domain getting added to the RegionBox's
    enumDomains.
    """
    def __init__(self, rboxes, domain):
        self.regionBoxes = rboxes
        self.domain = domain

class RegionBox:

    def __init__(self, regionBoxes, domainProps):
        self.regionBoxes = regionBoxes
        self.domainProps = domainProps
        self.symbols = {}

    def __call__(self, domain, prop):

        # No properties known yet? This should only happen while we're
        # inside RegionBoxes._getBoxes(). Then it means that this
        # domain needs to be defined before whatever we're doing here.
        if not self.domainProps.has_key(domain):
            raise DomainPrecendenceException(self.regionBoxes, domain)

        # If we don't have the property, this might also point towards
        # the need for enumeration.
        if not self.domainProps[domain].has_key(prop) and \
           not domain in self.regionBoxes.enumDomains:
            raise NeedRegionEnumerationException(self.regionBoxes, domain)

        # Not there?
        if not self.domainProps[domain].has_key(prop):
            raise Exception('Property %s is undefined for this region set of domain %s!'
                             % (prop, domain.name))

        # Look it up
        val = self.domainProps[domain][prop]

        # Symbolised? Pass region index as symbol. Otherwise just return as-is.
        if isinstance(val, Lambda):
            if not self.symbols.has_key(domain):
                self.symbols[domain] = Symbol("i_" + domain.name)
            return val(self.symbols[domain])
        return val

    def count(self):
        count = 1
        # Multiply out all non-enumerated (!) regions
        for dom in self.regionBoxes.regionsMap.iterkeys():
            if not dom in self.regionBoxes.enumDomains:
                count *= self(dom, COUNT_PROP)
        return count

    def the(self, prop):
        """Return the value of the given property for this region box"""
        return mk_lambda(prop)(self)

    def sum(self, prop):
        """Return the sum of the given property for this region box"""

        # Apply property to this region box, collect symbols
        self.symbols = {}
        expr = prop(self)

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
        for dom in self.regionBoxes.regionsMap.iterkeys():
            if not dom in self.symbols and not dom in self.regionBoxes.enumDomains:
                expr = self(dom, COUNT_PROP) * expr
        sums = []
        for dom in self.regionBoxes.regionsMap.iterkeys():
            if dom in self.symbols:
                sums.append((self.symbols[dom],
                             0, self(dom, COUNT_PROP) - 1))
        if len(sums) > 0:
            expr = Sum(expr, *sums).doit()
        return expr

    def domains(self):
        return self.domainProps.keys()

    def regions(self, dom):
        """Generates the properties of all regions of the given domain that
        are contained in this region box."""
        assert dom in self.domainProps
        if dom in self.regionBoxes.enumDomains:
            return iter([self.domainProps[dom]])
        else:
            return self.regionBoxes.regionsMap[dom].regions(self)

    def _cheapRegionBounds(self, dom):
        if dom in self.regionBoxes.enumDomains:
            return iter([(self.domainProps[dom][OFFSET_PROP],
                          self.domainProps[dom][SIZE_PROP])])
        else:
            return self.regionBoxes.regionsMap[dom]._cheapRegionBounds(self)

    def regionsEqual(self, rbox, dom):
        """Check whether we have the same regions for the given domain. This
        is a cheap heuristic: We might return False even though the
        regions are actually equal.
        """
        assert dom in self.domainProps and dom in rbox.domainProps
        if dom in self.regionBoxes.enumDomains:
            # Only one to compare - fall back to direct comparison
            return False
            return [self.domainProps[dom]] == list(rbox.regions(dom))
        else:
            return self.regionBoxes.regionsMap[dom].regionsEqual(
                self, rbox.regionBoxes.regionsMap[dom], rbox)

    def _edgeMult(lbox, rbox, dom):
        """Returns the edge multiplier between two region boxes concerning a
        domain. This is basically answering the question - if we have
        one domain that is split in two ways, how many edges would we
        have between two sub-sets of these splits?
        """

        # Domain enumerated on both sides? Then there's either one
        # edge or there is none. Check match.
        left = lbox.regionBoxes
        right = rbox.regionBoxes
        if dom in left.enumDomains and dom in right.enumDomains:
            # Must match so we don't zip both sides
            # all-to-all. If we find that the two boxes don't
            # match, we conclude that this box combination is
            # invalid and bail out completely.
            loff = lbox(dom, OFFSET_PROP)
            roff = rbox(dom, OFFSET_PROP)
            if loff < roff:
                if roff >= loff + lbox(dom, SIZE_PROP):
                    return 0
            else:
                if loff >= roff + rbox(dom, SIZE_PROP):
                    return 0
            return 1
        # Enumerated on just one side? Assuming equal spacing
        # on the other side, calculate how many region starts
        # on the other side fall into the current region.
        if dom in left.enumDomains:
            loff = lbox(dom, OFFSET_PROP)
            lsize = lbox(dom, SIZE_PROP)
            rsize = rbox(dom, SIZE_PROP)
            rcount = rbox(dom, COUNT_PROP)
            def bound(x): return min(rcount, max(0, floor(x)))
            return 1 + bound((loff + lsize - 1) / rsize) - bound(loff / rsize)
        if dom in right.enumDomains:
            lsize = lbox(dom, SIZE_PROP)
            lcount = lbox(dom, COUNT_PROP)
            roff = rbox(dom, OFFSET_PROP)
            rsize = rbox(dom, SIZE_PROP)
            def bound(x): return min(lcount, max(0, floor(x)))
            return 1 + bound((roff + rsize - 1) / lsize) - bound(roff / lsize)
        # Otherwise, the higher granularity gives the number
        # of edges.
        lcount = lbox(dom, COUNT_PROP)
        rcount = rbox(dom, COUNT_PROP)
        import math
        if lcount < rcount:
            return lcount * ((rcount + lcount - 1) / lcount)
        else:
            return rcount * ((lcount + rcount - 1) / rcount)

    def _zipSum(lbox, rbox, commonDoms, leftDoms, rightDoms, f):
        """Return sum of function, applied to all pairs of edges between
        individual region boxes between lbox and rbox.
        """

        # Match common domains
        right = rbox.regionBoxes
        mult = 1
        for dom in commonDoms:
            m = lbox._edgeMult(rbox, dom)
            mult *= m
            if mult == 0: return 0

        # Domains on only one side: Simply multiply out (where
        # un-enumerated)
        for dom in leftDoms:
            if not dom in lbox.regionBoxes.enumDomains:
                mult *= lbox(dom, COUNT_PROP)
        for dom in rightDoms:
            if not dom in rbox.regionBoxes.enumDomains:
                mult *= rbox(dom, COUNT_PROP)

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
            if not dom in cbox.regionBoxes.enumDomains:
                return lbox._edgeMult(rbox, dom)
        # In contrast to _edgeMult, we must go through all individual
        # regions separately. Slightly less efficient and more work,
        # but on the other hand this means that we do not need to care
        # about enumeration in the first place.

        # Set up region iterators
        lregs = lbox._cheapRegionBounds(dom)
        rregs = rbox._cheapRegionBounds(dom)
        cregs = cbox._cheapRegionBounds(dom)
        try:
            (loff, lsize) = lregs.next()
            (roff, rsize) = rregs.next()
            (coff, csize) = cregs.next()
        except StopIteration:
            return 0

        count = 0
        while True:

            # Advance iterators until there is at least some overlap
            try:
                while loff + lsize < roff:
                    (loff, lsize) = lregs.next()
                while roff + rsize < loff:
                    (roff, rsize) = rregs.next()
                while coff + csize <= loff or coff + csize <= roff:
                    (coff, csize) = cregs.next()
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
                    (loff, lsize) = lregs.next()
                else:
                    (roff, rsize) = rregs.next()
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
            lreg = lregs.next()
            creg = cregs.next()
        except StopIteration:
            return 0

        count = 0
        loff = lreg[OFFSET_PROP]; lsize = lreg[SIZE_PROP]
        coff = creg[OFFSET_PROP]; csize = creg[SIZE_PROP]
        while True:

            # Advance cross iterator
            try:
                while coff + csize < loff + lsize:
                    creg = cregs.next()
                    coff = creg[OFFSET_PROP]; csize = creg[SIZE_PROP]
            except StopIteration:
                break

            # Is our current edge crossing?
            if loff >= coff and loff < coff + csize:
                count += 1

            # Advance iterator
            try:
                lreg = lregs.next()
                loff = lreg[OFFSET_PROP]; lsize = lreg[SIZE_PROP]
            except StopIteration:
                break

        return count

    def _zipCrossSum(lbox, rbox, cbox, commonDoms, leftDoms, rightDoms, f):
        """Return sum of function, applied to all pairs of edges between
        individual region boxes between lbox and rbox that start in a
        different region of the cross domain than they end in.
        """

        # Match common domains
        right = rbox.regionBoxes
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
            else:
                cm = 0
            mults.append((m,cm))

        # Domains on only one side: Edge multiplier is given by region
        # count. Edge cross multiplier depends - if the domain in
        # question is not one we track edge crossings for, we
        # obviously have no such crossings. Otherwise, all edges
        # cross.
        for dom in leftDoms:
            if dom in cross_doms:
                m = cm = lbox._edgeCrossMultOneSided(cbox, dom)
            else:
                if dom in lbox.regionBoxes.enumDomains:
                    m = 1
                else:
                    m = lbox(dom, COUNT_PROP)
                cm = 0
            mults.append((m,cm))
        for dom in rightDoms:
            if dom in cross_doms:
                m = cm = rbox._edgeCrossMultOneSided(cbox, dom)
            else:
                if dom in rbox.regionBoxes.enumDomains:
                    m = 1
                else:
                    m = rbox(dom, COUNT_PROP)
                cm = 0
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
            if self.regionsMap.has_key(regions.domain):
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
            if not self.regionsMap.has_key(e.domain):
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

    def regions(self, dom):
        """Returns all regions we see for the given domain. Note that regions
        might be duplicated due to enumeration. This is the right
        behaviour, as different region boxes might have different
        regions for this domain.
        """
        assert dom in self.regionsMap
        from itertools import chain
        it = iter([])
        for box in self.boxes:
            it = chain(it, box.regions(dom))
        return it

    def _cheapRegionBounds(self, dom):
        assert dom in self.regionsMap
        from itertools import chain
        it = iter([])
        for box in self.boxes:
            it = chain(it, box._cheapRegionBounds(dom))
        return it

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

    def zipSum(left, right, f):
        return left._withEnums(lambda: right._withEnums(
            lambda: left._zipSum(right, f)))
    def _zipSum(left, right, f):
        """Return sum of function, applied to all edges from one region box
        set to another. An edge exists between two region boxes if all
        domains that exist on both sides agree. Agreement means 
        """

        # Classify domains
        leftDoms = []
        rightDoms = []
        commonDoms = []
        for dom in left.regionsMap.iterkeys():
            if right.regionsMap.has_key(dom):
                commonDoms.append(dom)
            else:
                leftDoms.append(dom)
        for dom in right.regionsMap.iterkeys():
            if not left.regionsMap.has_key(dom):
                rightDoms.append(dom)

        # Sum up result of zip
        result = 0
        for lbox in left.boxes:
            for rbox in right.boxes:
                result += lbox._zipSum(rbox, commonDoms, leftDoms, rightDoms, f)
        return result

    def zipCrossSum(left, right, cross, f):
        return left._withEnums(lambda: right._withEnums(
            lambda: left._zipCrossSum(right, cross, f)))
    def _zipCrossSum(left, right, cross, f):

        # Classify domains
        leftDoms = []
        rightDoms = []
        commonDoms = []
        for dom in left.regionsMap.iterkeys():
            if right.regionsMap.has_key(dom):
                commonDoms.append(dom)
            else:
                leftDoms.append(dom)
        for dom in right.regionsMap.iterkeys():
            if not left.regionsMap.has_key(dom):
                rightDoms.append(dom)

        # Sum up result of zip
        result = 0
        for cbox in cross.boxes:
            for lbox in left.boxes:
                for rbox in right.boxes:
                    result += lbox._zipCrossSum(rbox, cbox, commonDoms, leftDoms, rightDoms, f)
        return result

class Flow:
    """A flow is a set of results, produced for the cartesian product of a
    number of region sets. Producing these results will have costs
    associated with it - e.g. 'compute' costs to produce the result in
    the first place or 'transfer' cost for the size of the result when
    transferred to other nodes.
    """

    def __init__(self, name, regss = [], costs = {}, attrs = {}, deps = [],
                 cluster=''):
        self.name = name
        self.boxes = RegionBoxes(regss)
        self.costs = costs
        self.attrs = attrs
        self.deps = map(lambda d: (d,1), deps)
        self.cluster = cluster

    def depend(self, flow, weight=1):
        self.deps.append((flow, weight))

    def output(self, name, costs, attrs={}):
        """Make a new Flow for an output of this Flow. Useful when a Flow has
        multiple outputs that have different associated costs."""
        return Flow(name, self.regionsBox, costs=costs, attrs=attrs)

    def count(self):
        return self.boxes.count()

    def the(self, prop):
        return self.boxes.the(prop)

    def sum(self, prop):
        return self.boxes.sum(prop)

    def max(self, prop):
        return self.boxes.max(prop)

    def cost(self, name):
        if self.costs.has_key(name):
            return self.sum(self.costs[name])
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
            for dep, _ in node.deps:
                if not dep in recDeps:
                    active.append(dep)
                    recDeps.append(dep)

        return list(recDeps)

def flowsToDot(flows, t, computeSpeed=None,
               graph_attr={}, node_attr={'shape':'box'}, edge_attr={},
               crossRegs=None):

    # Make digraph
    dot = Digraph(graph_attr=graph_attr,node_attr=node_attr,edge_attr=edge_attr)

    # Assign IDs to flows, collect clusters
    flowIds = {}
    clusters = {}
    for i, flow in enumerate(flows):
        flowIds[flow] = "node.%d" % i
        if clusters.has_key(flow.cluster):
            clusters[flow.cluster].append(flow)
        else:
            clusters[flow.cluster] = [flow]

    # Make boxes for cross regions
    crossBoxes = None
    if not crossRegs is None:
        crossBoxes = RegionBoxes(crossRegs)

    # Make nodes per cluster
    for cluster, cflows in clusters.iteritems():

        if cluster == '':
            graph = dot
        else:
            graph = Digraph(name="cluster_"+cluster,
                            graph_attr={'label':cluster})

        for flow in cflows:

            # Start with flow name
            text = flow.name

            # Add relevant regions
            for region in flow.regions():
                count = flow.max(lambda rb: rb(region.domain, 'count'))
                size = flow.max(lambda rb: rb(region.domain, 'size'))
                if count == 1 and size == 1:
                    continue
                text += "\n%s: %d x %g %s" % \
                        (region.domain.name,
                         flow.max(lambda rb: rb(region.domain, 'count')),
                         flow.max(lambda rb: rb(region.domain, 'size')),
                         region.domain.unit)

            # Add compute count
            count = flow.count()
            if count != 1:
                text += "\nTasks: %d 1/s" % (count/t)
            try:
                compute = flow.cost('compute')
                transfer = flow.cost('transfer')
            except:
                print
                print "Exception raised while determining cost for '" + flow.name + "':"
                raise
            if compute > 0 and computeSpeed is not None:
                text += "\nTime: %.2g s/task" % (compute/count/computeSpeed)
            if compute > 0:
                text += "\nFLOPs: %.2f TOP/s" % (compute/t/Constants.tera)
            if transfer > 0:
                text += "\nOutput: %.2f TB/s" % (transfer/t/Constants.tera)

            attrs = flow.attrs
            graph.node(flowIds[flow], text, attrs)

            # Add dependencies
            for dep, weight in flow.deps:
                if flowIds.has_key(dep):

                    # Calculate number of edges, and use node counts
                    # on both sides to calculate (average!) in and out
                    # degrees.
                    edges = dep.boxes.zipSum(flow.boxes, lambda l, r: 1)
                    if dep.costs.has_key('transfer'):
                        def transfer_prop(l,r): return mk_lambda(dep.costs['transfer'])(l)
                        transfer = dep.boxes.zipSum(flow.boxes, transfer_prop)
                    depcount = dep.boxes.count()
                    flowcount = flow.boxes.count()
                    def format_bytes(n):
                        if n < 10**9: return '%.1f MB' % (n/10**6)
                        return '%.1f GB' % (n/10**9)
                    if dep.costs.has_key('transfer') and transfer > 0:
                        label = 'out: %d (%s)\nin: %d (%s)' % \
                                (edges/depcount, format_bytes(transfer/depcount),
                                 edges/flowcount, format_bytes(transfer/flowcount))
                    else:
                        label = 'out: %d\nin: %d' % (edges/depcount, edges/flowcount)

                    dot.edge(flowIds[dep], flowIds[flow], label, weight=str(weight))

                    #if crossBoxes:
                    #    print flow.name, dep.name
                    #    print dep.boxes.zipCrossSum(flow.boxes, crossBoxes, lambda l, r: 1)


        if cluster != '':
            dot.subgraph(graph)

    return dot

class DataFlowTests(unittest.TestCase):

    def setUp(self):

        # Set up a domain with four different splits - all, each, split
        # into 4 regions and split into 3 regions.
        self.size1 = 12
        self.dom1 = Domain("Domain 1")
        props = {'dummy': lambda rb: 0 }
        self.reg1 = self.dom1.regions(self.size1, props=props)
        self.split1 = self.reg1.split(self.size1, props=props)
        self.size1a = 4
        self.split1a = self.reg1.split(self.size1a, props=props)
        self.size1b = 3
        self.split1b = self.reg1.split(self.size1b, props=props)

        # Assumed by tests
        assert self.size1a > self.size1b and self.size1a < 2*self.size1b

        # Second domain, with splits "all" and "each".
        self.size2 = 7
        self.dom2 = Domain("Domain 2")
        self.reg2 = self.dom2.regions(self.size2, props=props)
        self.split2 = self.reg2.split(self.size2, props=props)

    def test_size(self):

        # Expected region properties
        self.assertEqual(self.reg1.size, self.size1)
        self.assertEqual(self.split1.size, self.size1)
        self.assertEqual(self.split1a.size, self.size1)
        self.assertEqual(self.split1b.size, self.size1)
        def countProp(rb): return rb(self.dom1, COUNT_PROP)
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

        # Summing up the box sizes should give us the region size
        def sizeProp(rb): return rb(self.dom1, SIZE_PROP)
        self.assertEqual(rboxes.the(sizeProp), boxSize)
        self.assertEqual(rboxes.sum(sizeProp), self.size1)

        # Should all still work when we force the boxes to be
        # enumerated by depending on the dummy property
        def dummySizeProp(rb): return rb(self.dom1, 'dummy') + rb(self.dom1, SIZE_PROP)
        self.assertFalse(self.dom1 in rboxes.enumDomains)
        self.assertEqual(rboxes.the(dummySizeProp), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp), self.size1)
        self.assertTrue(self.dom1 in rboxes.enumDomains)

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
        def sizeProp(rb): return rb(self.dom1, SIZE_PROP) * rb(self.dom2, SIZE_PROP)
        self.assertEqual(rboxes.the(sizeProp), boxSize)
        self.assertEqual(rboxes.sum(sizeProp), self.size1*self.size2)
        def dummySizeProp1(rb): return rb(self.dom1, 'dummy') + sizeProp(rb)
        self.assertEqual(rboxes.the(dummySizeProp1), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp1), self.size1*self.size2)
        rboxes = RegionBoxes(regss)
        def dummySizeProp2(rb): return rb(self.dom2, 'dummy') + sizeProp(rb)
        self.assertEqual(rboxes.the(dummySizeProp2), boxSize)
        self.assertEqual(rboxes.sum(dummySizeProp2), self.size1*self.size2)
        rboxes = RegionBoxes(regss)
        def dummySizeProp12(rb): return rb(self.dom1, 'dummy') + rb(self.dom2, 'dummy') + sizeProp(rb)
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

    def _test_rboxes_zip(self, regsA, regsB, domA, domB, edges):
        rboxesA = RegionBoxes(regsA)
        rboxesB = RegionBoxes(regsB)

        # Summing one per edge should give us the number of
        # edges. Enumerating shouldn't make a difference, as before.
        def oneProp(rbA, rbB): return 1
        self.assertEqual(rboxesA.zipSum(rboxesB, oneProp), edges)
        def onePropA(rbA, rbB): return 1 + rbA(domA, 'dummy')
        self.assertEqual(rboxesA.zipSum(rboxesB, onePropA), edges)
        self.assertTrue(domA in rboxesA.enumDomains)
        self.assertFalse(domA in rboxesB.enumDomains)
        rboxesA = RegionBoxes(regsA)
        rboxesB = RegionBoxes(regsB)
        def onePropB(rbA, rbB): return 1 + rbB(domB, 'dummy')
        self.assertEqual(rboxesA.zipSum(rboxesB, onePropB), edges)
        self.assertFalse(domB in rboxesA.enumDomains)
        self.assertTrue(domB in rboxesB.enumDomains)
        def onePropAB(rbA, rbB): return 1 + rbA(domA, 'dummy') + rbB(domB, 'dummy')
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
        self._test_rboxes_cross_zip([reg2],   [reg2],    [reg1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [reg2], 0)
        self._test_rboxes_cross_zip([reg2],   [reg2],    [reg1], 0)
        self._test_rboxes_cross_zip([reg1],   [reg1],    [reg2], 0)
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
        self._test_rboxes_cross_zip([split1,reg2],  [reg1],       [reg1, split2], 0)
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
            def oneProp(dom): return lambda rb: 1 + rb(dom, 'dummy')
            if enumA: rboxesA.sum(oneProp(domA))
            if enumB: rboxesB.sum(oneProp(domB))
            if enumC: rboxesC.sum(oneProp(domC))

            # Check that _cheapRegionBounds is consistent, as
            # zipCrossSum will depend on it heavily
            def toOffSize(reg): return (reg[OFFSET_PROP], reg[SIZE_PROP])
            self.assertEqual(map(toOffSize, rboxesA.regions(domA)), list(rboxesA._cheapRegionBounds(domA)))
            self.assertEqual(map(toOffSize, rboxesB.regions(domB)), list(rboxesB._cheapRegionBounds(domB)))
            self.assertEqual(map(toOffSize, rboxesC.regions(domC)), list(rboxesC._cheapRegionBounds(domC)))

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
        self.assertEqual(rbox.sum(lambda rb: rb(dom, 'p1')), size)
        self.assertEqual(rbox.sum(lambda rb: rb(dom, 'p2')), size)
        self.assertEqual(rbox.sum(lambda rb: rb(dom, 'p3')), size**2/2-size/2)

if __name__ == '__main__':
    unittest.main()
