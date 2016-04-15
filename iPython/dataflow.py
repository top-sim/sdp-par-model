
from parameter_definitions import Constants

from sympy import Max, Expr
from graphviz import Digraph
from math import floor

import unittest

INDEX_PROP = 'index'
OFFSET_PROP = 'offset'
COUNT_PROP = 'count'
SIZE_PROP = 'size'

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

    def region(self, size, props = {}):
        """Make a new unsplit region set for the domain."""
        return Regions(self, size, 1, props)

class Regions:
    """A set of regions of a domain."""

    def __init__(self, domain, size, count, props):
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

    def getProps(self, rb, i = None, off = None):
        """Get properties of enumerated region of the region set"""

        # Only copy over detailed properties if our domain is
        # enumerated - they are allowed to depend on the concrete
        # region.
        if not i is None:
            reg = {
                k: mk_lambda(v)(i)
                for k, v in self.props.iteritems()
            }
            reg[INDEX_PROP] = i
            reg[OFFSET_PROP] = off
        else:
            reg = {}

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

    def __call__(self, domain, prop):

        # Not enumerated yet?
        if not self.domainProps.has_key(domain):

            # Some properties we can calculate without enumeration
            if prop == COUNT_PROP:
                return self.regionBoxes.regions[domain].count(self)

            # Region boxes have it marked as to enumerate, but we
            # don't have properties for it? That means there's a
            # cyclic dependency...
            if domain in self.regionBoxes.enumDomains:
                raise Exception('Cyclic dependency involving domain %s!' % domain.name)

            # Otherwise we need to request this region to get
            # enumerated...
            raise NeedRegionEnumerationException(self.regionBoxes, domain)

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
        return self.domainProps[domain][prop]

    def count(self):
        count = 1
        # Multiply out all non-enumerated (!) regions
        for dom in self.regionBoxes.regions.iterkeys():
            if not dom in self.regionBoxes.enumDomains:
                count *= self(dom, COUNT_PROP)
        return count

    def the(self, prop):
        """Return the value of the given property for this region box"""
        return mk_lambda(prop)(self)

    def sum(self, prop):
        """Return the sum of the given property for this region box"""
        return self.count() * mk_lambda(prop)(self)

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
        self.regions = {}
        for regions in regionsBox:
            if self.regions.has_key(regions.domain):
                raise Exception('Domain %s has two regions in the same '
                                'box, this is not allowed!' % regions.domain.name)
            self.regions[regions.domain] = regions

        # Try to construct a list with all region boxes we need to
        # enumerate. This will add enumDomains as a side-effect
        self.enumDomains = []
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

        except NeedRegionEnumerationException as e:

            # Rethrow if this is for a different region boxes
            if e.regionBoxes != self:
                raise e

            # Sanity-check the domain we are supposed to enumerate
            if not self.regions.has_key(e.domain):
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
            regs = self.regions[edom]
            for regProps in rboxProps:

                # Make incomplete region box for evaluation count
                # and getProps. Both of these could raise an
                # NeedRegionEnumerationException.
                rbox = RegionBox(self, regProps)
                off = 0 # TODO: regions start offset?
                for i in range(regs.count(rbox)):
                    props = regs.getProps(rbox, i, off)
                    off += props[SIZE_PROP]
                    regProps = dict(regProps)
                    regProps[edom] = props
                    rboxPropsNew.append(regProps)
            rboxProps = rboxPropsNew

        # Done. Now we take care of all non-enumerated
        # domains. Ideally this should just mean calling getProps
        # on each, but again this might rais
        # NeedRegionEnumerationException, which sends us back to
        # square one.
        rboxes = []
        for props in rboxProps:
            # Make yet another temporary region box
            rbox = RegionBox(self, props)
            for regs in self.regionsBox:
                if not regs.domain in self.enumDomains:
                    props[regs.domain] = regs.getProps(rbox)
            # Props have changed, construct the (finally)
            # completed region box
            rboxes.append(RegionBox(self, props))
        return rboxes

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

        # Classify domains
        leftDoms = []
        rightDoms = []
        commonDoms = []
        for dom in left.regions.iterkeys():
            if right.regions.has_key(dom):
                commonDoms.append(dom)
            else:
                leftDoms.append(dom)
        for dom in right.regions.iterkeys():
            if not left.regions.has_key(dom):
                rightDoms.append(dom)

        # Local function to zip boxes together (to reduce indent level
        # and be able to use return, see below)
        def zipBoxes(lbox, rbox):

            # Match common domains
            mult = 1
            for dom in commonDoms:

                # Enumerated on both sides? Check match
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
                # Enumerated on just one side? Assuming equal spacing
                # on the other side, calculate how many region starts
                # on the other side fall into the current region.
                elif dom in left.enumDomains:
                    loff = lbox(dom, OFFSET_PROP)
                    lsize = lbox(dom, SIZE_PROP)
                    rsize = rbox(dom, SIZE_PROP)
                    rcount = rbox(dom, COUNT_PROP)
                    def bound(x): return min(rcount, max(0, floor(x)))
                    mult *= Max(1, bound((loff + lsize) / rsize) - bound(loff / rsize))
                elif dom in right.enumDomains:
                    lsize = lbox(dom, SIZE_PROP)
                    lcount = lbox(dom, COUNT_PROP)
                    roff = rbox(dom, OFFSET_PROP)
                    rsize = rbox(dom, SIZE_PROP)
                    def bound(x): return min(lcount, max(0, floor(x)))
                    mult *= Max(1, bound((roff + rsize) / lsize) - bound(roff / lsize))
                # Otherwise, the higher granularity gives the number
                # of edges.
                else:
                    lcount = lbox(dom, COUNT_PROP)
                    rcount = rbox(dom, COUNT_PROP)
                    mult *= Max(lcount, rcount)

            # Domains on only one side: Simply multiply out (where
            # un-enumerated)
            for dom in leftDoms:
                if not dom in left.enumDomains:
                    mult *= lbox(dom, COUNT_PROP)
            for dom in rightDoms:
                if not dom in right.enumDomains:
                    mult *= rbox(dom, COUNT_PROP)

            # Okay, now call and multiply
            return mult * f(lbox, rbox)

        # Sum up result of zip
        result = 0
        for lbox in left.boxes:
            for rbox in right.boxes:
                result += zipBoxes(lbox, rbox)
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
               graph_attr={}, node_attr={'shape':'box'}, edge_attr={}):

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

        if cluster != '':
            dot.subgraph(graph)

    return dot

class DataFlowTests(unittest.TestCase):

    def setUp(self):

        self.size1 = 12
        self.dom1 = Domain("Domain 1")
        self.reg1 = self.dom1.region(self.size1, props={'dummy': 0})
        self.split1 = self.reg1.split(self.size1, props={'dummy': 0})
        self.split1a = self.reg1.split(self.size1 / 2, props={'dummy': 0})

        self.size2 = 7
        self.dom2 = Domain("Domain 2")
        self.reg2 = self.dom2.region(self.size2, props={'dummy': 0})
        self.split2 = self.reg2.split(self.size2, props={'dummy': 0})

    def test_size(self):

        # Expected region properties
        self.assertEqual(self.reg1.size, self.size1)
        self.assertEqual(self.split1.size, self.size1)
        self.assertEqual(self.split1a.size, self.size1)
        def countProp(rb): return rb(self.dom1, COUNT_PROP)
        self.assertEqual(self.reg1.the(countProp), 1)
        self.assertEqual(self.split1.the(countProp), self.size1)
        self.assertEqual(self.split1a.the(countProp), self.size1 / 2)

    def test_rboxes_1(self):
        self._test_rboxes_1(self.reg1, 1, self.size1)
        self._test_rboxes_1(self.split1, self.size1, 1)
        self._test_rboxes_1(self.split1a, self.size1/2, 2)

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
        self._test_rboxes_2([self.reg2, self.reg1], 1, self.size1 * self.size2)
        self._test_rboxes_2([self.split2, self.reg1], self.size2, self.size1)
        self._test_rboxes_2([self.reg2, self.split1], self.size1, self.size2)
        self._test_rboxes_2([self.split2, self.split1], self.size1 * self.size2, 1)

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
        self._test_rboxes_zip([self.reg1],   [self.split1a], self.dom1, self.dom1, self.size1 / 2)
        self._test_rboxes_zip([self.split1], [self.reg1],    self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1], [self.split1],  self.dom1, self.dom1, self.size1)
        self._test_rboxes_zip([self.split1], [self.split1a], self.dom1, self.dom1, self.size1)

        # Edges between regions of different domains grow quadradically
        self._test_rboxes_zip([self.reg1],   [self.reg2],   self.dom1, self.dom2, 1)
        self._test_rboxes_zip([self.reg1],   [self.split2], self.dom1, self.dom2, self.size2)
        self._test_rboxes_zip([self.split1], [self.reg2],   self.dom1, self.dom2, self.size1)
        self._test_rboxes_zip([self.split1], [self.split2], self.dom1, self.dom2, self.size1 * self.size2)
        self._test_rboxes_zip([self.split1a],[self.split2], self.dom1, self.dom2, self.size1 * self.size2 / 2)
        self._test_rboxes_zip([self.split2], [self.split1a],self.dom2, self.dom1, self.size1 * self.size2 / 2)
        self._test_rboxes_zip([self.reg2],   [self.split1a],self.dom2, self.dom1, self.size1 / 2)
        self._test_rboxes_zip([self.split1a],[self.reg2],   self.dom1, self.dom2, self.size1 / 2)

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
                              self.size1 * self.size2 / 2)
        self._test_rboxes_zip([self.split2, self.split1a],[self.split2], self.dom2, self.dom2,
                              self.size1 * self.size2 / 2)
        self._test_rboxes_zip([self.split1, self.split2], [self.split1a],self.dom1, self.dom1,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split1, self.split2], [self.split1a],self.dom2, self.dom1,
                              self.size1 * self.size2)
        self._test_rboxes_zip([self.split1, self.reg2],   [self.split1a],self.dom1, self.dom1,
                              self.size1)
        self._test_rboxes_zip([self.split1, self.reg2],   [self.split1a],self.dom2, self.dom1,
                              self.size1)
        self._test_rboxes_zip([self.split2, self.split1a],[self.reg2],   self.dom1, self.dom2,
                              self.size1 * self.size2 / 2)
        self._test_rboxes_zip([self.split2, self.split1a],[self.reg2],   self.dom2, self.dom2,
                              self.size1 * self.size2 / 2)

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

if __name__ == '__main__':
    unittest.main()
