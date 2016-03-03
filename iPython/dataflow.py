
from parameter_definitions import Constants

from sympy import Max, Expr
from graphviz import Digraph

INDEX_PROP = 'index'
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

    def getProps(self, rb, i = None):
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

    def sum(self, prop):
        # Defer to RegionBoxes, which will care about enumerating the
        # regions if required.
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
    def __init__(self, domain):
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
            raise NeedRegionEnumerationException(domain)

        # If we don't have the property, this might also point towards
        # the need for enumeration.
        if not self.domainProps[domain].has_key(prop) and \
           not domain in self.regionBoxes.enumDomains:
            raise NeedRegionEnumerationException(domain)

        # Look it up
        return self.domainProps[domain][prop]

    def count(self):
        count = 1
        # Multiply out all non-enumerated (!) regions
        for dom in self.regionBoxes.regions.iterkeys():
            if not dom in self.regionBoxes.enumDomains:
                count *= self(dom, COUNT_PROP)
        return count

    def sum(self, prop):
        return self.count() * mk_lambda(prop)(self)

    def max(self, prop):
        return mk_lambda(prop)(self)

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
                for i in range(regs.count(rbox)):
                    props = regs.getProps(rbox, i)
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
        return self._withEnums(lambda: self._count())
    def _count(self):
        count = 0
        for box in self.boxes:
            count += box.count()
        return count

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
                s = box.max(prop)
            else:
                s = Max(s, box.max(prop))
        return s

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
        self.deps = list(deps)
        self.cluster = cluster

    def depend(self, flow):
        self.deps.append(flow)

    def output(self, name, costs, attrs={}):
        """Make a new Flow for an output of this Flow. Useful when a Flow has
        multiple outputs that have different associated costs."""
        return Flow(name, self.regionsBox, costs=costs, attrs=attrs)

    def count(self):
        return self.boxes.count()

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
            for dep in node.deps:
                if not dep in recDeps:
                    active.append(dep)
                    recDeps.append(dep)

        return list(recDeps)

def flowsToDot(flows, t, graph_attr={}, node_attr={'shape':'box'}, edge_attr={}):

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
            compute = flow.cost('compute')
            if compute > 0:
                text += "\nFLOPs: %.2f POP/s" % (compute/t/Constants.peta)

            attrs = flow.attrs
            graph.node(flowIds[flow], text, attrs)

            # Add dependencies
            for dep in flow.deps:
                if flowIds.has_key(dep):
                    dot.edge(flowIds[dep], flowIds[flow])

        if cluster != '':
            dot.subgraph(graph)

    return dot
