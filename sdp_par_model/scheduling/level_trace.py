
from sortedcontainers import SortedDict
import operator

class LevelTrace(object):
    """ Traces the level of some entity across a time span """

    def __init__(self, trace=None):
        """ Creates a new level trace, possibly copying from an existing object. """

        if trace is None:
            self._trace = SortedDict()
        elif isinstance(trace, LevelTrace):
            self._trace = SortedDict(trace._trace)
        else:
            self._trace = SortedDict(trace)

        # Make sure trace is terminated (returns to 0)
        if len(self._trace) > 0 and self._trace[self._trace.keys()[-1]] != 0:
            raise ValueError(
                "Trace not terminated - ends with {}:{}!".format(
                    self._trace.keys()[-1], self._trace[self._trace.keys()[-1]])
                )

    def __repr__(self):
        items = ', '.join(["{!r}: {!r}".format(k, v)
                           for k, v in self._trace.items()])
        return "LevelTrace({{{}}})".format(items)

    def __eq__(self, other):
        return self._trace == other._trace

    def __neg__(self):
        return self.map(operator.neg)
    def __sub__(self, other):
        return self.zip_with(other, operator.sub)
    def __add__(self, other):
        return self.zip_with(other, operator.add)

    def start(self):
        return self._trace.keys()[0]

    def end(self):
        return self._trace.keys()[-1]

    def length(self):
        if len(self._trace) == 0:
            return 0
        return self.end() - self.start()

    def get(self, time):
        ix = self._trace.bisect_right(time) - 1
        if ix < 0:
            return 0
        else:
            (_, lvl) = self._trace.peekitem(ix)
            return lvl

    def add(self, start, end, amount):
        """ Increases the level for some time range
        :param start: Start of range
        :param end: End of range
        :aram amount: Amount to add to level
        """

        # Check errors, no-ops
        if start > end:
            raise ValueError("End needs to be after start!")
        if start == end or amount == 0:
            return

        # Determine levels at start (and before start)
        start_ix = self._trace.bisect_right(start) - 1
        prev_lvl = lvl = 0
        if start_ix >= 0:
            (t, lvl) = self._trace.peekitem(start_ix)
            # If we have no entry exactly at our start point, the
            # level was constant at this point before
            if start > t:
                prev_lvl = lvl
            # Otherwise look up previous level. Default 0 (see above)
            elif start_ix > 0:
                (_, prev_lvl) = self._trace.peekitem(start_ix-1)

        # Prepare start
        if prev_lvl == lvl + amount:
            del self._trace[start]
        else:
            self._trace[start] = lvl + amount

        # Update all in-between states
        insert_start = True
        last_level = None
        for time in self._trace.irange(start, end, inclusive=(False, False)):
            if time == start:
                insert_start = False
            lvl = self._trace[time]
            self._trace[time] = lvl + amount

        # Add or remove end, if necessary
        if end not in self._trace:
            self._trace[end] = lvl
        elif lvl + amount == self._trace[end]:
            del self._trace[end]

    def foldl1(self, start, end, fn):
        """
        Does a left-fold over the levels present in the given range. Seeds
        with level at start.
        """

        if start > end:
            raise ValueError("End needs to be after start!")
        val = self.get(start)
        start_ix = self._trace.bisect_right(start)
        end_ix = self._trace.bisect_left(end)
        for lvl in self._trace.values()[start_ix:end_ix]:
            val = fn(val, lvl)
        return val

    def minimum(self, start, end):
        """ Returns the lowest level in the given range """
        return self.foldl1(start, end, min)
    def maximum(self, start, end):
        """ Returns the highest level in the given range """
        return self.foldl1(start, end, max)

    def foldl_time(self, start, end, val, fn):
        """
        Does a left-fold over the levels present in the given range,
        also passing how long the level was held. Seed passed.
        """

        if start > end:
            raise ValueError("End needs to be after start!")

        last_time = start
        last_lvl = self.get(start)

        start_ix = self._trace.bisect_right(start)
        end_ix = self._trace.bisect_left(end)
        for time, lvl in self._trace.items()[start_ix:end_ix]:
            val = fn(val, time-last_time, last_lvl)
            last_time = time
            last_lvl = lvl

        return fn(val, end-last_time, last_lvl)

    def integrate(self, start, end):
        """ Returns the integral over a range (sum below level curve) """
        return self.foldl_time(start, end, 0,
                               lambda v, time, lvl: v + time * lvl)
    def average(self, start, end):
        """ Returns the average level over a given range """
        return self.integrate(start, end) / (end - start)

    def find_above(self, time, level):
        """Returns the first time larger or equal to the given start time
        where the level is at least the specified value.
        """

        if self.get(time) >= level:
            return time
        ix = self._trace.bisect_right(time)
        for t, lvl in self._trace.items()[ix:]:
            if lvl >= level:
                return t
        return None

    def find_below(self, time, level):
        """Returns the first time larger or equal to the given start time
        where the level is less or equal the specified value.
        """

        if self.get(time) <= level:
            return time
        ix = self._trace.bisect_right(time)
        for t, lvl in self._trace.items()[ix:]:
            if lvl <= level:
                return t
        return None

    def find_below_backward(self, time, level):
        """Returns the last time smaller or equal to the given time where
        there exists a region to the left where the level is below the
        given value.
        """

        last = time
        ix = self._trace.bisect_right(time)-1
        if ix >= 0:
            for t, lvl in self._trace.items()[ix::-1]:
                if lvl <= level and time > t:
                    return last
                last = t
        if level >= 0:
            return last
        return None

    def find_above_backward(self, time, level):
        """Returns the last time smaller or equal to the given time where
        there exists a region to the left where the level is below the
        given value.
        """

        last = time
        ix = self._trace.bisect_right(time)-1
        if ix >= 0:
            for t, lvl in self._trace.items()[ix::-1]:
                if lvl >= level and time > t:
                    return last
                last = t
        if level <= 0:
            return last
        return None

    def find_period_below(self, start, end, target, length):
        """Returns a period where the level is below the target for a certain
        length of time, within a given start and end time"""

        if start > end:
            raise ValueError("End needs to be after start!")
        if length < 0:
            raise ValueError("Period length must be larger than zero!")

        period_start = (start if self.get(start) <= target else None)

        start_ix = self._trace.bisect_right(start)
        end_ix = self._trace.bisect_left(end)
        for time, lvl in self._trace.items()[start_ix:end_ix]:
            # Period long enough?
            if period_start is not None:
                if time >= period_start + length:
                    return period_start
            # Not enough space until end?
            elif time + length > end:
               return None
            # Above target? Reset period
            if lvl > target:
                period_start = None
            else:
                if period_start is None:
                    period_start = time

        # Possible at end?
        if period_start is not None and period_start+length <= end:
            return period_start

        # Nothing found
        return None

    def map(self, fn):

        return LevelTrace({t: fn(v) for t, v in self._trace.items()})

    def zip_with(self, other, fn):

        # Simple cases
        if len(self._trace) == 0:
            return other.map(lambda x: fn(0, x))
        if len(other._trace) == 0:
            return self.map(lambda x: fn(x, 0))

        # Read first item from both sides
        left = self._trace.items()
        right = other._trace.items()
        left_ix = 0
        right_ix = 0
        left_val = 0
        right_val = 0
        last_val = 0

        trace = SortedDict()

        # Go through pairs
        while left_ix < len(left) and right_ix < len(right):

            # Next items
            lt,lv = left[left_ix]
            rt,rv = right[right_ix]

            # Determine what to do
            if lt < rt:
                v = fn(lv, right_val)
                if v != last_val:
                    last_val = trace[lt] = v
                left_val = lv
                left_ix += 1
            elif lt > rt:
                v = fn(left_val, rv)
                if v != last_val:
                    last_val = trace[rt] = v
                right_val = rv
                right_ix += 1
            else:
                v = fn(lv, rv)
                if v != last_val:
                    last_val = trace[lt] = v
                left_val = lv
                left_ix += 1
                right_val = rv
                right_ix += 1

        # Handle left-overs
        while left_ix < len(left):
            lt,lv = left[left_ix]
            v = fn(lv, right_val)
            if v != last_val:
                last_val = trace[lt] = v
            left_ix += 1
        while right_ix < len(right):
            rt,rv = right[right_ix]
            v = fn(left_val, rv)
            if v != last_val:
                last_val = trace[rt] = v
            right_ix += 1

        return LevelTrace(trace)
