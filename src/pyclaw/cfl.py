r"""
Module for the CFL object, which is responsible for computing the CFL number.
This is the serial version, which does almost nothing.  The structure is designed
to accommodate the parallel version (petclaw.CFL).
"""

class CFL(object):
    def __init__(self, global_max):
        self._global_max = global_max
        
    def get_global_max(self):
        return self._global_max

    def get_cached_max(self):
        return self._global_max

    def set_local_max(self,new_local_max):
        self._global_max = new_local_max

    def update_global_max(self,new_local_max):
        self._global_max = new_local_max

