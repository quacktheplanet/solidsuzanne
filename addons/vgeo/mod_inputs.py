"""A Geometry Nodes modifier's input values, the same way on every supported Blender.

Up to 5.1 they are ID properties on the modifier: mod["Socket_2"], plus mod["Socket_2_use_attribute"]
and mod["Socket_2_attribute_name"] for inputs that can take an attribute. 5.2 moved them to RNA
(mod.properties.inputs.Socket_2.value / .type / .attribute_name), and mod[...] raises TypeError.

    of(mod)[ident] = value        of(mod).get(ident)        ident in of(mod)        of(mod).keys()

On 5.1 and earlier of(mod) is the modifier itself; on 5.2 it is a stand-in taking the old keys.
"""

from __future__ import annotations

_SUFFIXES = (("_use_attribute", "type"), ("_attribute_name", "attribute_name"))


def of(mod):
    if not hasattr(mod, "properties"):
        return mod
    return _Inputs(mod.properties.inputs)


class _Inputs:
    def __init__(self, inputs):
        self._inputs = inputs

    def _idents(self):
        return [p.identifier for p in self._inputs.bl_rna.properties if p.identifier not in ("rna_type", "name")]

    def _find(self, key):
        """(input, attribute) for an old-style key, or (None, None)."""
        ident, attr = key, "value"
        for suffix, a in _SUFFIXES:
            if key.endswith(suffix):
                ident, attr = key[:-len(suffix)], a
                break
        if ident not in self._idents():
            return None, None
        item = getattr(self._inputs, ident)
        if not hasattr(item, attr):
            return None, None
        return item, attr

    def __getitem__(self, key):
        item, attr = self._find(key)
        if item is None:
            raise KeyError(key)
        value = getattr(item, attr)
        return value == 'ATTRIBUTE' if attr == "type" else value

    def __setitem__(self, key, value):
        item, attr = self._find(key)
        if item is None:
            raise KeyError(key)
        if attr == "type":
            value = 'ATTRIBUTE' if value else 'VALUE'
        setattr(item, attr, value)

    def __contains__(self, key):
        return self._find(key)[0] is not None

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        out = []
        for ident in self._idents():
            item = getattr(self._inputs, ident)
            if hasattr(item, "value"):
                out.append(ident)
            out.extend(ident + suffix for suffix, attr in _SUFFIXES if hasattr(item, attr))
        return out
