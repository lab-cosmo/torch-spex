"""Parsing of spec dictionaries.

Directly taken from https://github.com/sirmarcel/specable

The spec dictionary format is:

```
{handle: payload}
```

"handle" is a string, identifies the class to be instantiated
"payload" is a a mapping (normally dict), specifying how this class is created

"""

from collections.abc import Mapping


def to_dict(handle, payload):
    return {handle: payload}


def is_valid(dct):
    if isinstance(dct, Mapping):
        if len(dct) == 1:
            handle = next(iter(dct))
            if isinstance(handle, str):
                if isinstance(dct[handle], Mapping):
                    return True

    return False


def parse_dict(dct, allow_stubs=False):
    if allow_stubs and isinstance(dct, str):
        return dct, {}

    if not is_valid(dct):
        raise ValueError("Improper spec dict format: " + str(dct))

    handle = next(iter(dct))
    inner = dct[handle]

    return handle, inner
