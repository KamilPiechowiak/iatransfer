import pkgutil


__all__ = []

# TODO:
# it would be nice to be able to import IAT from here
# importing it like: from iatransfer.toolkit import IAT would also cause it to import all of the required packages
# however, this doesn't seem to work
# refer to toolkit.test_iat error logs
# also, if we leave module_name without full path, any time we import something from here,
# the namespace seems to get polluted
# like:
# from iatransfer.toolkit import IAT
# import standardize # doesn't return any errors

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    full_module_name = f'iatransfer.toolkit.{module_name}'
    __all__.append(full_module_name)
    _module = loader.find_module(full_module_name).load_module(full_module_name)
    globals()[full_module_name] = _module

from .iat import IAT
