import pkgutil
import importlib

# Get the package name dynamically
package_name = __name__

# Import all modules inside this package
for _, module_name, _ in pkgutil.iter_modules(__path__):
    _ = importlib.import_module(f"{package_name}.{module_name}")
