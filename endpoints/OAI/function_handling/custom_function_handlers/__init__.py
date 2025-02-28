import importlib
import pkgutil

from endpoints.OAI.function_handling.default_function_handler import FunctionCallingBaseClass

# Path to the Foo directory
package_name = "endpoints.OAI.function_handling.custom_function_handlers"

# Import all modules in Foo
for _, module_name, _ in pkgutil.iter_modules([package_name]):
    _ = importlib.import_module(f"{package_name}.{module_name}")

# Get all subclasses of Base
subclasses = {cls.__name__: cls for cls in FunctionCallingBaseClass.__subclasses__()}
