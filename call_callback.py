import inspect
from collections.abc import Callable
from typing import Any

def call_callback(callback: Callable, **kwargs: dict[str,Any]):
    """
    For more flexibility with the callback only parameters wich are defined are passed to the callback.
    So e.g. instead of defining the callback for every possible argument, it ist now possible to define it solely for those of interest.

    Parameters
    ----------
    callback: The callback function of interest
    kwargs: All parameters deemed interesting for inspection

    """

    callback_signature = inspect.signature(callback)
    callback_keys = callback_signature.parameters.keys()
    filtered_kwargs = {key: kwargs[key] for key in callback_keys}
    callback(**filtered_kwargs)
