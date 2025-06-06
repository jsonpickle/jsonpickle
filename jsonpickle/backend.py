from types import ModuleType
from typing import Any, Dict, Optional, Type, Union


class JSONBackend:
    """Manages encoding and decoding using various backends.

    It tries these modules in this order:
        simplejson, json, ujson

    simplejson is a fast and popular backend and is tried first.
    json comes with Python and is tried second.

    """

    def _verify(self) -> None:
        """Ensures that we've loaded at least one JSON backend."""
        if self._verified:
            return
        raise AssertionError('jsonpickle could not load any json modules')

    def encode(
        self, obj: Any, indent: Optional[int] = None, separators: Optional[Any] = None
    ) -> str:
        """
        Attempt to encode an object into JSON.

        This tries the loaded backends in order and passes along the last
        exception if no backend is able to encode the object.

        """
        self._verify()

        if not self._fallthrough:
            name = self._backend_names[0]
            return self.backend_encode(name, obj, indent=indent, separators=separators)

        for idx, name in enumerate(self._backend_names):
            try:
                return self.backend_encode(
                    name, obj, indent=indent, separators=separators
                )
            except Exception as e:
                if idx == len(self._backend_names) - 1:
                    raise e

    # def dumps
    dumps = encode

    def decode(self, string: str) -> Any:
        """
        Attempt to decode an object from a JSON string.

        This tries the loaded backends in order and passes along the last
        exception if no backends are able to decode the string.

        """
        self._verify()

        if not self._fallthrough:
            name = self._backend_names[0]
            return self.backend_decode(name, string)

        for idx, name in enumerate(self._backend_names):
            try:
                return self.backend_decode(name, string)
            except self._decoder_exceptions[name] as e:
                if idx == len(self._backend_names) - 1:
                    raise e
                else:
                    pass  # and try a more forgiving encoder

    # def loads
    loads = decode

    def __init__(self, fallthrough: bool = True) -> None:
        # Whether we should fallthrough to the next backend
        self._fallthrough = fallthrough
        # The names of backends that have been successfully imported
        self._backend_names = []

        # A dictionary mapping backend names to encode/decode functions
        self._encoders = {}
        self._decoders = {}

        # Options to pass to specific encoders
        self._encoder_options = {}

        # Options to pass to specific decoders
        self._decoder_options = {}

        # The exception class that is thrown when a decoding error occurs
        self._decoder_exceptions = {}

        # Whether we've loaded any backends successfully
        self._verified = False

        self.load_backend('simplejson')
        self.load_backend('json')
        self.load_backend('ujson')

        # Defaults for various encoders
        json_opts = ((), {'sort_keys': False})
        self._encoder_options = {
            'ujson': ((), {'sort_keys': False, 'escape_forward_slashes': False}),
            'json': json_opts,
            'simplejson': json_opts,
            'django.util.simplejson': json_opts,
        }

    def enable_fallthrough(self, enable: bool) -> None:
        """
        Disable jsonpickle's fallthrough-on-error behavior

        By default, jsonpickle tries the next backend when decoding or
        encoding using a backend fails.

        This can make it difficult to force jsonpickle to use a specific
        backend, and catch errors, because the error will be suppressed and
        may not be raised by the subsequent backend.

        Calling `enable_backend(False)` will make jsonpickle immediately
        re-raise any exceptions raised by the backends.

        """
        self._fallthrough = enable

    def _store(self, dct: Dict[str, Any], backend: str, obj: ModuleType, name: str):
        try:
            dct[backend] = getattr(obj, name)
        except AttributeError:
            self.remove_backend(backend)
            return False
        return True

    def load_backend(
        self,
        name: str,
        dumps: str = 'dumps',
        loads: str = 'loads',
        loads_exc: Union[str, Type[Exception]] = ValueError,
    ) -> bool:
        """Load a JSON backend by name.

        This method loads a backend and sets up references to that
        backend's loads/dumps functions and exception classes.

        :param dumps: is the name of the backend's encode method.
          The method should take an object and return a string.
          Defaults to 'dumps'.
        :param loads: names the backend's method for the reverse
          operation -- returning a Python object from a string.
        :param loads_exc: can be either the name of the exception class
          used to denote decoding errors, or it can be a direct reference
          to the appropriate exception class itself.  If it is a name,
          then the assumption is that an exception class of that name
          can be found in the backend module's namespace.
        :param load: names the backend's 'load' method.
        :param dump: names the backend's 'dump' method.
        :rtype bool: True on success, False if the backend could not be loaded.

        """
        try:
            # Load the JSON backend
            mod = __import__(name)
        except ImportError:
            return False

        # Handle submodules, e.g. django.utils.simplejson
        try:
            for attr in name.split('.')[1:]:
                mod = getattr(mod, attr)
        except AttributeError:
            return False

        if not self._store(self._encoders, name, mod, dumps) or not self._store(
            self._decoders, name, mod, loads
        ):
            return False

        if isinstance(loads_exc, str):
            # This backend's decoder exception is part of the backend
            if not self._store(self._decoder_exceptions, name, mod, loads_exc):
                return False
        else:
            # simplejson uses ValueError
            self._decoder_exceptions[name] = loads_exc

        # Setup the default args and kwargs for this encoder/decoder
        self._encoder_options.setdefault(name, ([], {}))  # type: ignore
        self._decoder_options.setdefault(name, ([], {}))

        # Add this backend to the list of candidate backends
        self._backend_names.append(name)

        # Indicate that we successfully loaded a JSON backend
        self._verified = True
        return True

    def remove_backend(self, name: str) -> None:
        """Remove all entries for a particular backend."""
        self._encoders.pop(name, None)
        self._decoders.pop(name, None)
        self._decoder_exceptions.pop(name, None)
        self._decoder_options.pop(name, None)
        self._encoder_options.pop(name, None)
        if name in self._backend_names:
            self._backend_names.remove(name)
        self._verified = bool(self._backend_names)

    def backend_encode(
        self,
        name: str,
        obj: Any,
        indent: Optional[int] = None,
        separators: Optional[str] = None,
    ):
        optargs, optkwargs = self._encoder_options.get(name, ([], {}))
        encoder_kwargs = optkwargs.copy()
        if indent is not None:
            encoder_kwargs['indent'] = indent  # type: ignore[assignment]
        if separators is not None:
            encoder_kwargs['separators'] = separators  # type: ignore[assignment]
        encoder_args = (obj,) + tuple(optargs)
        return self._encoders[name](*encoder_args, **encoder_kwargs)

    def backend_decode(self, name: str, string: str) -> Any:
        optargs, optkwargs = self._decoder_options.get(name, ((), {}))
        decoder_kwargs = optkwargs.copy()
        return self._decoders[name](string, *optargs, **decoder_kwargs)

    def set_preferred_backend(self, name: str) -> None:
        """
        Set the preferred json backend.

        If a preferred backend is set then jsonpickle tries to use it
        before any other backend.

        For example::

            set_preferred_backend('simplejson')

        If the backend is not one of the built-in jsonpickle backends
        (json/simplejson) then you must load the backend
        prior to calling set_preferred_backend.

        AssertionError is raised if the backend has not been loaded.

        """
        if name in self._backend_names:
            self._backend_names.remove(name)
            self._backend_names.insert(0, name)
        else:
            errmsg = 'The "%s" backend has not been loaded.' % name
            raise AssertionError(errmsg)

    def set_encoder_options(self, name: str, *args: Any, **kwargs: Any) -> None:
        """
        Associate encoder-specific options with an encoder.

        After calling set_encoder_options, any calls to jsonpickle's
        encode method will pass the supplied args and kwargs along to
        the appropriate backend's encode method.

        For example::

            set_encoder_options('simplejson', sort_keys=True, indent=4)

        See the appropriate encoder's documentation for details about
        the supported arguments and keyword arguments.

        WARNING: If you pass sort_keys=True, and the object to encode
        contains ``__slots__``, and you set ``warn`` to True,
        a TypeError will be raised!
        """
        self._encoder_options[name] = (args, kwargs)

    def set_decoder_options(self, name: str, *args: Any, **kwargs: Any) -> None:
        """
        Associate decoder-specific options with a decoder.

        After calling set_decoder_options, any calls to jsonpickle's
        decode method will pass the supplied args and kwargs along to
        the appropriate backend's decode method.

        For example::

            set_decoder_options('simplejson', encoding='utf8', cls=JSONDecoder)

        See the appropriate decoder's documentation for details about
        the supported arguments and keyword arguments.

        """
        self._decoder_options[name] = (args, kwargs)


json = JSONBackend()
