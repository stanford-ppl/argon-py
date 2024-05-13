import inspect
import typing
import types

from argon.errors import ArgonError


### WARNING: This does not correctly handle shadowing of typevars -- every type parameter should be unique.
class ArgonMeta:
    def __init_subclass__(cls) -> None:
        # print(f"Concretizing Class {cls}")
        # The calling stack, where ostensibly all of the values were defined.
        localns = {}
        globalns = {}
        # For each frame, iterating outside in, excluding this one.
        for finfo in inspect.stack()[:0:-1]:
            localns.update(finfo.frame.f_locals)
            globalns.update(finfo.frame.f_globals)

        # Have to register ourselves too!
        localns[cls.__name__] = cls
        
        
        # TODO: Make sure that this is actually correct
        super_init = super().__init_subclass__()

        # To handle generic type parameters, we should look them up dynamically at runtime
        for ind, tparam in enumerate(cls.__type_params__):
            param_name = tparam.__name__

            def accessor_tparam(self, ind=ind):
                # print(f"Reading {self}.{param_name}")
                if not hasattr(self, "__orig_class__"):
                    raise TypeError(
                        f"Cannot access type parameter {param_name} of {self.__class__}."
                    )
                return self.__orig_class__.__args__[ind]

            accessor_tparam.__name__ = param_name
            setattr(cls, param_name, property(fget=accessor_tparam))

        # However, if the type parameter hole is filled, we should not use the old accessor anymore.
        # For example:
        # class Parent[T]: pass
        # class Child(Parent[int]): pass
        # When getting Child().T, it should not require __orig_class__ because it has been filled already.
        for arg in cls.__type_params__:
            print(f"cls = {cls}")
            print(arg)
            match arg:
                case typing.TypeVar():
                    print("This is ytpe ar")
                case _:
                    print("I don't know")

        for base in types.get_original_bases(cls):
            print(f"Base: {base}, type(base)= {type(base)}")
            if isinstance(base, typing._GenericAlias):  # type: ignore -- We don't have a great alternative way for checking if an object is a GenericAlias
                parent_params = typing.get_origin(base).__type_params__
                parent_args = typing.get_args(base)
                for param, arg in zip(parent_params, parent_args):

                    print(param, arg, cls)
                    match arg:
                        case typing.TypeVar():
                            print("In type var")
                            def accessor_override(self, arg=arg):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                print(f"typing.TypeVar():{arg}")
                                if hasattr(self, arg.__name__):
                                    return getattr(self, arg.__name__)
                                raise ArgonError(f"No arg named {arg}")

                        case typing.ForwardRef():

                            def accessor_override(self, arg=arg):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                
                                localns["T"]=int
                                print(f"typing.ForwardRef():{arg}")
                                retval = arg._evaluate(globalns, localns, frozenset())
                                if isinstance(retval, typing._GenericAlias):  # type: ignore -- We don't have a great alternative way for checking if an object is a GenericAlias
                                    print("It's a generic alias")

                                if isinstance(retval, typing.TypeVar):
                                    return getattr(self, retval.__name__)
                                return retval

                        case type():
                            

                            def accessor_override(self, arg=arg, param=param):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                print(f"type():arg={arg}, param={param}")
                                # print(f"Retrieving {param} = {arg}")
                                return arg
                        

                        case typing._GenericAlias():
                            

                            def accessor_override(self, arg=arg, param=param):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                print(f"typing._GenericAlias():arg={arg}, param={param}")
                                print(f"Retrieving {param} = {arg}")
                                return arg

                        case _:
                            raise ArgonError(
                                f"Failed to resolve type {param} into a concrete type, got {type(arg)}: {arg}"
                            )
                    accessor_override.__name__ = param.__name__
                    setattr(cls, param.__name__, property(fget=accessor_override))
            elif isinstance(base, typing.Generic):
                print("generic[T]")
        return super_init
