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

        aug_localns = {}
        for tparam in cls.__type_params__:
            aug_localns[tparam.__name__] = tparam.__name__

        # TODO: Make sure that this is actually correct
        super_init = super().__init_subclass__()


        # However, if the type parameter hole is filled, we should not use the old accessor anymore.
        # For example:
        # class Parent[T]: pass
        # class Child(Parent[int]): pass
        # When getting Child().T, it should not require __orig_class__ because it has been filled already.
        for base in types.get_original_bases(cls):
            if isinstance(base, typing._GenericAlias):  # type: ignore -- We don't have a great alternative way for checking if an object is a GenericAlias
                parent_params = typing.get_origin(base).__type_params__
                parent_args = typing.get_args(base)

                for param, arg in zip(parent_params, parent_args):

                    print(param, arg, cls)
                    match arg:
                        case typing.TypeVar():
                            print("arg is TypeVar")
                            def accessor_override(self, arg=arg):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs   
                                print("accessor_override: arg is TypeVar")
                                if hasattr(self, arg.__name__):
                                    return getattr(self, arg.__name__)
                                raise ArgonError(f"No arg named {arg}")

                        case typing.ForwardRef():
                            print("arg is ForwardRef")

                            def accessor_override(self, arg=arg):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                print("accessor_override: arg is ForwardRef")
                                # for key in aug_localns.keys():
                                #     # print(f"key:{key}")
                                #     if hasattr(self,key):
                                #         aug_localns[key] = getattr(self, key)
                                #         # print(f"aug_localns[key]={aug_localns[key]}")
                                
                                print(self.__orig_class__)
                                retval = arg._evaluate(globalns, localns, frozenset())
                                
                                print(self.__orig_class__)
                                # print(f"retval={retval}")
                                # if isinstance(retval, typing._GenericAlias):
                                #     # __orig_class__ is an attribute set in GenericAlias.__call__ to 
                                #     # the instance of the GenericAlias that created the called class
                                #     return self.__orig_class__

                                if isinstance(retval, typing.TypeVar):
                                    return getattr(self, retval.__name__)
                                return retval

                        case type():
                            print("arg is type")
                            
                            def accessor_override(self, arg=arg, param=param):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                print("accessor_override: arg is type")
                                # print(f"type():arg={arg}, param={param}")
                                # print(f"Retrieving {param} = {arg}")
                                return arg
                        

                        case typing._GenericAlias():
                            print("arg is _GenericAlias")
                            
                            def accessor_override(self, arg=arg, param=param): # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                print("accessor_override: arg is _GenericAlias")
                                # print(f"typing._GenericAlias():arg={arg}, param={param}")
                                # print(f"Retrieving {param} = {arg}")
                                return arg

                        case _:
                            raise ArgonError(
                                f"Failed to resolve type {param} into a concrete type, got {type(arg)}: {arg}"
                            )
                    accessor_override.__name__ = param.__name__
                    setattr(cls, param.__name__, property(fget=accessor_override))

        # To handle generic type parameters, we should look them up dynamically at runtime
        for ind, tparam in enumerate(cls.__type_params__):
            param_name = tparam.__name__

            #print(f"setting accessor_tparam for {param_name}")

            def accessor_tparam(self, ind=ind):
                # breakpoint()
                if not hasattr(self, "__orig_class__"):
                    raise TypeError(
                        f"Cannot access type parameter {param_name} of {self.__class__}."
                    )
                #aug_localns[param_name] = self.__orig_class__.__args__[ind]
                return self.__orig_class__.__args__[ind]

            accessor_tparam.__name__ = param_name
            prop = property(fget=accessor_tparam)
            print(prop)
            setattr(cls, param_name, prop)
            print(getattr(cls,param_name))
        
        return super_init