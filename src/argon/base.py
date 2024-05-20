import inspect
import typing
import types

from argon.errors import ArgonError

# This is a function to recursively resolve arguments that are instances of the GenericAlias type
def r_resolve(globalns, localns, rarg):
    match rarg:
        case typing.TypeVar():
            if isinstance(globalns[rarg.__name__], type) and isinstance(localns[rarg.__name__], type):
                return localns[rarg.__name__]
            else:
                raise ArgonError(
                    f"Failed to resolve type {rarg}: All the type variables should have been resolved in advance"
                ) 
        case type():
            # print(f"in type(): {rarg}")
            return rarg
        case typing._GenericAlias():
            inner_arg_list=[]
            for inner_rarg in typing.get_args(rarg):
                inner_arg_list.append(r_resolve(globalns, localns, inner_rarg))
            return typing.GenericAlias(typing.get_origin(rarg), tuple(inner_arg_list))
        case _:
            raise ArgonError(
                f"Failed to resolve type {rarg}"
            )     
                  

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

        tparam_set = set()

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

                    match arg:
                        case typing.TypeVar():

                            def accessor_override(self, arg=arg):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                if hasattr(self, arg.__name__):
                                    return getattr(self, arg.__name__)
                                raise ArgonError(f"No arg named {arg}")

                        case typing.ForwardRef():

                            def accessor_override(self, arg=arg):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                retval = arg._evaluate(globalns, localns, frozenset())
                                if isinstance(retval, typing._GenericAlias):
                                    aug_ns = {}
                                    for key in tparam_set:
                                        if isinstance(globalns[key], typing.TypeVar) and isinstance(localns[key], typing.TypeVar):
                                            # Resolve the type parameters in this class that hasn't been resolved yet
                                            aug_ns[key] = getattr(self, key)                                        

                                    # augment the namespace
                                    globalns.update(aug_ns)
                                    localns.update(aug_ns)

                                    return arg._evaluate(globalns, localns, frozenset())


                                if isinstance(retval, typing.TypeVar):
                                    return getattr(self, retval.__name__)
                                return retval

                        case type():

                            def accessor_override(self, arg=arg, param=param):  # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                # print(f"Retrieving {param} = {arg}")
                                return arg

                        case typing._GenericAlias():
                            
                            def accessor_override(self, arg=arg): # type: ignore -- PyRight and other tools falsely report this as conflicting defs
                                
                                aug_ns = {}
                                for key in tparam_set:
                                    if isinstance(globalns[key], typing.TypeVar) and isinstance(localns[key], typing.TypeVar):
                                        # Resolve the type parameters in this class that hasn't been resolved yet
                                        aug_ns[key] = getattr(self, key)                                        

                                # augment the namespace
                                globalns.update(aug_ns)
                                localns.update(aug_ns)
                                
                                # recursively resolve the GenericAlias
                                arg_list=[]
                                for arg_i in typing.get_args(arg):
                                    arg_list.append(r_resolve(globalns, localns, arg_i)) 
                                return typing.GenericAlias(typing.get_origin(arg), tuple(arg_list))
                                                                
                        case _:
                            raise ArgonError(
                                f"Failed to resolve type {param} into a concrete type, got {type(arg)}: {arg}"
                            )
                    accessor_override.__name__ = param.__name__
                    setattr(cls, param.__name__, property(fget=accessor_override))

        # To handle generic type parameters, we should look them up dynamically at runtime
        for ind, tparam in enumerate(cls.__type_params__):
            param_name = tparam.__name__
            tparam_set.add(param_name)
            

            #print(f"setting accessor_tparam for {param_name}")

            def accessor_tparam(self, ind=ind):
                # breakpoint()
                if not hasattr(self, "__orig_class__"):
                    raise TypeError(
                        f"Cannot access type parameter {param_name} of {self.__class__}."
                    )
                return self.__orig_class__.__args__[ind]

            accessor_tparam.__name__ = param_name
            setattr(cls, param_name, property(fget=accessor_tparam))
            
        return super_init
