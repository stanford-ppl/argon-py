from argon.virtualization.virtualizer.virtualizer_function_call import TransformerFunctionCall
from argon.virtualization.virtualizer.virtualizer_ifthenelse import TransformerIfThenElse
from argon.virtualization.virtualizer.virtualizer_loop import TransformerLoop


class TransformerTop(TransformerFunctionCall, TransformerIfThenElse, TransformerLoop):
    def __init__(self, file_name, calls, ifs, if_exps, loops):
        super().__init__(file_name, calls, ifs, if_exps, loops)