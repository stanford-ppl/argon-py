import ast


class VariableTracker:
    def __init__(self):
        self.write_set_stack = [set()]
        self.read_set_stack = [set()]

    def push_context(self):
        self.write_set_stack.append(set())
        self.read_set_stack.append(set())

    def pop_context(self):
        return self.write_set_stack.pop(), self.read_set_stack.pop()

    def fold_context(self):
        curr_write_set, curr_read_set = self.pop_context()
        self.current_read_set().update(curr_read_set - self.current_write_set())
        self.current_write_set().update(curr_write_set)

    def current_write_set(self):
        return self.write_set_stack[-1]

    def previous_write_set(self):
        return self.write_set_stack[-2]

    def current_read_set(self):
        return self.read_set_stack[-1]

    def previous_read_set(self):
        return self.read_set_stack[-2]

    def add_written_var(self, var):
        self.current_write_set().add(var)

    def add_read_var(self, var):
        self.current_read_set().add(var)


class TransformerBase(ast.NodeTransformer):
    def __init__(self, file_name, calls, ifs, if_exps, loops):
        super().__init__()
        self.counter = 0
        self.unique_prefix = "__________"
        self.file_name = file_name
        self.calls = calls
        self.ifs = ifs
        self.if_exps = if_exps
        self.loops = loops
        self.concrete_to_abstract_flag = False
        self.variable_tracker = VariableTracker()

    def generate_temp_var(self, *args) -> str:
        return self.unique_prefix + "_".join(args) + "_" + str(self.counter)

    def concrete_to_abstract(self, node):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="__________argon", ctx=ast.Load()),
                            attr="argon",
                            ctx=ast.Load(),
                        ),
                        attr="virtualization",
                        ctx=ast.Load(),
                    ),
                    attr="type_mapper",
                    ctx=ast.Load(),
                ),
                attr="concrete_to_abstract",
                ctx=ast.Load(),
            ),
            args=[node],
            keywords=[],
        )

    def visit(self, node):
        self.variable_tracker.push_context()
        node = super().visit(node)
        self.variable_tracker.fold_context()
        return node

    def visit_Constant(self, node):
        if self.concrete_to_abstract_flag:
            return self.concrete_to_abstract(node)
        return node

    def visit_Name(self, node):
        # Save the loaded variables
        if isinstance(node.ctx, ast.Load):
            self.variable_tracker.add_read_var(node.id)

        if self.concrete_to_abstract_flag:
            return self.concrete_to_abstract(node)
        return node

    def visit_Subscript(self, node):
        """
        Handle subscript operations like layers[i] or list[0].

        Recursively visits the value and slice, which will convert them to
        abstract types if concrete_to_abstract_flag is True. This allows
        subscript operations to work with abstract types like NNModuleList.
        """
        # Recursively visit the value (e.g., 'layers') and slice (e.g., 'i')
        # This will convert them to abstract types if the flag is set
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)
        return node

    def visit_Assign(self, node):
        # Visit RHS of assignment
        node.value = self.visit(node.value)

        # Recursively process each target to extract all written variables
        for target in node.targets:
            self._process_target(target)

        return node

    def visit_AugAssign(self, node):
        # Visit RHS of assignment
        node.value = self.visit(node.value)

        if isinstance(node.target, ast.Name):
            self.variable_tracker.add_written_var(node.target.id)
        else:
            raise NotImplementedError(
                "Only support single-variable assignments for augmented assignments"
            )

        return node

    def _process_target(self, target):
        # Handle simple variable assignment
        if isinstance(target, ast.Name):
            self.variable_tracker.add_written_var(target.id)
        # Handle tuple or list unpacking
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._process_target(element)
        else:
            raise NotImplementedError("Unsupported target type on LHS of assignment")

    def visit_NamedExpr(self, node):
        # Visit RHS of assignment
        node.value = self.visit(node.value)

        if isinstance(node.target, ast.Name):
            self.variable_tracker.add_written_var(node.target.id)
        else:
            raise NotImplementedError(
                "Only support single-variable assignments for walrus operators"
            )

        return node
