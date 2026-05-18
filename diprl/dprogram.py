"""
Program Derivation Graph Implementation for diprl

This module implements the core components of the programmatic policy architecture:
- WEIGHTCell: Neural cells that weight primitive policies
- SYMBOLICCell: Neural cells that learn symbolic predicates
- Program: Nested if-then-else (ITE) program structure
- FusionPrograms: Multiple programs with shared parameters
- SearchFusionProgram: Program derivation graph for architecture search

The key idea is to search over a space of programmatic policies that combine
pre-trained primitive policies using learned symbolic predicates.
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import copy


class WEIGHTCell(nn.Module):
    """
    Neural module that learns weights for combining primitive policies.

    Each WEIGHTCell outputs a weight vector that determines how to blend
    multiple primitive policies in the programmatic policy structure.
    These cells appear at the leaf nodes of the program AST (then/else branches).

    Args:
        index: Index of this cell in the program structure
        num_models: Number of primitive policies to weight
    """

    def __init__(self, index, num_models):

        super().__init__()

        self.index = index

        self.num_models = num_models

        self.W = nn.Parameter(torch.randn(1, self.num_models))

    def forward(self, x):
        """
        Forward pass - returns weight vector.

        Args:
            x: Input state (not used, weights are state-independent)

        Returns:
            Weight vector of shape (1, num_models)
        """

        return self.W


class SYMBOLICCell(nn.Module):
    """
    Neural module that learns symbolic predicates for program branching.

    Each SYMBOLICCell learns to compute a boolean-like value (probability)
    that determines which branch (then/else) to follow in the program.
    It combines multiple predefined symbolic features (e.g., position, velocity)
    using a learned linear combination followed by sigmoid activation.

    Args:
        index: Index of this cell in the program structure
        functions: List of symbolic feature functions (predicates)
        all_functions: Configuration of which functions to use (as binary code)
    """

    def __init__(self, index, functions, all_functions):

        super().__init__()

        self.index = index

        self.functions = functions

        self.all_functions = all_functions

        assert len(self.all_functions) == 1

        self.P = nn.Linear(self.all_functions[0][1], 1, bias=True)

    def forward(self, x):
        """
        Compute predicate probability from state.

        Args:
            x: Input state tensor

        Returns:
            Predicate probability P(x) ∈ [0, 1] indicating branch selection
        """

        f_code = self.all_functions[0][0]

        SYMBOLIC = []

        for j in range(len(self.functions)):

            if f_code[j]:

                SYMBOLIC.append(self.functions[j][0](x))

        sym_input = torch.cat(SYMBOLIC, dim=1)

        P = torch.sigmoid(self.P(sym_input))

        return P


class Program(nn.Module):
    """
    Nested if-then-else (ITE) program for programmatic policies.
    
    This class represents a single program with a LEFT-SKEWED tree structure where:
    - **THEN branches** (if predicate is TRUE): Lead directly to weight cells (primitives)
    - **ELSE branches** (if predicate is FALSE): Continue to next if-then-else or final weight
    
    This creates a sequential decision chain where predicates are checked in order,
    and the first TRUE predicate determines which primitive policies to use.
    
    Example with depth=2:
                if P[0]
               /       \
            W[0]      W[1]       <- then: W[0], else: W[1]
           (then)    (else)
    
    Example with depth=3:
                if P[0]
               /       \
            W[0]      if P[1]    <- then: W[0], else: check P[1]
           (then)    /       \
                  W[1]      W[2] <- then: W[1], else: W[2]
                 (then)    (else)
    
    Sequential decision logic:
    - Check P[0]: If TRUE → use W[0]
    - Check P[0]: If FALSE → Check P[1]:
                              If TRUE → use W[1]
                              If FALSE → use W[2]
    
    Args:
        depth: AST depth of the program (1 = simple controller, 2+ = nested ITE)
        models: List of pre-trained primitive policy models
        functions: List of symbolic feature functions
        all_functions: Configuration of which functions to use
        input_dim: Dimension of input state
        num_action_space: Dimension of action space
        index_action_space: Indices for slicing relevant state features
        beta: Temperature parameter for softmax over primitive weights
    """

    def __init__(
        self,
        depth,
        models,
        functions,
        all_functions,
        input_dim,
        num_action_space,
        index_action_space,
        beta=0.5,
    ):

        super().__init__()

        self.depth = depth

        self.models = models

        self.num_models = len(self.models)

        self.functions = functions

        self.all_functions = all_functions

        self.input_dim = input_dim

        self.num_action_space = num_action_space

        self.index_action_space = index_action_space

        self.beta = beta

        self.cells = nn.ModuleList()

        self.num_cells = self.depth * 2 - 1

        for i in range(self.num_cells):

            if i % 2 == 0 and i != self.num_cells - 1:

                self.cells.append(SYMBOLICCell(i, self.functions, self.all_functions))

            else:

                self.cells.append(WEIGHTCell(i, self.num_models))

        self.P_map = np.eye(self.depth, self.depth - 1)

        for i in range(self.P_map.shape[0]):

            for j in range(self.P_map.shape[1]):

                if i > j:

                    self.P_map[i, j] = -1

    @staticmethod
    def get_action(model, x):
        """
        Get action from a primitive policy model.

        Args:
            model: Pre-trained primitive policy
            x: Input state

        Returns:
            Action from the primitive policy
        """

        with torch.no_grad():

            if isinstance(x, torch.Tensor):

                obs_tensor = x.detach()

                output_device = obs_tensor.device

            else:

                obs_tensor = torch.as_tensor(x, dtype=torch.float32)

                output_device = obs_tensor.device

            try:

                model_device = next(model.parameters()).device

                obs_tensor = obs_tensor.to(model_device)

            except Exception:

                pass

            is_torch_model = isinstance(model, torch.nn.Module)

            action = None

            if hasattr(model, "act"):

                act_fn = model.act

                try:

                    action = act_fn(obs_tensor, deterministic=True)

                except TypeError:

                    try:

                        action = act_fn(obs_tensor)

                    except Exception:

                        if is_torch_model:

                            raise

                        action = act_fn(obs_tensor.detach().cpu().numpy())

                except Exception:

                    if is_torch_model:

                        raise

                    action = act_fn(obs_tensor.detach().cpu().numpy())

            elif hasattr(model, "get_action"):

                get_action_fn = model.get_action

                try:

                    action = get_action_fn(obs_tensor)

                except Exception:

                    if is_torch_model:

                        raise

                    action = get_action_fn(obs_tensor.detach().cpu().numpy())

                if isinstance(action, (list, tuple)) and len(action) > 0:

                    action = action[0]

            else:

                raise AttributeError(
                    "Primitive model must implement act() or get_action()."
                )

            if isinstance(action, torch.Tensor):

                action_tensor = action.detach()

            else:

                action_tensor = torch.as_tensor(action, dtype=torch.float32)

            return action_tensor.to(output_device)

    def forward(self, x):
        """
        Execute the program to compute an action.

        The forward pass:
        1. Evaluates all predicates (SYMBOLICCells) and weights (WEIGHTCells)
        2. Computes coefficients for each weight based on predicate values
           (which path through the tree leads to each leaf)
        3. Combines weights using coefficients
        4. Applies softmax to get primitive policy distribution
        5. Blends primitive policy actions according to distribution

        Args:
            x: Input state tensor of shape (batch_size, state_dim)

        Returns:
            Action tensor of shape (batch_size, action_dim)
        """

        if self.depth == 1:

            W = [self.cells[0](x)]

            W_coefficient = [1]

        else:

            P = []

            W = []

            for i in range(self.num_cells):

                c = self.cells[i]

                if i % 2 == 0 and i != self.num_cells - 1:

                    P.append(c(x))

                else:

                    W.append(c(x))

            W_coefficient = [1 for i in range(len(W))]

            P_map = self.P_map

            for i in range(P_map.shape[0]):

                for j in range(P_map.shape[1]):

                    if P_map[i, j] == 1:

                        W_coefficient[i] *= P[j]

                    elif P_map[i, j] == -1:

                        W_coefficient[i] *= 1 - P[j]

                    elif P_map[i, j] == 0:

                        pass

        w = torch.zeros(x.shape[0], self.num_models, device=x.device)

        for i in range(len(W)):

            w += W[i] * W_coefficient[i]

        w = (w / self.beta).softmax(dim=1)

        action = torch.zeros(x.shape[0], self.num_action_space, device=x.device)

        for i in range(self.num_models):

            action += w[:, i : i + 1] * self.get_action(
                self.models[i], x[:, self.index_action_space]
            )

        return action

    def discrete_forward(self, x):
        """
        Execute the program to compute logits for discrete actions.

        For discrete action spaces where primitives directly correspond to actions
        (e.g., primitive 0 = action 0, primitive 1 = action 1), we:
        1. Compute the weight distribution over primitives
        2. Use these weights as logits for the action distribution

        Args:
            x: Input state tensor of shape (batch_size, state_dim)

        Returns:
            Logits for discrete actions of shape (batch_size, num_actions)
            where num_actions = num_models (assuming 1-to-1 correspondence)
        """

        if self.depth == 1:

            W = [self.cells[0](x)]

            W_coefficient = [1]

        else:

            P = []

            W = []

            for i in range(self.num_cells):

                c = self.cells[i]

                if i % 2 == 0 and i != self.num_cells - 1:

                    P.append(c(x))

                else:

                    W.append(c(x))

            W_coefficient = [1 for i in range(len(W))]

            P_map = self.P_map

            for i in range(P_map.shape[0]):

                for j in range(P_map.shape[1]):

                    if P_map[i, j] == 1:

                        W_coefficient[i] *= P[j]

                    elif P_map[i, j] == -1:

                        W_coefficient[i] *= 1 - P[j]

                    elif P_map[i, j] == 0:

                        pass

        logits = torch.zeros(x.shape[0], self.num_models, device=x.device)

        for i in range(len(W)):

            logits += W[i] * W_coefficient[i]

        return logits / self.beta

    def string_format(self, function_names=None, primitive_names=None):
        """
        Format the program as a human-readable string with if-else structure.

        Args:
            function_names: List of feature names (e.g., ['xobj', 'yobj', 'xarm', 'yarm'])
                          If None, will use generic names like 'x0', 'x1', etc.
            primitive_names: List of primitive policy names (e.g., ['PUSH-DOWN', 'PUSH-LEFT'])
                           If None, will try to extract from self.models or use generic names

        Returns:
            String representation of the program in if-else format
        """

        import torch.nn.functional as F

        if primitive_names is None:

            if hasattr(self, "models") and len(self.models) > 0:

                primitive_names = []

                for model in self.models:

                    if isinstance(model, str):

                        primitive_names.append(model.upper())

                    elif hasattr(model, "name"):

                        primitive_names.append(model.name.upper())

                    else:

                        primitive_names.append(f"π{len(primitive_names)}")

            else:

                primitive_names = [f"π{i}" for i in range(self.num_models)]

        if function_names is None:

            f_code = self.all_functions[0][0]

            function_names = []

            for j, (func, dim) in enumerate(self.functions):

                if f_code[j]:

                    func_name = func.__name__ if hasattr(func, "__name__") else f"f{j}"

                    if dim == 1:

                        function_names.append(func_name)

                    else:

                        for d in range(dim):

                            function_names.append(f"{func_name}_{d}")

        output = []

        predicates = []

        weights = []

        for i in range(self.num_cells):

            c = self.cells[i]

            if i % 2 == 0 and i != self.num_cells - 1:

                pred_idx = len(predicates)

                theta = c.P.weight.data.squeeze().cpu().numpy()

                theta_c = c.P.bias.data.item()

                predicates.append((theta, theta_c))

            else:

                w = c.W.data.squeeze().cpu().numpy()

                w_softmax = F.softmax(torch.tensor(w) / self.beta, dim=0).numpy()

                weights.append(w_softmax)

        indent = 0

        for i in range(self.depth):

            if i < self.depth - 1:

                theta, theta_c = predicates[i]

                if i == 0:

                    output.append(f"if (θ{i+1}c + θ{i+1}ᵀ · X > 0)")

                else:

                    output.append("  " * i + f"else if (θ{i+1}c + θ{i+1}ᵀ · X > 0)")

                w_str = " + ".join(
                    [
                        f"{w*100:.0f}% · π{primitive_names[j]}(s)"
                        for j, w in enumerate(weights[i])
                        if w > 0.005
                    ]
                )

                output.append("  " * (i + 1) + f"then ( {w_str} )")

            else:

                if self.depth > 1:

                    w_str = " + ".join(
                        [
                            f"{w*100:.0f}% · π{primitive_names[j]}(s)"
                            for j, w in enumerate(weights[i])
                            if w > 0.005
                        ]
                    )

                    output.append("  " * i + f"else ( {w_str} )")

                else:

                    w_str = " + ".join(
                        [
                            f"{w*100:.0f}% · π{primitive_names[j]}(s)"
                            for j, w in enumerate(weights[0])
                            if w > 0.005
                        ]
                    )

                    output.append(f"( {w_str} )")

        output.append("")

        if len(function_names) > 0:

            x_str = ", ".join(function_names)

            output.append(f"X = [ {x_str} ]")

        for i, (theta, theta_c) in enumerate(predicates):

            theta_str = ", ".join([f"{t:.3f}" for t in theta])

            output.append(f"θ{i+1} = [ {theta_str} ], θ{i+1}c = {theta_c:.3f}")

        return "\n".join(output)


class SimpleSearchMap(nn.Module):
    """
    Simple search map for program selection using a single softmax distribution.

    This maintains a probability distribution over programs of different depths.
    A single parameter vector is used, and softmax gives the probability of
    selecting each program depth.

    NOTE: This is a simplified version. For DSLs with more than two production
    rules at each level, it's recommended to use independent parameters for
    each layer of the derivation graph (see ArchitectureSearchMap).

    Args:
        depth: Maximum program depth to search over
    """

    def __init__(self, depth):

        super().__init__()

        self.depth = depth

        self.type = "simple"

        self.v = nn.Parameter(torch.zeros(self.depth), requires_grad=True)

        torch.nn.init.ones_(self.v)

    def freeze(self):
        """Freeze parameters during program optimization phase."""

        self.v.requires_grad = False

    def unfreeze(self):
        """Unfreeze parameters during architecture search phase."""

        self.v.requires_grad = True


class ArchitectureSearchMap(nn.Module):
    """
    Layer-by-layer architecture search map for program derivation graph.

    This implements a more sophisticated search strategy where at each level
    of the derivation graph, we decide whether to stop (depth i) or continue
    (depth i+1). This forms a sequential decision process.

    At each level i, we have a binary choice:
    - option[0]: Stop and use program of current depth
    - option[1]: Continue expanding to next depth

    The probability of selecting a program with depth d is the product of:
    - Probabilities of continuing at all levels before d
    - Probability of stopping at level d

    Args:
        depth: Maximum program depth to search over
    """

    def __init__(self, depth, equally_initialize=False):

        super().__init__()

        self.depth = depth

        self.type = "architecture"

        self.options = nn.ParameterList()

        if equally_initialize:

            for i in range(self.depth - 1):

                p_0 = 1 / (self.depth - i)

                p_1 = 1 - p_0

                self.options.append(
                    nn.Parameter(torch.tensor(np.log([p_0, p_1])), requires_grad=True)
                )

        else:

            for _ in range(self.depth - 1):

                self.options.append(nn.Parameter(torch.rand(2), requires_grad=True))

    def freeze(self):
        """Freeze parameters during program optimization phase."""

        for option in self.options:

            option.requires_grad = False

    def unfreeze(self):
        """Unfreeze parameters during architecture search phase."""

        for option in self.options:

            option.requires_grad = True


class FusionPrograms(nn.Module):
    """
    Fusion of multiple ITE programs with shared parameters (Program Derivation Graph).
    
    This class represents a "superposition" of multiple LEFT-SKEWED programs with 
    different depths, all sharing common parameters where possible. This enables 
    efficient joint training and architecture search over program structures.
    
    Key insight: Programs of depth d and d+1 can share their first d-1 layers.
    This creates a derivation graph structure where:
    - Shared cells: Common to multiple program depths (predicates and weights)
    - Exclusive cells: Unique to each program depth (final weight at each depth)
    
    Example with depth=3 (searching over programs of depth 1, 2, and 3):
    
      Depth 1:  W_ex[0]  (just a single weight, no predicates)
                ^^^^^^^^ EXCLUSIVE - only used by depth-1 program
      
      Depth 2:       if P[0]  (shared predicate)
                    /       \
                 W[0]     W_ex[1]    <- then: W[0], else: W_ex[1]
               (shared)  (EXCLUSIVE - only used by depth-2 program)
      
      Depth 3:       if P[0]  (shared predicate - reused from depth 2!)
                    /       \
                 W[0]      if P[1]   <- then: W[0] (shared), else: check P[1]
               (shared)   /       \
                       W[1]     W_ex[2]  <- then: W[1], else: W_ex[2]
                    (shared)  (EXCLUSIVE - only used by depth-3 program)
    
    Notice the left-skewed structure:
    - THEN branches always lead to weight cells
    - ELSE branches continue with more if-then-else or exclusive final weight
    - Programs share P[0], W[0], P[1], W[1]... but have exclusive final weights
    
    Concrete Example - What each program actually does:
    
    Program 1 (Depth 1):
        OUTPUT: W_ex[0]
        → No predicates, just outputs exclusive weight directly
        → Uses: ex_cells[0] only
    
    Program 2 (Depth 2):
        if P[0]:                     ← SHARED (predicate)
            then: W[0]               ← SHARED (weight)
            else: W_ex[1]            ← EXCLUSIVE (unique to this depth)
        → Uses: cells[0]=P[0], cells[1]=W[0], ex_cells[1]=W_ex[1]
    
    Program 3 (Depth 3):
        if P[0]:                     ← SHARED (reused from Program 2!)
            then: W[0]               ← SHARED (reused from Program 2!)
            else:
                if P[1]:             ← SHARED
                    then: W[1]       ← SHARED
                    else: W_ex[2]    ← EXCLUSIVE (unique to this depth)
        → Uses: cells[0]=P[0], cells[1]=W[0], cells[2]=P[1], cells[3]=W[1], ex_cells[2]=W_ex[2]
    
    Key insight: Programs 2 and 3 both share P[0] and W[0], but have different
    terminal weights (W_ex[1] vs W_ex[2]). This sharing enables efficient training
    while allowing each program to have its own distinct final behavior.
    
    By maintaining this structure, we can:
    1. Train all programs simultaneously with shared gradients
    2. Use a search_map to weight different program depths
    3. Extract the best program after training
    
    Args:
        depth: Maximum program depth (will represent programs 1 to depth)
        models: List of pre-trained primitive policy models
        functions: List of symbolic feature functions
        all_functions: Configuration of which functions to use
        input_dim: Dimension of input state
        num_action_space: Dimension of action space
        index_action_space: Indices for slicing relevant state features
        beta: Temperature parameter for softmax over primitive weights
    """

    def __init__(
        self,
        depth,
        models,
        functions,
        all_functions,
        input_dim,
        num_action_space,
        index_action_space,
        beta=0.5,
    ):

        super().__init__()

        self.depth = depth

        self.models = models

        self.num_models = len(self.models)

        self.functions = functions

        self.all_functions = all_functions

        self.input_dim = input_dim

        self.num_action_space = num_action_space

        self.index_action_space = index_action_space

        self.beta = beta

        self.cells = nn.ModuleList()

        self.num_shared_cells = 2 * self.depth - 2

        for i in range(self.num_shared_cells):

            if i % 2 == 0:

                self.cells.append(SYMBOLICCell(i, self.functions, self.all_functions))

            else:

                self.cells.append(WEIGHTCell(i, self.num_models))

        self.ex_cells = nn.ModuleList()

        self.num_exclusive_cells = self.depth

        for i in range(self.num_exclusive_cells):

            self.ex_cells.append(WEIGHTCell(i, self.num_models))

        self.P_maps = []

        for d in range(self.depth):

            depth = d + 1

            P_map = np.eye(depth, depth - 1)

            for i in range(P_map.shape[0]):

                for j in range(P_map.shape[1]):

                    if i > j:

                        P_map[i, j] = -1

            self.P_maps.append(P_map)

    @staticmethod
    def get_action(model, x):
        """
        Get action from a primitive policy model.

        Args:
            model: Pre-trained primitive policy
            x: Input state

        Returns:
            Action from the primitive policy
        """

        with torch.no_grad():

            if isinstance(x, torch.Tensor):

                obs_tensor = x.detach()

                output_device = obs_tensor.device

            else:

                obs_tensor = torch.as_tensor(x, dtype=torch.float32)

                output_device = obs_tensor.device

            try:

                model_device = next(model.parameters()).device

                obs_tensor = obs_tensor.to(model_device)

            except Exception:

                pass

            is_torch_model = isinstance(model, torch.nn.Module)

            action = None

            if hasattr(model, "act"):

                act_fn = model.act

                try:

                    action = act_fn(obs_tensor, deterministic=True)

                except TypeError:

                    try:

                        action = act_fn(obs_tensor)

                    except Exception:

                        if is_torch_model:

                            raise

                        action = act_fn(obs_tensor.detach().cpu().numpy())

                except Exception:

                    if is_torch_model:

                        raise

                    action = act_fn(obs_tensor.detach().cpu().numpy())

            elif hasattr(model, "get_action"):

                get_action_fn = model.get_action

                try:

                    action = get_action_fn(obs_tensor)

                except Exception:

                    if is_torch_model:

                        raise

                    action = get_action_fn(obs_tensor.detach().cpu().numpy())

                if isinstance(action, (list, tuple)) and len(action) > 0:

                    action = action[0]

            else:

                raise AttributeError(
                    "Primitive model must implement act() or get_action()."
                )

            if isinstance(action, torch.Tensor):

                action_tensor = action.detach()

            else:

                action_tensor = torch.as_tensor(action, dtype=torch.float32)

            return action_tensor.to(output_device)

    def forward(self, x, search_map):
        """
        Execute the fused programs to compute a weighted action.

        This method computes actions from ALL programs at different depths,
        then blends them according to probabilities from the search_map.
        This enables gradient-based architecture search.

        Args:
            x: Input state tensor of shape (batch_size, state_dim)
            search_map: SimpleSearchMap or ArchitectureSearchMap providing
                       probability distribution over program depths

        Returns:
            Action tensor of shape (batch_size, action_dim) - weighted blend
            of actions from all programs
        """

        if search_map.type == "simple":

            v = F.softmax(search_map.v, dim=0)

        elif search_map.type == "architecture":

            v = nn.Parameter(
                torch.ones(self.depth, device=x.device), requires_grad=False
            )

            for i in range(len(v)):

                options = search_map.options

                if i == 0:

                    v[i] = options[0].softmax(dim=0)[0]

                else:

                    prev = 1

                    for j in range(i):

                        prev *= options[j].softmax(dim=0)[1]

                    if i == len(v) - 1:

                        v[i] = prev

                    else:

                        option_value = options[i].softmax(dim=0)

                        v[i] = prev * option_value[0]

        action = torch.zeros(x.shape[0], self.num_action_space, device=x.device)

        primitive_actions = []

        for i in range(self.num_models):

            primitive_actions.append(
                self.get_action(self.models[i], x[:, self.index_action_space])
            )

        if self.depth == 1:

            w = self.ex_cells[0](x)

            w = (w / self.beta).softmax(dim=1)

            for i in range(self.num_models):

                action += w[:, i : i + 1] * primitive_actions[i]

        else:

            P = []

            W = []

            ex_W = []

            for i in range(self.num_shared_cells):

                c = self.cells[i]

                if i % 2 == 0:

                    P.append(c(x))

                else:

                    W.append(c(x))

            for i in range(self.num_exclusive_cells):

                c = self.ex_cells[i]

                ex_W.append(c(x))

            for d in range(self.depth):

                depth = d + 1

                if depth == 1:

                    w = ex_W[0]

                    w = (w / self.beta).softmax(dim=1)

                    for i in range(self.num_models):

                        action += v[0] * w[:, i : i + 1] * primitive_actions[i]

                else:

                    P_map = self.P_maps[d]

                    W_coefficient = [1 for i in range(P_map.shape[0])]

                    for i in range(P_map.shape[0]):

                        for j in range(P_map.shape[1]):

                            if P_map[i, j] == 1:

                                W_coefficient[i] *= P[j]

                            elif P_map[i, j] == -1:

                                W_coefficient[i] *= 1 - P[j]

                            elif P_map[i, j] == 0:

                                pass

                    w = torch.zeros(x.shape[0], self.num_models, device=x.device)

                    for i in range(len(W_coefficient) - 1):

                        w += W[i] * W_coefficient[i]

                    w += ex_W[d] * W_coefficient[i + 1]

                    w = (w / self.beta).softmax(dim=1)

                    for i in range(self.num_models):

                        action += v[d] * w[:, i : i + 1] * primitive_actions[i]

        return action

    def discrete_forward(self, x, search_map):
        """
        Execute the fused programs to compute logits for discrete actions.

        For discrete action spaces where primitives directly correspond to actions,
        we compute logits as a weighted combination of each program's logits,
        weighted by the search_map probability distribution.

        Args:
            x: Input state tensor of shape (batch_size, state_dim)
            search_map: SimpleSearchMap or ArchitectureSearchMap providing
                       probability distribution over program depths

        Returns:
            Logits for discrete actions of shape (batch_size, num_actions)
        """

        if search_map.type == "simple":

            v = F.softmax(search_map.v, dim=0)

        elif search_map.type == "architecture":

            v = nn.Parameter(
                torch.ones(self.depth, device=x.device), requires_grad=False
            )

            for i in range(len(v)):

                options = search_map.options

                if i == 0:

                    v[i] = options[0].softmax(dim=0)[0]

                else:

                    prev = 1

                    for j in range(i):

                        prev *= options[j].softmax(dim=0)[1]

                    if i == len(v) - 1:

                        v[i] = prev

                    else:

                        option_value = options[i].softmax(dim=0)

                        v[i] = prev * option_value[0]

        overall_logits = torch.zeros(x.shape[0], self.num_models, device=x.device)

        if self.depth == 1:

            w = self.ex_cells[0](x) / self.beta

            overall_logits += v[0] * w

        else:

            P = []

            W = []

            ex_W = []

            for i in range(self.num_shared_cells):

                c = self.cells[i]

                if i % 2 == 0:

                    P.append(c(x))

                else:

                    W.append(c(x))

            for i in range(self.num_exclusive_cells):

                c = self.ex_cells[i]

                ex_W.append(c(x))

            for d in range(self.depth):

                depth = d + 1

                if depth == 1:

                    logits = ex_W[0] / self.beta

                    overall_logits += v[0] * logits

                else:

                    P_map = self.P_maps[d]

                    W_coefficient = [1 for i in range(P_map.shape[0])]

                    for i in range(P_map.shape[0]):

                        for j in range(P_map.shape[1]):

                            if P_map[i, j] == 1:

                                W_coefficient[i] *= P[j]

                            elif P_map[i, j] == -1:

                                W_coefficient[i] *= 1 - P[j]

                            elif P_map[i, j] == 0:

                                pass

                    w = torch.zeros(x.shape[0], self.num_models, device=x.device)

                    for i in range(len(W_coefficient) - 1):

                        w += W[i] * W_coefficient[i]

                    w += ex_W[d] * W_coefficient[i + 1]

                    logits = w / self.beta

                    overall_logits += v[d] * logits

        return overall_logits

    def freeze(self):
        """
        Freeze all program parameters (cells).
        Used during architecture search phase - only search_map is trained.
        """

        for cell in self.cells:

            for param in cell.parameters():

                param.requires_grad = False

        for cell in self.ex_cells:

            for param in cell.parameters():

                param.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze all program parameters (cells).
        Used during program optimization phase - cells are trained, search_map is frozen.
        """

        for cell in self.cells:

            for param in cell.parameters():

                param.requires_grad = True

        for cell in self.ex_cells:

            for param in cell.parameters():

                param.requires_grad = True


class SearchFusionProgram(nn.Module):
    """
    Program Derivation Graph for diprl architecture search.

    This is the top-level module that orchestrates the diprl program search framework.
    It combines FusionPrograms (the program structures) with a search_map
    (the architecture selector) to enable joint optimization.

    The training alternates between two phases:
    1. Architecture Search (pointer=0): Train search_map, freeze programs
       - Optimize which program depth to use
    2. Program Optimization (pointer=1): Train programs, freeze search_map
       - Optimize predicates and weights in the programs

    This bi-level optimization allows the system to:
    - Discover the right program complexity (depth)
    - Learn effective symbolic predicates
    - Find good combinations of primitive policies

    NOTE: graph_depth is different from program AST depth!
    - graph_depth = number of program options to search over
    - program AST depth = complexity of each individual program
    - Relationship: program depths range from 1 to (graph_depth - 1)

    Example: graph_depth=6 means searching over programs with AST depths 1-5
    - Depth 1: Simple weighted blend of primitives
    - Depth 2: Single if-then-else
    - Depth 3: Nested if-then-else with 2 predicates
    - ... and so on

    Args:
        graph_depth: Number of program depths to search over (max_depth + 1)
        domain: Task domain name (e.g., 'ant_cross_maze', 'half_cheetah_hurdle')
        beta: Temperature parameter for softmax over primitives
        search_map_type: Type of search map ('architecture' or 'simple')
    """

    def __init__(
        self,
        graph_depth,
        domain,
        parse_domain,
        beta=0.5,
        search_map_type="architecture",
        equally_initialize=False,
    ):

        super().__init__()

        self.domain = domain

        self.graph_depth = graph_depth

        self.search_map_type = search_map_type

        if search_map_type == "architecture":

            self.search_map = ArchitectureSearchMap(
                depth=self.graph_depth - 1,
                equally_initialize=equally_initialize,
            )

        elif search_map_type == "simple":

            self.search_map = SimpleSearchMap(depth=self.graph_depth - 1)

        self.fusion_programs = FusionPrograms(
            depth=self.graph_depth - 1, **parse_domain(domain), beta=beta
        )

        self.search_map.unfreeze()

        self.fusion_programs.freeze()

        self.pointer = 0

    def flip(self):
        """
        Toggle between architecture search and program optimization phases.

        This implements the alternating optimization strategy:
        - Phase 0 → Phase 1: Freeze search_map, unfreeze programs
        - Phase 1 → Phase 0: Freeze programs, unfreeze search_map

        Typically called after a fixed number of training iterations
        in the current phase.
        """

        if self.pointer == 0:

            self.pointer = 1

            self.search_map.freeze()

            self.fusion_programs.unfreeze()

        elif self.pointer == 1:

            self.pointer = 0

            self.search_map.unfreeze()

            self.fusion_programs.freeze()

    def forward(self, x):
        """
        Compute action from the current superposition of programs.

        Args:
            x: Input state tensor

        Returns:
            Action tensor - weighted blend of all programs' actions
        """

        action = self.fusion_programs(x, self.search_map)

        return action

    def discrete_forward(self, x):
        """
        Compute discrete action from the current superposition of programs.

        For discrete primitives, selects the most probable primitive
        based on the weighted combination of all programs.

        Args:
            x: Input state tensor

        Returns:
            Discrete action indices
        """

        action = self.fusion_programs.discrete_forward(x, self.search_map)

        return action

    def extract(self, parse_domain):
        """
        Extract a single discrete program from the derivation graph.

        After training, this method selects the most probable program
        and extracts it as a standalone Program object. The selected
        program inherits the learned parameters (predicates and weights)
        from the fusion programs.

        Selection strategy:
        - Choose program with highest probability according to search_map
        - Copy the relevant cells (both shared and exclusive) from fusion_programs

        Returns:
            Program: A discrete program with fixed structure and learned parameters
        """

        self.search_map.freeze()

        self.fusion_programs.unfreeze()

        if self.search_map_type == "simple":

            index = self.search_map.v.argmax()

        elif self.search_map_type == "architecture":

            v = nn.Parameter(torch.ones(self.graph_depth - 1), requires_grad=False)

            for i in range(len(v)):

                options = self.search_map.options

                if i == 0:

                    v[i] = options[0].softmax(dim=0)[0]

                else:

                    prev = 1

                    for j in range(i):

                        prev *= options[j].softmax(dim=0)[1]

                    if i == len(v) - 1:

                        v[i] = prev

                    else:

                        option_value = options[i].softmax(dim=0)

                        v[i] = prev * option_value[0]

            index = v.argmax()

        domain_dict = parse_domain(self.domain)

        prog = Program(depth=index + 1, **domain_dict, beta=self.fusion_programs.beta)

        prog.cells = nn.ModuleList()

        for i in range(2 * index):

            prog.cells.append(copy.deepcopy(self.fusion_programs.cells[i]))

        prog.cells.append(copy.deepcopy(self.fusion_programs.ex_cells[index]))

        return prog
