import re
from typing import List, Tuple, Any, Optional, Union

import torch
import torch.nn as nn

from iatransfer.toolkit.base_transfer import Transfer


class TraceTransfer(Transfer):
    """Trace tensor transfer algorithm for IAT.
    """

    def __init__(self, reverse_priority: bool = False, **kwargs) -> None:
        if reverse_priority:
            self.sgn = +1
        else:
            self.sgn = -1
        super().__init__(**kwargs)

    layers_classes = [
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.Linear
    ]
    layers_mapping = {}

    def create_layers_mapping(self, matched: List[Tuple[nn.Module, nn.Module]]):
        for matching in matched:
            if isinstance(matching, list):
                self.create_layers_mapping(matching)
            elif matching[0] is not None and matching[1] is not None:
                self.layers_mapping[matching[1]] = matching[0]

    class TorchSubstitute:
        """
            substitutes every call to torch library
        """

        def __getattr__(self, attr: Any) -> Any:
            def f(*args, **kwargs):
                return args[0] if len(args) > 0 else None

            return f

    class Module:

        def __init__(self, trace: 'torch.jit._trace.TracedModule', module: nn.Module,
                     outer_class: 'TraceTransfer') -> None:
            self.trace = trace
            self.module = module
            self.outer_class = outer_class

        def __getattr__(self, attr: Any) -> Any:
            if attr == "torch":
                return self.outer_class.TorchSubstitute()
            return self.__class__(getattr(self.trace, attr), getattr(self.module, attr), self.outer_class)

        def forward(self, ids: Optional[List[int]] = None):
            if len(list(self.module.children())) == 0 or any(
                    [isinstance(self.module, clazz) for clazz in self.outer_class.layers_classes]):
                # print("ids: ", ids)
                return self._transfer(self.module, ids)
            else:
                instructions = self.trace.code
                instructions = re.sub(r"forward[^\(]*\(", "forward(", instructions)
                instructions = instructions.replace("torch", "self.torch")
                instructions = instructions.split("\n")
                input_name = instructions[1].strip()[:instructions[1].strip().find(":")]
                exec(f"{input_name} = ids")
                i = 2
                while i < len(instructions) and instructions[i].find("return") == -1:
                    # print(instructions[i].strip(), "\n")
                    exec(instructions[i].strip())
                    i += 1
                if i < len(instructions):
                    return eval(re.search(r"return(.+)", instructions[i]).group(1).strip())
                return None

        def _get_center_slices(self, a: int, b: int) -> Tuple[slice, slice]:
            if a < b:
                return slice(0, a), slice((b - a) // 2, -((b - a + 1) // 2))
            elif a > b:
                return slice((a - b) // 2, -((a - b + 1) // 2)), slice(0, b)
            else:
                return slice(0, a), slice(0, b)

        def _get_output_channels(self, from_tensor: torch.Tensor, to_tensor: torch.Tensor) -> Union[
            Tuple[torch.Tensor, slice], Tuple[slice, slice]]:
            dims = [i for i in range(1, len(from_tensor.shape))]
            if from_tensor.shape[0] > to_tensor.shape[0]:
                if len(dims) == 0:
                    out_channels = list(enumerate(from_tensor))
                else:
                    out_channels = list(enumerate(from_tensor.abs().sum(dim=dims)))
                out_channels.sort(key=lambda w: self.outer_class.sgn * w[1])
                c = to_tensor.shape[0]
                from_ids = sorted([w[0] for w in out_channels[:c]])
                return torch.tensor(from_ids), slice(0, c)
            else:
                return self._get_center_slices(from_tensor.shape[0], to_tensor.shape[0])

        def _get_slices(self, from_tensor: torch.Tensor, to_tensor: torch.Tensor, ids: Optional[List[int]]) -> Tuple[
            Tuple[None, ...], Tuple[None, ...]]:
            n = len(from_tensor.shape)
            from_slices, to_slices = [None] * n, [None] * n
            if n > 1:  # use input ids to choose input channels
                a, b = from_tensor.shape[1], to_tensor.shape[1]
                # if ids is not None and a > b and len(ids) != b:
                #     print("HERE")
                if ids is not None and a > b and len(ids) == b and torch.max(ids) < from_tensor.shape[1]:
                    # print("ENTERED")
                    from_slices[1] = ids
                    to_slices[1] = slice(0, b)
                    from_tensor = from_tensor.index_select(1, ids)
                else:
                    from_slices[1], to_slices[1] = self._get_center_slices(a, b)
            # choose output channels
            if n > 1 or ids is None or len(ids) != to_tensor.shape[0] or to_tensor.shape[0] > from_tensor.shape[0]:
                from_slices[0], to_slices[0] = self._get_output_channels(from_tensor, to_tensor)
            else:
                from_slices[0], to_slices[0] = ids, slice(0, to_tensor.shape[0])
            # choose centers on the remaining channels
            for i in range(2, n):
                from_slices[i], to_slices[i] = self._get_center_slices(from_tensor.shape[i], to_tensor.shape[i])

            total_unsqueeze = 0
            for i in range(len(from_slices) - 1, -1, -1):
                if isinstance(from_slices[i], torch.Tensor):
                    # print(i)
                    for _ in range(total_unsqueeze):
                        from_slices[i] = from_slices[i].unsqueeze(-1)
                    total_unsqueeze += 1
            # print('total_unsqueeze: ', total_unsqueeze)

            return tuple(from_slices), tuple(to_slices)

        def _transfer(self, to_module: nn.Module, ids: Optional[List[int]]) -> Optional[List[int]]:
            if to_module not in self.outer_class.layers_mapping:
                return ids  # for example non-linearity - then pass ids
            from_module = self.outer_class.layers_mapping[to_module]
            if from_module is None:
                return ids
            # print("in: ", len(ids) if ids is not None else 0, ids)
            # print(to_module, from_module)

            from_slices, to_slices = self._get_slices(from_module.weight, to_module.weight, ids)
            # print(to_slices, from_slices)
            if to_module.weight is not None and from_module.weight is not None:
                to_module.weight[to_slices] = from_module.weight[from_slices]
            if to_module.bias is not None and from_module.bias is not None:
                to_ids = to_slices[0]
                from_ids = from_slices[0]
                if isinstance(to_ids, torch.Tensor):
                    to_ids = to_ids.flatten()
                if isinstance(from_ids, torch.Tensor):
                    from_ids = from_ids.flatten()
                to_module.bias[to_ids] = from_module.bias[from_ids]
            output_ids = from_slices[0]
            if isinstance(output_ids, slice):
                output_ids = torch.tensor([i for i in range(from_module.weight.shape[0])][output_ids])
            #     print("before out: ", [i for i in range(from_module.weight.shape[0])])
            #     print("slices: ", from_slices[0])
            # print("out: ", output_ids)
            return output_ids.flatten()

    def transfer(self, matched: List[Tuple[nn.Module, nn.Module]], *args, **kwargs) -> None:
        to_module = args[0] if len(args) > 0 else kwargs['context']['to_module']
        with torch.no_grad():
            self.create_layers_mapping(matched)
            to_module.eval()
            module = self.Module(torch.jit.trace(to_module, torch.randn(1, 3, 300, 300)), to_module, self)
            module.forward()

    def transfer_layer(self, tensor_from: torch.Tensor, tensor_to: torch.Tensor, *args, **kwargs) -> None:
        raise ValueError("Not implemented")
