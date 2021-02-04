import torch

from iatransfer.toolkit.base_transfer import Transfer


class ClipTransfer(Transfer):
    """Clip tensor transfer algorithm for IAT.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def transfer_layer(self, tensor_from: torch.Tensor, tensor_to: torch.Tensor, *args, **kwargs) -> None:
        if tensor_from is None or tensor_to is None:
            return
        from_slices, to_slices = [], []
        for a, b in zip(tensor_from.shape, tensor_to.shape):
            if a < b:
                from_slices.append(slice(0, a))
                to_slices.append(slice((b - a) // 2, -((b - a + 1) // 2)))
            elif a > b:
                from_slices.append(slice((a - b) // 2, -((a - b + 1) // 2)))
                to_slices.append(slice(0, b))
            else:
                from_slices.append(slice(0, a))
                to_slices.append(slice(0, b))
        tensor_to[tuple(to_slices)] = tensor_from[tuple(from_slices)]

