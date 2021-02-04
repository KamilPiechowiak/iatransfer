import torch

from iatransfer.toolkit.base_transfer import Transfer


class FullTransfer(Transfer):
    """Full tensor transfer algorithm for IAT.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def transfer_layer(self, tensor_from: torch.Tensor, tensor_to: torch.Tensor, *args, **kwargs) -> None:
        if tensor_from is None or tensor_to is None:
            return
        from_slices, to_slices = [], []
        for a, b in zip(tensor_from.shape, tensor_to.shape):
            if a < b:
                ids = torch.randint(a, (b,))
                ids[slice((b - a) // 2, -((b - a + 1) // 2))] = torch.arange(a)
                from_slices.append(ids)
            elif a > b:
                from_slices.append((a - b) // 2 + torch.arange(b))
            else:
                from_slices.append(torch.arange(a))
            to_slices.append(slice(0, b))

        total_unsqueeze = 0
        for i in range(len(from_slices) - 1, -1, -1):
            if isinstance(from_slices[i], torch.Tensor):
                for _ in range(total_unsqueeze):
                    from_slices[i] = from_slices[i].unsqueeze(-1)
                total_unsqueeze += 1

        tensor_to[tuple(to_slices)] = tensor_from[tuple(from_slices)]
