from iatransfer.toolkit.methods.transfer_method import TransferMethod
import torch

class CentersTransfer(TransferMethod):
    def transfer_tensors(self, from_tensor, to_tensor):
        if from_tensor is None or to_tensor is None:
            return
        from_slices, to_slices = [], []
        for a, b in zip(from_tensor.shape, to_tensor.shape):
            if a < b:
                from_slices.append(slice(0, a))
                to_slices.append(slice((b - a) // 2, -((b - a + 1) // 2)))
            elif a > b:
                from_slices.append(slice((a - b) // 2, -((a - b + 1) // 2)))
                to_slices.append(slice(0, b))
            else:
                from_slices.append(slice(0, a))
                to_slices.append(slice(0, b))
        to_tensor[tuple(to_slices)] = from_tensor[tuple(from_slices)]

    def transfer_matched(self, matched):
        with torch.no_grad():
            for matching in matched:
                if isinstance(matching, list):
                    self.transfer_matched(matching)
                elif matching[0] is not None and matching[1] is not None:
                    self.transfer_tensors(matching[0].weight, matching[1].weight)
                    self.transfer_tensors(matching[0].bias, matching[1].bias)