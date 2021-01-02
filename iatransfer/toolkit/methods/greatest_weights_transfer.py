from iatransfer.toolkit.methods.transfer_method import TransferMethod
import torch

class GreatestWeightsTransfer(TransferMethod):
    """
        :param reverse_priority: if set to True, choose channels with smallest weights' sum
        :param input_constrained_by_output: if set to True, choose input channels considering only already chosen output channels. If set to False, choose input channels taking into account all weights.
    """
    def __init__(self, reverse_priority = False, input_constrained_by_output = True):
        if reverse_priority:
            self.sgn = +1
        else:
            self.sgn = -1
        self.cut_from_tensor = input_constrained_by_output
    
    def transfer_with_greatest_weights(self, from_tensor, to_tensor, idx, from_slices, to_slices):
        dims = [i for i in range(len(from_tensor.shape))]
        dims.remove(idx)
        if from_tensor.shape[idx] > to_tensor.shape[idx]:
            # print("HERE", from_tensor.shape, to_tensor.shape)
            if len(dims) == 0:
                out_channels = list(enumerate(from_tensor))
            else: 
                out_channels = list(enumerate(from_tensor.abs().sum(dim=dims)))
            out_channels.sort(key=lambda w : self.sgn*w[1])
            c = to_tensor.shape[idx]
            from_ids = sorted([w[0] for w in out_channels[:c]])
            from_slices.append(from_ids)
            to_slices.append(slice(0, c))
            if self.cut_from_tensor:
                from_tensor = torch.index_select(from_tensor, idx, torch.tensor(from_ids))
        else:
            from_slices.append(slice(0, from_tensor.shape[idx]))
            to_slices.append(slice(0, from_tensor.shape[idx]))

    def get_slices(self, from_tensor, to_tensor):
        if from_tensor is None or to_tensor is None:
            return
        n = len(from_tensor.shape)
        from_slices, to_slices = [], []
        self.transfer_with_greatest_weights(from_tensor, to_tensor, 0, from_slices, to_slices)
        if n > 1:
            self.transfer_with_greatest_weights(from_tensor, to_tensor, 1, from_slices, to_slices)

        for a, b in zip(from_tensor.shape[2:], to_tensor.shape[2:]):
            if a < b:
                from_slices.append(slice(0, a))
                to_slices.append(slice((b - a) // 2, -((b - a + 1) // 2)))
            elif a > b:
                from_slices.append(slice((a - b) // 2, -((a - b + 1) // 2)))
                to_slices.append(slice(0, b))
            else:
                from_slices.append(slice(0, a))
                to_slices.append(slice(0, b))
        total_unsqueeze = 0
        for i in range(len(from_slices)-1, -1, -1):
            if isinstance(from_slices[i], list):
                from_slices[i] = torch.tensor(from_slices[i])
                for _ in range(total_unsqueeze):
                    from_slices[i] = from_slices[i].unsqueeze(-1)
                total_unsqueeze+=1
        return tuple(to_slices), tuple(from_slices)

    def transfer_matched(self, matched):
        with torch.no_grad():
            for matching in matched:
                if isinstance(matching, list):
                    self.transfer_matched(matching)
                elif matching[0] is not None and matching[1] is not None:
                    to_slices, from_slices = self.get_slices(matching[0].weight, matching[1].weight)
                    import sys
                    # print(to_slices, "<=", from_slices, file=sys.stderr)
                    if matching[1].weight is not None and matching[0].weight is not None:
                        # print(matching[1].weight[to_slices].shape)
                        # print(matching[0].weight.shape)
                        matching[1].weight[to_slices] = matching[0].weight[from_slices]
                    if matching[1].bias is not None and matching[0].bias is not None:
                        # print(matching[0].bias.shape)
                        # print(matching[1].bias.shape)
                        to_ids = to_slices[0]
                        from_ids = from_slices[0]
                        if isinstance(to_ids, torch.Tensor):
                            to_ids = to_ids.flatten()
                        if isinstance(from_ids, torch.Tensor):
                            from_ids = from_ids.flatten()
                        matching[1].bias[to_ids] = matching[0].bias[from_ids]