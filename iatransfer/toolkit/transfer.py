import numpy as np
import torch
import torch.nn as nn


def transfer(from_model, to_model):
    def flatten_with_blocks(model):
        depth, conv_num, layers = 1, 0, []
        if isinstance(model, nn.Conv2d) or isinstance(model, nn.ConvTranspose2d):
            conv_num = 1
            layers = [model]
        else:
            for child in model.children():
                child_depth, child_conv_num, child_layers = flatten_with_blocks(child)
                if child_depth > 1 or (child_conv_num <= 1):
                    layers += child_layers
                    conv_num += child_conv_num
                    depth = max(depth, child_depth)
                else:
                    layers += [child_layers]
                    depth = 2
            if len(layers) == 0:
                layers = [model]

        return depth, conv_num, layers

    def compute_score(from_model, to_model):
        all_are_of_this_clazz = lambda layers, clazz: all([isinstance(x, clazz) for x in layers])
        score = 0
        layers = [from_model, to_model]
        if all_are_of_this_clazz(layers, list):
            score, _, _ = match_models(from_model, to_model)
        else:
            classes = [
                nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.Linear
            ]
            for clazz in classes:
                if all_are_of_this_clazz(layers, clazz):
                    score = 1
                    for x, y in zip(from_model.weight.shape, to_model.weight.shape):
                        score *= min(x / y, y / x)
                    break

        return score

    def match_models(from_model, to_model):
        m = len(from_model)
        n = len(to_model)
        dp = np.zeros((n + 1, m + 1))
        transition = np.zeros((n + 1, m + 1))  #
        scores = np.zeros((n + 1, m + 1))
        # reduction_coeff = 0.7
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                scores[i, j] = compute_score(from_model[j - 1], to_model[i - 1])

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i, j] = dp[i, j - 1]
                transition[i, j] = i
                current_reduction = 1
                cumulative_sum = 0
                for k in range(i, 0, -1):
                    cumulative_sum += scores[k, j]
                    # score = cumulative_sum*current_reduction+dp[k-1,j-1]
                    score = cumulative_sum / (i - k + 1) ** 0.5 + dp[k - 1, j - 1]
                    # let d be the number of layers matched in to_model to the current layer in from_model
                    # max possible score = d*f(d)
                    # we want d*f(d) to be increasing - adding more matchings should give better score
                    # we want f(d) to be decreasing - adding more matchings should give lower score per layer,
                    # thanks to it we encourage dynamic programming not to choose single layer all the time
                    if score > dp[i, j]:
                        dp[i, j] = score
                        transition[i, j] = k - 1
                    # current_reduction*=reduction_coeff

        matched = []
        matched_indices = []
        i, j = n, m

        while i > 0:
            t = transition[i, j]
            j -= 1
            from_model_layer_included = False
            while t < i:
                i -= 1
                if scores[i + 1, j + 1] > 0:
                    if isinstance(from_model[j], list) and isinstance(to_model[i], list):
                        _, sub_matched, sub_matched_indices = match_models(from_model[j], to_model[i])
                        matched.append(sub_matched)
                        matched_indices.append(sub_matched_indices)
                        matched_indices.append((j, i))
                    else:
                        matched.append((from_model[j], to_model[i]))
                        matched_indices.append((j, i))
                    from_model_layer_included = True
                else:
                    matched.append((None, to_model[i]))
                    matched_indices.append((None, i))
            if from_model_layer_included == False:
                matched_indices.append((j, None))
                matched.append((from_model[j], None))
        #     print(dp)
        #     print(transition)
        matched.reverse()
        matched_indices.reverse()
        return dp[n][m], matched, matched_indices

    def _transfer_tensors(from_tensor, to_tensor):
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

    def _transfer(matched):
        with torch.no_grad():
            for matching in matched:
                if isinstance(matching, list):
                    _transfer(matching)
                elif matching[0] is not None and matching[1] is not None:
                    _transfer_tensors(matching[0].weight, matching[1].weight)
                    _transfer_tensors(matching[0].bias, matching[1].bias)

    _, _, flattened_from_model = flatten_with_blocks(from_model)
    _, _, flattened_to_model = flatten_with_blocks(to_model)
    _, matched, _ = match_models(flattened_from_model, flattened_to_model)
    _transfer(matched)


def get_stats(model):
    return [layer.float().abs().mean() for layer in model.state_dict().values()]


if __name__ == '__main__':
    from efficientnet_pytorch import EfficientNet

    amodel = EfficientNet.from_pretrained('efficientnet-b0')
    bmodel = EfficientNet.from_pretrained('efficientnet-b3')

    stats_before = get_stats(bmodel)
    transfer(amodel, bmodel)
    stats_after = get_stats(bmodel)
    print('\n'.join(
        [str((x, y)) for x, y in zip(stats_before, stats_after)]
    ))
