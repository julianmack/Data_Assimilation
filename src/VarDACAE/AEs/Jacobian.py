import torch

class Jacobian():
    """Jacobian helpers. These are ***SLOW*** methods of
    getting Network Jacobian (when AE.jac_explicit() is not implemented)"""
    @staticmethod
    def accumulated_slow_model(inputs, model, device=None):
        inputs.requires_grad = True
        if device == None:
            device = Jacobian.get_device()
        model.to(device)
        output = model.decode(inputs).flatten()

        print("inputs.shape", inputs.shape)
        print("output.shape", output.shape)

        return Jacobian.accumulated_slow(inputs, output)


    @staticmethod
    def accumulated_slow( inputs, outputs):
        """Computes a jacobian of two torch tensor.
        Uses a loop so linear time-complexity in dimension of output.

        This (slow) function is used to test the much faster .jac_explicit()
        functions in AutoEncoders.py"""
        dims = len(inputs.shape)

        if dims > 1:
            return Jacobian.__batched_jacobian_slow(inputs, outputs)
        else:
            return Jacobian.__no_batch_jacobian_slow(inputs, outputs)

    @staticmethod
    def __batched_jacobian_slow(inputs, outputs):
        dims = len(inputs.shape)
        return torch.transpose(torch.stack([torch.autograd.grad([outputs[:, i].sum()], inputs, retain_graph=True, create_graph=True)[0]
                            for i in range(outputs.size(1))], dim=-1), \
                            dims - 1, dims)
    @staticmethod
    def __no_batch_jacobian_slow(inputs, outputs):
        X = [torch.autograd.grad([outputs[i].sum()], inputs, retain_graph=True, create_graph=True)[0]
                            for i in range(outputs.size(0))]
        X = torch.stack(X, dim=-1)
        return X.t()
