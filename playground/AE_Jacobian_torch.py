"""Code snippet for calculating jacobian in torch"""
import time
def jacobian(inputs, outputs):
    return torch.stack([torch.autograd.grad([outputs[:, i].sum()], inputs, retain_graph=True, create_graph=True)[0]
                        for i in range(outputs.size(1))], dim=-1)
latent_size = 2
kwargs = {"input_size": n, "latent_size": latent_size,"hid_layers":[1000, 200]}
encoder, decoder = utils.ML_utils.load_AE(AE.VanillaAE, settings.AE_MODEL, **kwargs)
w_0 = torch.zeros((1, latent_size), requires_grad = True)
u_0 = decoder(w_0)

outputs = u_0
inputs = w_0
num_grad = 10
t1 = time.time()
jac1 = torch.stack([torch.autograd.grad([outputs[:, i].sum()], inputs, retain_graph=True, create_graph=True)[0]
                    for i in range(num_grad)], dim=-1)
t2 = time.time()
print("time taken per grad: {:.4f}".format((t2 - t1)/num_grad) )
print(jac1)
print(jac1.shape)
exit()
jac = jacobian(w_0, u_0)
print(jac)
print(jac.shape)
