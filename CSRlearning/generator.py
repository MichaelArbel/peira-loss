import torch

from csr_types import DataInfo

class LinearCCADataGenerator:
    """
    A generator that samples fresh Gaussian data on the fly.
    """
    def __init__(self, d1=50, d2=50, k=10, sigma=0.1, batch_size=64, 
                device='cpu',
                dtype=torch.float32,
                deterministic=False,
                empirical=False,
                sign_flip=False):

        self.d1 = d1
        self.d2 = d2
        self.k = k
        self.sigma = sigma
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.deterministic = deterministic
        self.empirical = empirical
        self.sign_flip = sign_flip

        # Pre-define fixed projection matrices (Ground Truth)
        # These define the linear relationship that the CCA should find.
        self.A1, self.A2, self.info = generate_correlated_matrices(k, d1, d2, sigma, 
                                                                    device, dtype, 
                                                                    sign_flip=self.sign_flip)

        #self.A1 = torch.randn(k, d1, device=device, dtype=dtype)
        #self.A2 = torch.randn(k, d2, device=device, dtype=dtype)
        if deterministic:
            if empirical:
                z = torch.randn(self.batch_size, self.k, device=self.device, dtype=self.dtype)
                eps_1 = torch.randn(self.batch_size, self.d1, device=self.device, dtype=self.dtype)
                eps_2 = torch.randn(self.batch_size, self.d2, device=self.device, dtype=self.dtype)

            
            else:

                z, eps_1, eps_2 = generate_deterministic_data(b=self.batch_size, d=[self.k, self.d1, self.d2], 
                                                          device=device, dtype=dtype)

            self.data = self.gen_data(z,eps_1,eps_2)


    def __iter__(self):
        """Allows use in: for x1, x2 in generator:"""
        return self

    def __next__(self):
        """Samples a fresh batch of correlated Gaussian data."""
        # 1. Sample shared latent variable z ~ N(0, I)
        if self.deterministic:
            return self.data

        z = torch.randn(self.batch_size, self.k, device=self.device, dtype=self.dtype)
        eps_1 = torch.randn(self.batch_size, self.d1, device=self.device, dtype=self.dtype)
        eps_2 = torch.randn(self.batch_size, self.d2, device=self.device, dtype=self.dtype)
        return self.gen_data(z,eps_1,eps_2)
        # 2. Project to high-dimensional views

    def gen_data(self,z,eps_1,eps_2):
        x1 = z @ self.A1.t() + self.sigma * eps_1
        x2 = z @ self.A2.t() + self.sigma * eps_2
        return x1, x2




def generate_correlated_matrices(dim_z, dim_x1, dim_x2, sigma, device, dtype, sign_flip=False):
    """
    Generates A1 and A2 such that svdvals(A2 @ A1.T) == target_svs
    """
    # Ensure target_svs is a double tensor
    S_vals_1 = torch.logspace(0,-4, steps=dim_z, device =device, dtype=dtype)
    S_vals_2 = 1.*S_vals_1#torch.logspace(-3,0, steps=dim_z, device =device, dtype=dtype)


    # S_vals_1 = S_vals_1[indices]
    # S_vals_2 = S_vals_2[indices]
    if sign_flip:
        S_vals_2[1:] = -S_vals_2[1:]  

    k = len(S_vals_1)
    

    spectrum = torch.abs(S_vals_1 * S_vals_2) / (
        torch.sqrt(S_vals_1 ** 2 + sigma ** 2) * torch.sqrt(S_vals_2 ** 2 + sigma ** 2)
    )
    spectrum, index = torch.sort(spectrum, descending=True)
    S_vals_1 = S_vals_1[index]
    S_vals_2 = S_vals_2[index]

    # 1. Create a shared orthogonal basis for the latent space
    # We use QR to get a clean orthonormal matrix
    Q, _ = torch.linalg.qr(torch.randn(dim_z, k, device = device, dtype=dtype))
    
    # 2. Create orthogonal projections for each view
    P1, _ = torch.linalg.qr(torch.randn(dim_x1, k, device = device, dtype=dtype))
    
    # 3. Construct A1 and A2
    # A1 = P1 @ Q.T
    # A2 = P2 @ diag(S) @ Q.T
    # Then A2 @ A1.T = P2 @ S @ Q.T @ Q @ P1.T = P2 @ S @ P1.T
    # The singular values of P2 @ S @ P1.T are exactly the elements of S.
    
    A1 = P1 @ torch.diag(S_vals_1)@ Q.t()
    A2 = P1 @ torch.diag(S_vals_2) @ Q.t()

    cov_1 = P1@torch.diag(S_vals_1**2 + sigma**2)@P1.t()
    cov_2 = P1@torch.diag(S_vals_2**2 + sigma**2)@P1.t()
    
    info = DataInfo(
        singular_vectors_1=P1,
        singular_vectors_2=P1,
        cov_1=cov_1,
        cov_2=cov_2,
        intra_spectrum_1=S_vals_1**2 + sigma**2,
        intra_spectrum_2=S_vals_2**2 + sigma**2,
        inter_spectrum=torch.abs(S_vals_2 * S_vals_1),
        spectrum=spectrum,
    )
    
    return A1, A2, info




def generate_deterministic_data(b=512, d=[10, 128, 128], device='cpu', dtype=torch.float64):
    # 1. Generate a random matrix
    # Use double precision to ensure z.T @ z / b is exactly I to ~1e-16
    d_tot = sum(d)
    raw_z = torch.randn(b, d_tot, dtype=torch.float64, device=device)
    
    # 2. Extract the orthogonal component
    # 'reduced' mode returns Q of shape (b, d) where Q.T @ Q = I
    Q, _ = torch.linalg.qr(raw_z, mode='reduced')
    
    # 3. Scale so that (z.T @ z) / b = I
    # Since Q.T @ Q = I, then (sqrt(b)*Q).T @ (sqrt(b)*Q) = b * I
    # Dividing that by b gives I.
    data = Q * torch.sqrt(torch.tensor(b, dtype=torch.float64))
    z = data[:, :d[0]]
    eps_1 = data[:, d[0]:d[1]+d[0]]
    eps_2 = data[:, d[1]+d[0]:]

    return z, eps_1, eps_2



