import torch, warnings

'''
This file contains functions to apply flat field weights and noise to projections, with pytorch library.
The flat field weights are computed with the get_flatfield_weights function, which gives a noise-free estimate of the flat field pattern.
The apply_flatfield_weights_and_noise function applies the flat field weights and noise to the projections.
The functions are optimized for speed, but they are not tested for when `requires_grad` is needed


Author: Domenico Iuso
Date: 16 Nov 2023
'''

def project_point(x, y, z, a, b, c):
    """
    Projects the points with coordinates x, y, z onto the plane
    defined by a*x + b*y + c*z = 1
    """ 
    assert False
    vector_norm = a*a + b*b + c*c
    normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
    point_in_plane = np.array([a, b, c]) / vector_norm

    points = np.column_stack((x, y, z))
    points_from_point_in_plane = points - point_in_plane
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                     normal_vector)
    proj_onto_plane = (points_from_point_in_plane - proj_onto_normal_vector[:, None]*normal_vector)

    return point_in_plane + proj_onto_plane

def project_point(src_pos, det_pos, n):
    d = -(n * det_pos).sum(dim=1)
    D = (n * src_pos).sum(dim=1) + d
    return src_pos - n*D.unsqueeze(1)

def get_flatfield_weights(src_pos, det_pos, det_u, det_v, det_cols, det_rows, J=1e20):
    '''
    Computes the flat field weights for a given source-detector configuration, so that pixels that are further away from the source are less sensitive to the source intensity.
    The weights tell how many photons are detected on a pixel, given the source intensity and the distance from the source.
    
    Assumptions:
    1) the pixels is far away for the source compared to the sizes of the pixel
        - The radiated photons do not change much the propagation direction within the pixel 
        - The radiative decay over distance do not change much within the pixel
    
    params:
        src_pos: (n_angles, 3) tensor of source positions
        det_pos: (n_angles, 3) tensor of detector positions
        det_u: (n_angles, 3) tensor of detector u vectors
        det_v: (n_angles, 3) tensor of detector v vectors
        det_cols: number of detector columns
        det_rows: number of detector rows
        J: source intensity (total number of photons per steradian)
    '''
    if src_pos.requires_grad or det_pos.requires_grad or det_u.requires_grad or det_v.requires_grad:
        warnings.warn("get_flatfield_weights: src_pos, det_pos, det_u, det_v should not require gradients")
    
    device, dtype = src_pos.device, src_pos.dtype
    n_angles = src_pos.shape[0]
    
    u = det_u / torch.linalg.vector_norm(det_u, ord=2, dim=1, keepdim=True)
    v = det_v / torch.linalg.vector_norm(det_v, ord=2, dim=1, keepdim=True)
    w = torch.cross(u, v, dim=1)

    u, v, w = u.unsqueeze(2), v.unsqueeze(2), w.unsqueeze(2)
    R_w_to_s = torch.cat((u, v, w), dim=2).inverse()
    ts_w_to_s = -R_w_to_s @ src_pos.unsqueeze(2)
    T_w_to_s = torch.cat((R_w_to_s, ts_w_to_s), dim=2)
    
    d = torch.linalg.vector_norm(src_pos-det_pos, ord=2, dim=1)
    
    opt_axis_pos = project_point(src_pos.unsqueeze(2), det_pos.unsqueeze(2), w)
    opt_shift = opt_axis_pos - det_pos.unsqueeze(2)
    opt_shift_s = (R_w_to_s @ opt_shift).squeeze(2)
    pu, pv = torch.linalg.vector_norm(det_u[0], ord=2), torch.linalg.vector_norm(det_v[0], ord=2)
    A = pu*pv
    
    # if opt_shift mean is not changing among projections, as well as for the src-det distance, then we can use only one projection for the flat field
    if opt_shift_s.allclose(opt_shift_s.mean(dim=0), atol=1e-3) and d.allclose(d.mean(), atol=1e-3):
        lu = torch.linspace(-pu*det_cols/2 - opt_shift_s[0,0], pu*det_cols/2 - opt_shift_s[0,0], det_cols, device=device)
        lv = torch.linspace(-pv*det_rows/2 - opt_shift_s[0,1], pv*det_rows/2 - opt_shift_s[0,1], det_rows, device=device)
        U, V = torch.meshgrid(lu, lv, indexing='xy')
        
        R = torch.linalg.vector_norm(opt_axis_pos[0]-src_pos[0], ord=2)
        weights = J*A*R / (R.pow(2) + U.pow(2) + V.pow(2)).pow(3/2)
        weights = weights.unsqueeze(0)
    else:
        weights = torch.zeros((n_angles, det_rows, det_cols), dtype=dtype, device=device)
        for i in range(n_angles):
            lu = torch.linspace(-pu*det_cols/2 - opt_shift_s[i,0], pu*det_cols/2 - opt_shift_s[i,0], det_cols, device=device)
            lv = torch.linspace(-pv*det_rows/2 - opt_shift_s[i,1], pv*det_rows/2 - opt_shift_s[i,1], det_rows, device=device)
            U, V = torch.meshgrid(lu, lv, indexing='xy')
            
            R = torch.linalg.vector_norm(opt_axis_pos[i]-src_pos[i], ord=2)
            weights[i] = J*A*R / (R.pow(2) + U.pow(2) + V.pow(2)).pow(3/2)
    return weights

def apply_flatfield_weights_and_noise(projs, flatfield_weights=None, noise='fast', generator=None):
    '''
    Applies flat field weights and noise to projections. The projections are meant to be flat-dark-field corrected, which means that the range is [0,1].
    The value is 0 when there is no photon detected on a pixel, and 1 when all the photons fired on that pixel are detected.
    
    params:
        projs: (n_angles, det_rows, det_cols) tensor of projections
        flatfield_weights: ([1 or n_angles], det_rows, det_cols) tensor of flat field weights. None means no flat field weights
        noise: 
        - 'fast': fast poisson noise generation using gaussian approximation
        - 'poisson': poisson noise generation
        - 'none': no noise
        generator: torch.Generator object for reproducible noise generation
    '''
    
    tmp = projs * flatfield_weights if flatfield_weights is not None else projs
    dtype = tmp.dtype
    if noise == 'fast':
        out_projs = torch.normal(tmp, torch.sqrt(tmp), generator=generator)
        out_projs = torch.clamp(out_projs, min=0)
    elif noise == 'poisson':
        out_projs = torch.poisson(tmp, generator=generator).to(dtype)
    else:
        pass
    return out_projs
    
def test_get_flat_field_weights():
    import matplotlib.pyplot as plt
    
    device = 0
    t_device = torch.device('cuda', device)
    dtype = torch.float32

    # geometry values
    ds = 1
    det_cols, det_rows = 2880//ds, 1880//ds
    pixel_length = 0.15*ds
    sod = 200
    mag = 2
    sdd = mag*sod
    n_angles = 5
    odd = sdd-sod
    proj_angles = torch.linspace(0,2*torch.pi,n_angles+1,device=t_device, dtype=dtype)[:-1]
    
    zeros = torch.zeros_like(proj_angles)
    ones  = torch.ones_like(proj_angles)
    rot_z = lambda angle: torch.stack([torch.cos(angle), -torch.sin(angle), zeros, zeros,
                                        torch.sin(angle),  torch.cos(angle), zeros, zeros,
                                        zeros,             zeros,            ones, zeros,
                                        zeros,             zeros,            zeros, ones], dim=1).reshape(-1,4,4)
    
    '''
    Source and detector positions, and detector versors.
    They should be (n_angles, :) tensors. As we are using homogeneous coordinates, we add a 1 at the end of each vector which is removed after the rotation.
    All this is purely exemplaty and not needed for applying the flat field weights and noise.
    '''
    src_pos    = rot_z(proj_angles) @ torch.tensor([-sod, 0, 0, 1], dtype=dtype, device=t_device)
    det_pos    = rot_z(proj_angles) @ torch.tensor([odd,  0, 0, 1], dtype=dtype, device=t_device)
    det_vers_1 = rot_z(proj_angles) @ torch.tensor([0, -pixel_length, 0, 1], dtype=dtype, device=t_device)
    det_vers_2 = rot_z(proj_angles) @ torch.tensor([0, 0, -pixel_length, 1], dtype=dtype, device=t_device)
    src_pos, det_pos, det_vers_1, det_vers_2 = src_pos[:,:3], det_pos[:,:3], det_vers_1[:,:3], det_vers_2[:,:3]


    # Calculation of flat field weights. The noise will be inherently dependant on the source intensity J, so we set it to a very high value as in the real case.
    # J is the total number of photons per steradian
    flatfield_weights = get_flatfield_weights(src_pos, det_pos, det_vers_1, det_vers_2, det_cols, det_rows, J=1e17)
    plt.figure()
    plt.imshow(flatfield_weights[0].cpu())
    plt.title('Flat field weights')
    
    # Simulate few projections
    simulated_projections = torch.zeros((n_angles, det_rows, det_cols), dtype=dtype, device=t_device)
    simulated_projections[:,det_rows//2-200:det_rows//2+200,det_cols//2-200:det_cols//2+200] = 1e1
    corrected_projs = torch.exp(-simulated_projections)
    
    # Apply flat field weights and noise to the projections
    out_projs = apply_flatfield_weights_and_noise(projs=corrected_projs, flatfield_weights=flatfield_weights, noise='fast')
    plt.figure()
    plt.imshow(out_projs[0].cpu())
    plt.title('Projections with Flat field pattern and noise')
    plt.show()
    pass



if __name__ == "__main__":
    test_get_flat_field_weights()