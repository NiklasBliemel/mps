function custom_mpo(::Type{ElT}, sites::Vector{<:Index}, linkdim::Vector{Int}) where {ElT<:Number}
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    if N == 0
      return MPO()
    elseif N == 1
      v[1] = ITensor(ElT, dag(sites[1]), sites[1]')
      return MPO(v)
    end
    l = [Index(linkdim[ii], "Link,l=$ii") for ii in 1:(N - 1)]
    for ii in eachindex(sites)
      s = sites[ii]
      if ii == 1
        v[ii] = ITensor(ElT, dag(s), s', l[ii])
      elseif ii == N
        v[ii] = ITensor(ElT, dag(l[ii - 1]), dag(s), s')
      else
        v[ii] = ITensor(ElT, dag(l[ii - 1]), dag(s), s', l[ii])
      end
    end
  return MPO(v)
end

function Diff2_fill(mpo, J, alpha_, beta, gamma)
    Id = zeros(eltype(J), size(J))
    Id[:,:] = I(size(J)[1])
    Jp = transpose(J)
    P = J + Jp

    for (i,v) in enumerate(mpo)
        mat = fill(0.0, dim.(inds(v)))
        if i == 1
            mat[:,:,1] = Id
            mat[:,:,2] = P
            mat[:,:,3] = P
        elseif i == R
            mat[1,:,:] = (alpha_*Id + beta * J + gamma * Jp)
            mat[2,:,:] = gamma * J
            mat[3,:,:] = beta * Jp
        else
            mat[1,:,:,1] = Id
            mat[1,:,:,2] = Jp
            mat[1,:,:,3] = J 
            mat[2,:,:,2] = J 
            mat[3,:,:,3] = Jp
        end
        mpo[i] = ITensor(mat, inds(v))
    end
    return mpo
end

function Diff_1_2_x(h, sites)
    out = custom_mpo(Float64, sites, [3 for i in 1:length(s)-1])
    alpha_ = 0
    beta = 1 / (2*h)
    gamma = -1 / (2*h)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[2,1] = 1
    J[4,3] = 1
    out = Diff2_fill(out, J, alpha_, beta, gamma)
    return out
end

function Diff_1_2_y(h, sites)
    out = custom_mpo(Float64, sites, [3 for i in 1:length(s)-1])
    alpha_ = 0
    beta = 1 / (2*h)
    gamma = -1 / (2*h)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[3,1] = 1
    J[4,2] = 1
    out = Diff2_fill(out, J, alpha_, beta, gamma)
    return out
end

function Diff_2_2_x(h, sites)
    out = custom_mpo(Float64, sites, [3 for i in 1:length(s)-1])
    alpha_ = -2 / h^2
    beta = 1 / h^2
    gamma = 1 / h^2
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[2,1] = 1
    J[4,3] = 1
    out = Diff2_fill(out, J, alpha_, beta, gamma)
    return out
end

function Diff_2_2_y(h, sites)
    out = custom_mpo(Float64, sites, [3 for i in 1:length(s)-1])
    alpha_ = -2 / h^2
    beta = 1 / h^2
    gamma = 1 / h^2
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[3,1] = 1
    J[4,2] = 1
    out = Diff2_fill(out, J, alpha_, beta, gamma)
    return out
end

function Diff8_1_fill(mpo, J, a::Vector{Float64})
    Id = zeros(eltype(J), size(J))
    Id[:,:] = I(size(J)[1])
    Jp = transpose(J)
    P = J + Jp

    for (i,v) in enumerate(mpo)
        mat = fill(0.0, dim.(inds(v)))
        if i == 1
            mat[:,:,1] = Id
            mat[:,:,2] = P
            mat[:,:,3] = P
        elseif i == R - 1
            mat[1,:,:,1] = Id
            mat[1,:,:,2] = Jp
            mat[1,:,:,3] = J 
            mat[2,:,:,2] = J 
            mat[3,:,:,3] = Jp
            mat[2,:,:,4] = Id
            mat[3,:,:,5] = Id
        elseif i == R
            mat[1,:,:] = a[1]*Id + a[2]*Jp + a[3]*J
            mat[2,:,:] = a[2]*J  + a[4]*Id + a[5]*Jp
            mat[3,:,:] = a[3]*Jp - a[4]*Id - a[5]*J
            mat[4,:,:] = a[5]*J  + a[6]*Id
            mat[5,:,:] =-a[5]*Jp - a[6]*Id
        else
            mat[1,:,:,1] = Id
            mat[1,:,:,2] = Jp
            mat[1,:,:,3] = J 
            mat[2,:,:,2] = J 
            mat[3,:,:,3] = Jp
        end
        mpo[i] = ITensor(mat, inds(v))
    end
    return mpo
end

function Diff8_2_fill(mpo, J, a::Vector{Float64})
    Id = zeros(eltype(J), size(J))
    Id[:,:] = I(size(J)[1])
    Jp = transpose(J)
    P = J + Jp

    for (i,v) in enumerate(mpo)
        mat = fill(0.0, dim.(inds(v)))
        if i == 1
            mat[:,:,1] = Id
            mat[:,:,2] = P
            mat[:,:,3] = P
        elseif i == R - 1
            mat[1,:,:,1] = Id
            mat[1,:,:,2] = Jp
            mat[1,:,:,3] = J 
            mat[2,:,:,2] = J 
            mat[3,:,:,3] = Jp
            mat[2,:,:,4] = Id
            mat[3,:,:,5] = Id
        elseif i == R
            mat[1,:,:] = a[1]*Id + a[2]*Jp + a[3]*J
            mat[2,:,:] = a[2]*J  + a[4]*Id + a[5]*Jp
            mat[3,:,:] = a[3]*Jp + a[4]*Id + a[5]*J
            mat[4,:,:] = a[5]*J  + a[6]*Id
            mat[5,:,:] = a[5]*Jp + a[6]*Id
        else
            mat[1,:,:,1] = Id
            mat[1,:,:,2] = Jp
            mat[1,:,:,3] = J 
            mat[2,:,:,2] = J 
            mat[3,:,:,3] = Jp
        end
        mpo[i] = ITensor(mat, inds(v))
    end
    return mpo
end

function Diff_1_8_x(h, sites)
    out = custom_mpo(Float64, sites, [i != length(s)-1 ? 3 : 5 for i in 1:length(s)-1])
    a = [0, -4/5, 4/5, 1/5, -4/105, 1/280] / h
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[2,1] = 1
    J[4,3] = 1
    out = Diff8_1_fill(out, J, a)
    return out
end

function Diff_1_8_y(h, sites)
    out = custom_mpo(Float64, sites, [i != length(s)-1 ? 3 : 5 for i in 1:length(s)-1])
    a = [0, -4/5, 4/5, 1/5, -4/105, 1/280] / h
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[3,1] = 1
    J[4,2] = 1
    out = Diff8_1_fill(out, J, a)
    return out
end

function Diff_2_8_x(h, sites)
    out = custom_mpo(Float64, sites, [i != length(s)-1 ? 3 : 5 for i in 1:length(s)-1])
    a = [-205/72, 8/5, 8/5, -1/5, 8/315, -1/560] / (h^2)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[2,1] = 1
    J[4,3] = 1
    out = Diff8_2_fill(out, J, a)
    return out
end
function Diff_2_8_y(h, sites)
    out = custom_mpo(Float64, sites, [i != length(s)-1 ? 3 : 5 for i in 1:length(s)-1])
    a = [-205/72, 8/5, 8/5, -1/5, 8/315, -1/560] / (h^2)
    d = dim(sites[1])
    J = fill(0.0, (d,d))
    J[3,1] = 1
    J[4,2] = 1
    out = Diff8_2_fill(out, J, a)
    return out
end

function mps_to_array_2D(mps)
    R = length(mps)
    plot_tensor = array(prod(mps[:]))
    plot_tensor = reshape(plot_tensor, ntuple(i -> 2, 2*R)...)
    perm = Vector{Int}(undef, 2*R)
    for i in 1:R
        perm[R + i] = 2*R - (2 * i - 1)
        perm[i] = 2*R - (2 * i - 2)
    end
    plot_tensor = permutedims(plot_tensor, perm)
    return reshape(plot_tensor, 2^R, 2^R)
end

function plot_mps(mps; grid_size=8, trunc=1e-5, save_name="")
    R = length(mps)
    grid_size = min(grid_size, R)
    plot_tensor = array(prod(mps[:]))
    plot_tensor = reshape(plot_tensor, ntuple(i -> 2, 2*R)...)
    perm = Vector{Int}(undef, 2*R)
    for i in 1:R
        perm[R + i] = 2*R - (2 * i - 1)
        perm[i] = 2*R - (2 * i - 2)
    end
    plot_tensor = permutedims(plot_tensor, perm)
    plot_tensor = reshape(plot_tensor, 2^R, 2^R)
    dummy = 1:2^(R - grid_size):2^R
    mps_vals = plot_tensor[dummy,dummy]
    mps_vals[abs.(mps_vals) .< trunc] .= 0.
    xvals = collect(range(0, 1, 2^grid_size))
    yvals = collect(range(0, 1, 2^grid_size))
        
    if save_name == ""
        contour(xvals, yvals, mps_vals, fill=true)
    else
        p = contour(xvals, yvals, mps_vals, fill=true)
        savefig(p, save_name)
    end
end

function find_max_on_2D_grid(func, R, x_min=0., x_max=1., y_min=0., y_max=1.)
    num_points_per_dim = 2^R
    x_range = range(x_min, x_max, num_points_per_dim)
    y_range = range(y_min, y_max, num_points_per_dim)
    max_val = -Inf
    for x in x_range
        for y in y_range
            max_val = max(max_val, func(x,y))
        end
    end
    return max_val
end

function contract_except_center(mps_left, mps_right, center)
    result_l = ITensor(1.0)
    result_r = ITensor(1.0)

    mps_right_p = prime(linkinds, mps_right)

    for i in 1:center-1
        result_l *= dag(mps_left[i]) * mps_right_p[i]
    end

    for i in length(s):-1:center+1
        result_r *= dag(mps_left[i]) * mps_right_p[i]
    end
    out = noprime(mps_right_p[center] * result_l * result_r)
    return permute(out, inds(mps_left[center]), allow_alias=true)
end

function make_beta(v1, v2, a1, a2, b1, b2, center, delta_t, nu, d1x, d1y, d2x, d2y, del, max_bond)
    b_dotx = MPO(*(del, b1'')[:])
    b_doty = MPO(*(del, b2'')[:])
    d1x_b1 = apply(d1x, b1; alg="naive", maxdim=max_bond)
    d1x_b2 = apply(d1x, b2; alg="naive", maxdim=max_bond)
    d1y_b1 = apply(d1y, b1; alg="naive", maxdim=max_bond)
    d1y_b2 = apply(d1y, b2; alg="naive", maxdim=max_bond)
    b_dotx_b1 = apply(b_dotx, b1; alg="naive", maxdim=max_bond)
    b_dotx_b2 = apply(b_dotx, b2; alg="naive", maxdim=max_bond)
    b_doty_b1 = apply(b_doty, b1; alg="naive", maxdim=max_bond)
    b_doty_b2 = apply(b_doty, b2; alg="naive", maxdim=max_bond)

    B_1 = 0.5 * (apply(b_dotx, d1x_b1; alg="naive", maxdim=max_bond) + apply(b_doty, d1y_b1; alg="naive", maxdim=max_bond))
    B_1 += 0.5 * (apply(d1x, b_dotx_b1; alg="naive", maxdim=max_bond) + apply(d1y, b_doty_b1; alg="naive", maxdim=max_bond))
    B_1 += -nu * (apply(d2x, b1; alg="naive", maxdim=max_bond) + apply(d2y, b1; alg="naive", maxdim=max_bond))

    B_2 = 0.5 * (apply(b_dotx, d1x_b2; alg="naive", maxdim=max_bond) + apply(b_doty, d1y_b2; alg="naive", maxdim=max_bond))
    B_2 += 0.5 * (apply(d1x, b_dotx_b2; alg="naive", maxdim=max_bond) + apply(d1y, b_doty_b2; alg="naive", maxdim=max_bond))
    B_2 += -nu * (apply(d2x, b2; alg="naive", maxdim=max_bond) + apply(d2y, b2; alg="naive", maxdim=max_bond))

    out_1 = contract_except_center(v1, a1 - delta_t * B_1, center)
    out_2 = contract_except_center(v2, a2 - delta_t * B_2, center)

    return vcat(vec(array(out_1)), vec(array(out_2)))
end

function get_c_vec(v1, v2, center)
    return vcat(vec(array(v1[center])), vec(array(v2[center])))
end

function insert_c_vec(c_vec, v1, v2, center)
    length_1 = prod(size(v1[center]))
    length_2 = prod(size(v2[center]))
    out1 = deepcopy(v1)
    out2 = deepcopy(v2)
    out1[center] = ITensor(c_vec[1:length_1], inds(v1[center]))
    out2[center] = ITensor(c_vec[length_1+1:length_1+length_2], inds(v2[center]))
    return out1, out2
end

function right_left_C(left_tn, right_tn, C, out_inds)
    out = noprime(C * left_tn * right_tn)
    return permute(out, out_inds)
end

function right_left_C_W(left_tn, right_tn, C, W, out_inds)
    out = C * W
    out = noprime(out * left_tn * right_tn)
    return permute(out, out_inds)
end

function right_left_C(left_tn, right_tn, C, out_inds)
    out = noprime(C * left_tn * right_tn)
    return permute(out, out_inds)
end

function make_left_right_mps_mps(mps_A, mps_B, center)
    R = length(mps_A)

    mps_A_p = prime(linkinds, mps_A)

    left_tn = Vector{ITensor}(undef, R)
    right_tn = Vector{ITensor}(undef, R)
    left_tn[1] = ITensor(1.0)
    right_tn[R] = ITensor(1.0)

    for i in 1:center-1
        left_tn[i+1] = left_tn[i] * dag(mps_A_p[i]) * mps_B[i]
    end
    for i in length(s):-1:center+1
        right_tn[i-1] = right_tn[i] * dag(mps_A_p[i]) * mps_B[i]
    end
    return left_tn, right_tn
end

function make_left_right_mps_mpo(mps_A, mps_B, mpo, center)
    R = length(mps_A)

    mps_A_p = prime(mps_A)

    left_tn = Vector{ITensor}(undef, R)
    right_tn = Vector{ITensor}(undef, R)
    left_tn[1] = ITensor(1.0)
    right_tn[R] = ITensor(1.0)

    for i in 1:center-1
        left_tn[i+1] = left_tn[i] * dag(mps_A_p[i]) * mpo[i] * mps_B[i]
    end

    for i in R:-1:center+1
        right_tn[i-1] = right_tn[i] * dag(mps_A_p[i]) * mpo[i] * mps_B[i]
    end
    return left_tn, right_tn
end

function update_tn_mps_mps(left_tn, mps_A, mps_B)
    mps_A_p = prime(mps_A, "link")
    return left_tn * dag(mps_A_p) * mps_B
end

function update_tn_mps_mpo(left_tn, mps_A, mps_B, mpo)
    mps_A_p = prime(mps_A)
    return left_tn * dag(mps_A_p) * mpo * mps_B
end

function optimize_core(center, v1, v2, a1, a2, b1, b2, delta_t, nu, d1x, d1y, d2x, d2y, del, dx_dx, dy_dy, dx_dy, max_bond; info=false)
    
    orthogonalize!(v1, center)
    orthogonalize!(v2, center)

    if info
        t0 = time()
    end

    beta = make_beta(v1, v2, a1, a2, b1, b2, center, delta_t, nu, d1x, d1y, d2x, d2y, del, max_bond)
    v1_dx_dx_v1_left, v1_dx_dx_v1_right = make_left_right_mps_mpo(v1, v1, dx_dx, center)
    v1_dx_dy_v2_left, v1_dx_dy_v2_right = make_left_right_mps_mpo(v1, v2, dx_dy, center)
    v2_dx_dy_v1_left, v2_dx_dy_v1_right = make_left_right_mps_mpo(v2, v1, dx_dy, center)
    v2_dy_dy_v2_left, v2_dy_dy_v2_right = make_left_right_mps_mpo(v2, v2, dy_dy, center)
    
    if info
        t1 = time()
        println("Time to construct LR TensorNetworks: $(t1 - t0) sek")
    end

    c_vec = get_c_vec(v1, v2, center)
    c1_inds = inds(v1[center])
    c2_inds = inds(v2[center])
    c1_length = prod(size(v1[center]))
    c2_length = prod(size(v2[center]))

    function A_function(c_vec)
        c_1 = ITensor(c_vec[1:c1_length], c1_inds)
        c_2 = ITensor(c_vec[c1_length+1:c1_length+c2_length], c2_inds)
        out_1 = right_left_C_W(v1_dx_dx_v1_left[center], v1_dx_dx_v1_right[center], c_1, dx_dx[center], c1_inds)
        out_1 += right_left_C_W(v1_dx_dy_v2_left[center], v1_dx_dy_v2_right[center], c_2, dx_dy[center], c1_inds)
        out_2 = right_left_C_W(v2_dx_dy_v1_left[center], v2_dx_dy_v1_right[center], c_1, dx_dy[center], c2_inds)
        out_2 += right_left_C_W(v2_dy_dy_v2_left[center], v2_dy_dy_v2_right[center], c_2, dy_dy[center], c2_inds)
        return c_vec - penalty_coefficient * delta_t^2 * vcat(vec(array(out_1)), vec(array(out_2)))
    end
    A = FunctionMap{Float64,false}(A_function, c1_length + c2_length)

    if info
        t0 = time()
        history = cg!(c_vec, A, beta, abstol=1e-5, verbose=false, maxiter=100, log=info)[2]
        t1 = time()
        println(history)
        println("Center $center in $(t1 - t0) sek, Error: $(norm(A(c_vec)-beta))")
    else
        cg!(c_vec, A, beta, abstol=1e-5, verbose=false, maxiter=100, log=false)
    end
    return insert_c_vec(c_vec, v1, v2, center)
end

function optimize_sweep(v1, v2, A1, A2, B1, B2, delta_t, nu, d1x, d1y, d2x, d2y, del, dx_dx, dy_dy, dx_dy, max_bond; info=false, eps=1e-5)

    v1_copy = copy(v1)
    v2_copy = copy(v2)
    a1 = copy(A1)
    a2 = copy(A2)
    b1 = copy(B1)
    b2 = copy(B2)

    if info
        t0 = time()
    end

    R = length(v1_copy)

    orthogonalize!(v1_copy, 1)
    orthogonalize!(v2_copy, 1)

    b_dotx = MPO(*(del, b1'')[:])
    b_doty = MPO(*(del, b2'')[:])
    b_dotx_d1x = apply(b_dotx, d1x)
    b_doty_d1y = apply(b_doty, d1y)
    d1x_b_dotx = apply(d1x, b_dotx)
    d1y_b_doty = apply(d1y, b_doty)

    v1_a1_left, v1_a1_right = make_left_right_mps_mps(v1_copy, a1, 1)
    v1_d2x_b1_left, v1_d2x_b1_right = make_left_right_mps_mpo(v1_copy, b1, d2x, 1)
    v1_d2y_b1_left, v1_d2y_b1_right = make_left_right_mps_mpo(v1_copy, b1, d2y, 1)
    v1_b_dotx_d1x_b1_left, v1_b_dotx_d1x_b1_right = make_left_right_mps_mpo(v1_copy, b1, b_dotx_d1x, 1)
    v1_b_doty_d1y_b1_left, v1_b_doty_d1y_b1_right = make_left_right_mps_mpo(v1_copy, b1, b_doty_d1y, 1)
    v1_d1x_b_dotx_b1_left, v1_d1x_b_dotx_b1_right = make_left_right_mps_mpo(v1_copy, b1, d1x_b_dotx, 1)
    v1_d1y_b_doty_b1_left, v1_d1y_b_doty_b1_right = make_left_right_mps_mpo(v1_copy, b1, d1y_b_doty, 1)

    v2_a2_left, v2_a2_right = make_left_right_mps_mps(v2_copy, a2, 1)
    v2_d2x_b2_left, v2_d2x_b2_right = make_left_right_mps_mpo(v2_copy, b2, d2x, 1)
    v2_d2y_b2_left, v2_d2y_b2_right = make_left_right_mps_mpo(v2_copy, b2, d2y, 1)
    v2_b_dotx_d1x_b2_left, v2_b_dotx_d1x_b2_right = make_left_right_mps_mpo(v2_copy, b2, b_dotx_d1x, 1)
    v2_b_doty_d1y_b2_left, v2_b_doty_d1y_b2_right = make_left_right_mps_mpo(v2_copy, b2, b_doty_d1y, 1)
    v2_d1x_b_dotx_b2_left, v2_d1x_b_dotx_b2_right = make_left_right_mps_mpo(v2_copy, b2, d1x_b_dotx, 1)
    v2_d1y_b_doty_b2_left, v2_d1y_b_doty_b2_right = make_left_right_mps_mpo(v2_copy, b2, d1y_b_doty, 1)

    v1_dx_dx_v1_left, v1_dx_dx_v1_right = make_left_right_mps_mpo(v1_copy, v1_copy, dx_dx, 1)
    v1_dx_dy_v2_left, v1_dx_dy_v2_right = make_left_right_mps_mpo(v1_copy, v2_copy, dx_dy, 1)

    v2_dx_dy_v1_left, v2_dx_dy_v1_right = make_left_right_mps_mpo(v2_copy, v1_copy, dx_dy, 1)
    v2_dy_dy_v2_left, v2_dy_dy_v2_right = make_left_right_mps_mpo(v2_copy, v2_copy, dy_dy, 1)

    c_vec = get_c_vec(v1_copy, v2_copy, 1)
    c1_inds = inds(v1_copy[1])
    c2_inds = inds(v2_copy[1])
    c1_length = prod(size(v1_copy[1]))
    c2_length = prod(size(v2_copy[1]))

    beta_1 = right_left_C(v1_a1_left[1], v1_a1_right[1], a1[1], c1_inds)
    beta_1 += nu * right_left_C_W(v1_d2x_b1_left[1], v1_d2x_b1_right[1], b1[1], d2x[1], c1_inds)
    beta_1 += nu * right_left_C_W(v1_d2y_b1_left[1], v1_d2y_b1_right[1], b1[1], d2y[1], c1_inds)
    beta_1 += -delta_t * 0.5 * right_left_C_W(v1_b_dotx_d1x_b1_left[1], v1_b_dotx_d1x_b1_right[1], b1[1], b_dotx_d1x[1], c1_inds)
    beta_1 += -delta_t * 0.5 * right_left_C_W(v1_b_doty_d1y_b1_left[1], v1_b_doty_d1y_b1_right[1], b1[1], b_doty_d1y[1], c1_inds)
    beta_1 += -delta_t * 0.5 * right_left_C_W(v1_d1x_b_dotx_b1_left[1], v1_d1x_b_dotx_b1_right[1], b1[1], d1x_b_dotx[1], c1_inds)
    beta_1 += -delta_t * 0.5 * right_left_C_W(v1_d1y_b_doty_b1_left[1], v1_d1y_b_doty_b1_right[1], b1[1], d1y_b_doty[1], c1_inds)
    
    beta_2 = right_left_C(v2_a2_left[1], v2_a2_right[1], a2[1], c2_inds)
    beta_2 += nu * right_left_C_W(v2_d2x_b2_left[1], v2_d2x_b2_right[1], b2[1], d2x[1], c2_inds)
    beta_2 += nu * right_left_C_W(v2_d2y_b2_left[1], v2_d2y_b2_right[1], b2[1], d2y[1], c2_inds)
    beta_2 += -delta_t * 0.5 * right_left_C_W(v2_b_dotx_d1x_b2_left[1], v2_b_dotx_d1x_b2_right[1], b2[1], b_dotx_d1x[1], c2_inds)
    beta_2 += -delta_t * 0.5 * right_left_C_W(v2_b_doty_d1y_b2_left[1], v2_b_doty_d1y_b2_right[1], b2[1], b_doty_d1y[1], c2_inds)
    beta_2 += -delta_t * 0.5 * right_left_C_W(v2_d1x_b_dotx_b2_left[1], v2_d1x_b_dotx_b2_right[1], b2[1], d1x_b_dotx[1], c2_inds)
    beta_2 += -delta_t * 0.5 * right_left_C_W(v2_d1y_b_doty_b2_left[1], v2_d1y_b_doty_b2_right[1], b2[1], d1y_b_doty[1], c2_inds)

    beta = vcat(vec(array(beta_1)), vec(array(beta_2)))

    E_0 = 1e-10
    E_1 = 2*eps
    run = 0

    if info
        println("Setup in $(time()-t0) sek")
        t0 = time()
        t1 = time()
    end

    while abs((E_1 - E_0) / E_0) > eps
        run += 1
        for center in 2:R

            function A_function_center_left(c_vec)
                c_1 = ITensor(c_vec[1:c1_length], c1_inds)
                c_2 = ITensor(c_vec[c1_length+1:c1_length+c2_length], c2_inds)
                out_1 = right_left_C_W(v1_dx_dx_v1_left[center-1], v1_dx_dx_v1_right[center-1], c_1, dx_dx[center-1], c1_inds)
                out_1 += right_left_C_W(v1_dx_dy_v2_left[center-1], v1_dx_dy_v2_right[center-1], c_2, dx_dy[center-1], c1_inds)
                out_2 = right_left_C_W(v2_dx_dy_v1_left[center-1], v2_dx_dy_v1_right[center-1], c_1, dx_dy[center-1], c2_inds)
                out_2 += right_left_C_W(v2_dy_dy_v2_left[center-1], v2_dy_dy_v2_right[center-1], c_2, dy_dy[center-1], c2_inds)
                return c_vec - penalty_coefficient * delta_t^2 * vcat(vec(array(out_1)), vec(array(out_2)))
            end
            A = FunctionMap{Float64,false}(A_function_center_left, c1_length + c2_length)

            cg!(c_vec, A, beta, reltol=1e-6, abstol=1e-12, verbose=false, maxiter=100, log=false)
            v1_copy, v2_copy = insert_c_vec(c_vec, v1_copy, v2_copy, center-1)

            orthogonalize!(v1_copy, center)
            orthogonalize!(v2_copy, center)

            c_vec = get_c_vec(v1_copy, v2_copy, center)
            c1_inds = inds(v1_copy[center])
            c2_inds = inds(v2_copy[center])
            c1_length = prod(size(v1_copy[center]))
            c2_length = prod(size(v2_copy[center]))

            v1_a1_left[center] = update_tn_mps_mps(v1_a1_left[center-1], v1_copy[center-1], a1[center-1])
            v1_d2x_b1_left[center] = update_tn_mps_mpo(v1_d2x_b1_left[center-1], v1_copy[center-1], b1[center-1], d2x[center-1])
            v1_d2y_b1_left[center] = update_tn_mps_mpo(v1_d2y_b1_left[center-1], v1_copy[center-1], b1[center-1], d2y[center-1])
            v1_b_dotx_d1x_b1_left[center] = update_tn_mps_mpo(v1_b_dotx_d1x_b1_left[center-1], v1_copy[center-1], b1[center-1], b_dotx_d1x[center-1])
            v1_b_doty_d1y_b1_left[center] = update_tn_mps_mpo(v1_b_doty_d1y_b1_left[center-1], v1_copy[center-1], b1[center-1], b_doty_d1y[center-1])
            v1_d1x_b_dotx_b1_left[center] = update_tn_mps_mpo(v1_d1x_b_dotx_b1_left[center-1], v1_copy[center-1], b1[center-1], d1x_b_dotx[center-1])
            v1_d1y_b_doty_b1_left[center] = update_tn_mps_mpo(v1_d1y_b_doty_b1_left[center-1], v1_copy[center-1], b1[center-1], d1y_b_doty[center-1])

            v2_a2_left[center] = update_tn_mps_mps(v2_a2_left[center-1], v2_copy[center-1], a2[center-1])
            v2_d2x_b2_left[center] = update_tn_mps_mpo(v2_d2x_b2_left[center-1], v2_copy[center-1], b2[center-1], d2x[center-1])
            v2_d2y_b2_left[center] = update_tn_mps_mpo(v2_d2y_b2_left[center-1], v2_copy[center-1], b2[center-1], d2y[center-1])
            v2_b_dotx_d1x_b2_left[center] = update_tn_mps_mpo(v2_b_dotx_d1x_b2_left[center-1], v2_copy[center-1], b2[center-1], b_dotx_d1x[center-1])
            v2_b_doty_d1y_b2_left[center] = update_tn_mps_mpo(v2_b_doty_d1y_b2_left[center-1], v2_copy[center-1], b2[center-1], b_doty_d1y[center-1])
            v2_d1x_b_dotx_b2_left[center] = update_tn_mps_mpo(v2_d1x_b_dotx_b2_left[center-1], v2_copy[center-1], b2[center-1], d1x_b_dotx[center-1])
            v2_d1y_b_doty_b2_left[center] = update_tn_mps_mpo(v2_d1y_b_doty_b2_left[center-1], v2_copy[center-1], b2[center-1], d1y_b_doty[center-1])

            v1_dx_dx_v1_left[center] = update_tn_mps_mpo(v1_dx_dx_v1_left[center-1], v1_copy[center-1], v1_copy[center-1], dx_dx[center-1])
            v1_dx_dy_v2_left[center] = update_tn_mps_mpo(v1_dx_dy_v2_left[center-1], v1_copy[center-1], v2_copy[center-1], dx_dy[center-1])
            v2_dx_dy_v1_left[center] = update_tn_mps_mpo(v2_dx_dy_v1_left[center-1], v2_copy[center-1], v1_copy[center-1], dx_dy[center-1])
            v2_dy_dy_v2_left[center] = update_tn_mps_mpo(v2_dy_dy_v2_left[center-1], v2_copy[center-1], v2_copy[center-1], dy_dy[center-1])
            
            beta_1 = right_left_C(v1_a1_left[center], v1_a1_right[center], a1[center], c1_inds)
            beta_1 += delta_t * nu * right_left_C_W(v1_d2x_b1_left[center], v1_d2x_b1_right[center], b1[center], d2x[center], c1_inds)
            beta_1 += delta_t * nu * right_left_C_W(v1_d2y_b1_left[center], v1_d2y_b1_right[center], b1[center], d2y[center], c1_inds)
            beta_1 += -delta_t * 0.5 * right_left_C_W(v1_b_dotx_d1x_b1_left[center], v1_b_dotx_d1x_b1_right[center], b1[center], b_dotx_d1x[center], c1_inds)
            beta_1 += -delta_t * 0.5 * right_left_C_W(v1_b_doty_d1y_b1_left[center], v1_b_doty_d1y_b1_right[center], b1[center], b_doty_d1y[center], c1_inds)
            beta_1 += -delta_t * 0.5 * right_left_C_W(v1_d1x_b_dotx_b1_left[center], v1_d1x_b_dotx_b1_right[center], b1[center], d1x_b_dotx[center], c1_inds)
            beta_1 += -delta_t * 0.5 * right_left_C_W(v1_d1y_b_doty_b1_left[center], v1_d1y_b_doty_b1_right[center], b1[center], d1y_b_doty[center], c1_inds)
            
            beta_2 = right_left_C(v2_a2_left[center], v2_a2_right[center], a2[center], c2_inds)
            beta_2 += delta_t * nu * right_left_C_W(v2_d2x_b2_left[center], v2_d2x_b2_right[center], b2[center], d2x[center], c2_inds)
            beta_2 += delta_t * nu * right_left_C_W(v2_d2y_b2_left[center], v2_d2y_b2_right[center], b2[center], d2y[center], c2_inds)
            beta_2 += -delta_t * 0.5 * right_left_C_W(v2_b_dotx_d1x_b2_left[center], v2_b_dotx_d1x_b2_right[center], b2[center], b_dotx_d1x[center], c2_inds)
            beta_2 += -delta_t * 0.5 * right_left_C_W(v2_b_doty_d1y_b2_left[center], v2_b_doty_d1y_b2_right[center], b2[center], b_doty_d1y[center], c2_inds)
            beta_2 += -delta_t * 0.5 * right_left_C_W(v2_d1x_b_dotx_b2_left[center], v2_d1x_b_dotx_b2_right[center], b2[center], d1x_b_dotx[center], c2_inds)
            beta_2 += -delta_t * 0.5 * right_left_C_W(v2_d1y_b_doty_b2_left[center], v2_d1y_b_doty_b2_right[center], b2[center], d1y_b_doty[center], c2_inds)
            
            beta = vcat(vec(array(beta_1)), vec(array(beta_2)))
        end

        for center in R-1:-1:1

            function A_function_center_right(c_vec)
                c_1 = ITensor(c_vec[1:c1_length], c1_inds)
                c_2 = ITensor(c_vec[c1_length+1:c1_length+c2_length], c2_inds)
                out_1 = right_left_C_W(v1_dx_dx_v1_left[center+1], v1_dx_dx_v1_right[center+1], c_1, dx_dx[center+1], c1_inds)
                out_1 += right_left_C_W(v1_dx_dy_v2_left[center+1], v1_dx_dy_v2_right[center+1], c_2, dx_dy[center+1], c1_inds)
                out_2 = right_left_C_W(v2_dx_dy_v1_left[center+1], v2_dx_dy_v1_right[center+1], c_1, dx_dy[center+1], c2_inds)
                out_2 += right_left_C_W(v2_dy_dy_v2_left[center+1], v2_dy_dy_v2_right[center+1], c_2, dy_dy[center+1], c2_inds)
                return c_vec - penalty_coefficient * delta_t^2 * vcat(vec(array(out_1)), vec(array(out_2)))
            end
            A = FunctionMap{Float64,false}(A_function_center_right, c1_length + c2_length)

            cg!(c_vec, A, beta, reltol=1e-6, abstol=1e-12, verbose=false, maxiter=100, log=false)
            v1_copy, v2_copy = insert_c_vec(c_vec, v1_copy, v2_copy, center+1)

            orthogonalize!(v1_copy, center)
            orthogonalize!(v2_copy, center)
        
            c_vec = get_c_vec(v1_copy, v2_copy, center)
            c1_inds = inds(v1_copy[center])
            c2_inds = inds(v2_copy[center])
            c1_length = prod(size(v1_copy[center]))
            c2_length = prod(size(v2_copy[center]))

            v1_a1_right[center] = update_tn_mps_mps(v1_a1_right[center+1], v1_copy[center+1], a1[center+1])
            v1_d2x_b1_right[center] = update_tn_mps_mpo(v1_d2x_b1_right[center+1], v1_copy[center+1], b1[center+1], d2x[center+1])
            v1_d2y_b1_right[center] = update_tn_mps_mpo(v1_d2y_b1_right[center+1], v1_copy[center+1], b1[center+1], d2y[center+1])
            v1_b_dotx_d1x_b1_right[center] = update_tn_mps_mpo(v1_b_dotx_d1x_b1_right[center+1], v1_copy[center+1], b1[center+1], b_dotx_d1x[center+1])
            v1_b_doty_d1y_b1_right[center] = update_tn_mps_mpo(v1_b_doty_d1y_b1_right[center+1], v1_copy[center+1], b1[center+1], b_doty_d1y[center+1])
            v1_d1x_b_dotx_b1_right[center] = update_tn_mps_mpo(v1_d1x_b_dotx_b1_right[center+1], v1_copy[center+1], b1[center+1], d1x_b_dotx[center+1])
            v1_d1y_b_doty_b1_right[center] = update_tn_mps_mpo(v1_d1y_b_doty_b1_right[center+1], v1_copy[center+1], b1[center+1], d1y_b_doty[center+1])

            v2_a2_right[center] = update_tn_mps_mps(v2_a2_right[center+1], v2_copy[center+1], a2[center+1])
            v2_d2x_b2_right[center] = update_tn_mps_mpo(v2_d2x_b2_right[center+1], v2_copy[center+1], b2[center+1], d2x[center+1])
            v2_d2y_b2_right[center] = update_tn_mps_mpo(v2_d2y_b2_right[center+1], v2_copy[center+1], b2[center+1], d2y[center+1])
            v2_b_dotx_d1x_b2_right[center] = update_tn_mps_mpo(v2_b_dotx_d1x_b2_right[center+1], v2_copy[center+1], b2[center+1], b_dotx_d1x[center+1])
            v2_b_doty_d1y_b2_right[center] = update_tn_mps_mpo(v2_b_doty_d1y_b2_right[center+1], v2_copy[center+1], b2[center+1], b_doty_d1y[center+1])
            v2_d1x_b_dotx_b2_right[center] = update_tn_mps_mpo(v2_d1x_b_dotx_b2_right[center+1], v2_copy[center+1], b2[center+1], d1x_b_dotx[center+1])
            v2_d1y_b_doty_b2_right[center] = update_tn_mps_mpo(v2_d1y_b_doty_b2_right[center+1], v2_copy[center+1], b2[center+1], d1y_b_doty[center+1])

            v1_dx_dx_v1_right[center] = update_tn_mps_mpo(v1_dx_dx_v1_right[center+1], v1_copy[center+1], v1_copy[center+1], dx_dx[center+1])
            v1_dx_dy_v2_right[center] = update_tn_mps_mpo(v1_dx_dy_v2_right[center+1], v1_copy[center+1], v2_copy[center+1], dx_dy[center+1])
            v2_dx_dy_v1_right[center] = update_tn_mps_mpo(v2_dx_dy_v1_right[center+1], v2_copy[center+1], v1_copy[center+1], dx_dy[center+1])
            v2_dy_dy_v2_right[center] = update_tn_mps_mpo(v2_dy_dy_v2_right[center+1], v2_copy[center+1], v2_copy[center+1], dy_dy[center+1])
            
            beta_1 = right_left_C(v1_a1_left[center], v1_a1_right[center], a1[center], c1_inds)
            beta_1 += delta_t * nu * right_left_C_W(v1_d2x_b1_left[center], v1_d2x_b1_right[center], b1[center], d2x[center], c1_inds)
            beta_1 += delta_t * nu * right_left_C_W(v1_d2y_b1_left[center], v1_d2y_b1_right[center], b1[center], d2y[center], c1_inds)
            beta_1 += -delta_t * 0.5 * right_left_C_W(v1_b_dotx_d1x_b1_left[center], v1_b_dotx_d1x_b1_right[center], b1[center], b_dotx_d1x[center], c1_inds)
            beta_1 += -delta_t * 0.5 * right_left_C_W(v1_b_doty_d1y_b1_left[center], v1_b_doty_d1y_b1_right[center], b1[center], b_doty_d1y[center], c1_inds)
            beta_1 += -delta_t * 0.5 * right_left_C_W(v1_d1x_b_dotx_b1_left[center], v1_d1x_b_dotx_b1_right[center], b1[center], d1x_b_dotx[center], c1_inds)
            beta_1 += -delta_t * 0.5 * right_left_C_W(v1_d1y_b_doty_b1_left[center], v1_d1y_b_doty_b1_right[center], b1[center], d1y_b_doty[center], c1_inds)
            
            beta_2 = right_left_C(v2_a2_left[center], v2_a2_right[center], a2[center], c2_inds)
            beta_2 += delta_t * nu * right_left_C_W(v2_d2x_b2_left[center], v2_d2x_b2_right[center], b2[center], d2x[center], c2_inds)
            beta_2 += delta_t * nu * right_left_C_W(v2_d2y_b2_left[center], v2_d2y_b2_right[center], b2[center], d2y[center], c2_inds)
            beta_2 += -delta_t * 0.5 * right_left_C_W(v2_b_dotx_d1x_b2_left[center], v2_b_dotx_d1x_b2_right[center], b2[center], b_dotx_d1x[center], c2_inds)
            beta_2 += -delta_t * 0.5 * right_left_C_W(v2_b_doty_d1y_b2_left[center], v2_b_doty_d1y_b2_right[center], b2[center], b_doty_d1y[center], c2_inds)
            beta_2 += -delta_t * 0.5 * right_left_C_W(v2_d1x_b_dotx_b2_left[center], v2_d1x_b_dotx_b2_right[center], b2[center], d1x_b_dotx[center], c2_inds)
            beta_2 += -delta_t * 0.5 * right_left_C_W(v2_d1y_b_doty_b2_left[center], v2_d1y_b_doty_b2_right[center], b2[center], d1y_b_doty[center], c2_inds)

            beta = vcat(vec(array(beta_1)), vec(array(beta_2)))
        end
        E_0 = E_1
        E_1 = dot(c_vec, c_vec)
        if info
            println("Sweep no. $run in $(time()-t1) sek, norm_diff = $(abs((E_1 - E_0) / E_0))")
            t1 = time()
        end
        if run == 100
            println("Stopped optimization after $(run) sweeps")
            break
        end
    end

    if info
        println("Optimization in $(time()-t0) sek")
    end

    return v1_copy, v2_copy
end

function RK4_step(ux, uy, delta_t, nu, d1x, d1y, d2x, d2y, del, dx_dx, dy_dy, dx_dy, max_bond; info=false, eps=1e-5)

    if info
        t0 = time()
        t1 = time()
    end

    U1x, U1y = optimize_sweep(ux, uy, ux/4, uy/4, ux, uy, delta_t/6, nu, d1x, d1y, d2x, d2y, del, dx_dx, dy_dy, dx_dy, max_bond; eps=eps)
    
    if info
        println("U1x U1y in $(time()-t1) sek")
        t1 = time()
    end
    
    U2x, U2y = optimize_sweep(ux, uy, ux/4, uy/4, +(3*U1x, ux/4; maxdim=max_bond), +(3*U1y, uy/4; maxdim=max_bond), delta_t/3, nu, d1x, d1y, d2x, d2y, del, dx_dx, dy_dy, dx_dy, max_bond; eps=eps)
    
    if info
        println("U2x U2y in $(time()-t1) sek")
        t1 = time()
    end

    U3x, U3y = optimize_sweep(ux, uy, ux/4, uy/4, +(3/2*U2x, 5/8*ux; maxdim=max_bond), +(3/2*U2y, 5/8*uy; maxdim=max_bond), delta_t/3, nu, d1x, d1y, d2x, d2y, del, dx_dx, dy_dy, dx_dy, max_bond; eps=eps)
    
    if info
        println("U3x U3y in $(time()-t1) sek")
        t1 = time()
    end
    
    U4x, U4y = optimize_sweep(ux, uy, ux/4, uy/4, +(3*U3x, ux/4; maxdim=max_bond), +(3*U3y, uy/4; maxdim=max_bond), delta_t/6, nu, d1x, d1y, d2x, d2y, del, dx_dx, dy_dy, dx_dy, max_bond; eps=eps)
    
    if info
        println("U4x U4y in $(time()-t1) sek")
        println("Full Time step in $(time()-t0) sek")
    end

    return +(+(+(U1x, U2x; maxdim=max_bond), U3x; maxdim=max_bond), U4x; maxdim=max_bond), +(+(+(U1y, U2y; maxdim=max_bond), U3y; maxdim=max_bond), U4y; maxdim=max_bond)
end