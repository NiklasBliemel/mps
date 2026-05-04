using Plots
using ITensorMPS
using ITensors


function mps_to_array_2D(mps)
    R = length(mps)
    plot_tensor = array(prod(mps[:])) # contract all cores of the MPS into a single tensor

    # reshape tensor 2D (2^R x 2^R) Matrix
    plot_tensor = reshape(plot_tensor, ntuple(i -> 2, 2*R)...)
    perm = Vector{Int}(undef, 2*R)
    for i in 1:R
        perm[R + i] = 2*R - (2 * i - 1)
        perm[i] = 2*R - (2 * i - 2)
    end
    out = permutedims(plot_tensor, perm)
    return reshape(out, 2^R, 2^R)
end

function plot_mps(mps; grid_size=8, trunc=1e-5, save_name="", title="")
    plot_tensor = mps_to_array_2D(mps) # change velocity field from mps to 2D array format

    R = length(mps)
    grid_size = min(grid_size, R) # max grid size is 2^R, since the plot_tensor is 2^R x 2^R
    corse_grid = 1:2^(R - grid_size):2^R
    mps_vals = plot_tensor[corse_grid, corse_grid]
    mps_vals[abs.(mps_vals) .< trunc] .= 0. # set small values to zero for better visualization

    # create x and y values for contour plot
    xvals = collect(range(0, 1, 2^grid_size))
    yvals = collect(range(0, 1, 2^grid_size))
        
    if save_name == ""
        contour(xvals, yvals, mps_vals, title=title, fill=true)
    else
        p = contour(xvals, yvals, mps_vals, title=title, fill=true)
        savefig(p, save_name)
    end
end