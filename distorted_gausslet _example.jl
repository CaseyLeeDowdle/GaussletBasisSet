begin
    using Quiqbox
    using Plots
    using BenchmarkTools
    using LinearAlgebra
    using QuadGK
    #using FiniteDifferences
    using Tullio
    using LaTeXStrings
    #using Serialization
    using Measures
    #using LsqFit
    #using Optim
    using StaticArrays
    using Base.Threads
    using JLD2
    using Optim
    using Interpolations
    #using SparseArrays
    #using Arpack
end

# Global Variables

# G6 Gausslet coefficients (cite Steven White 2017 Gausslet paper)
const G6_coeff = [
    0.651079912213856, 0.374890195133727, 0.0939399437214329, -0.1569006465627569,
    -0.0948155527751206, 0.023264625608686, 0.0216613768304792, 0.0361805021062946,
    -0.0317148981502408, 0.0133915814065059, -0.0129995132051085, 0.0083444621145336,
    -0.0035602045266604, 0.0012544959501549, 0.0003594627655807, -0.0007848423809006,
    0.0005747148102592, -0.0003070053297971, 0.000149988658848, -0.0000895654658735,
    0.000052757953984, -0.000028480265623, 0.0000150015015272, -0.0000068808321161,
    0.0000028004555091, -0.0000013067346743, 0.0000007168554123, -0.0000003912054599,
    0.0000001613546686, -0.0000000401650166, 0.0000000012233441, 0.0000000063863555,
    -0.0000000030684215, 0.0000000004500457, -0.0000000002218040, 0.0000000000913882,
    0.0000000000104569, -0.0000000000043992
]

const G6_coeff_all = vcat(reverse(G6_coeff[2:end]),G6_coeff)


function inv_sinh_map(x,c,s,x0)
    return (1.0/s) * asinh((x-x0)/c)
end

struct Inv_sinh_map{T<:AbstractFloat} <: Function
    c::T
    s::T
    x0::T
end

(f::Inv_sinh_map{T})(x::T) where {T} = inv_sinh_map(x, f.c, f.s, f.x0)

function d_inv_sinh_map(x,c,s,x0)
    return ((1.0/s)*(1.0/sqrt((x-x0)^2 +c^2))) 
end

struct D_inv_sinh_map{T<:AbstractFloat} <: Function
    c::T
    s::T
    x0::T
end

(f::D_inv_sinh_map{T})(x::T) where {T} = d_inv_sinh_map(x, f.c, f.s, f.x0)

function sinh_map(u,c,s,x0)
    return c*sinh(s*u) + x0
end

struct Sinh_map{T<:AbstractFloat} <: Function
    c::T
    s::T
    x0::T
end

(f::Sinh_map{T})(x::T) where {T} = sinh_map(x, f.c, f.s, f.x0)


# user function for Gausslet centered at 'n' with 
# coefficients 'c' (typically use G6_coeff_all)
function make_G(c::AbstractVector{<:Real}, n::Integer)
    @assert length(c) == 75 "length of G6_coeff_all"
    ivals    = -37:37
    centers  = n .+ ivals ./ 3          
    coeffs   = c                        
    return function (x)
        
        Δ   = @. x - centers            
        val = sum(coeffs .* @. exp(-4.5 * Δ^2))
        return val
    end
end


# function for mean zero contracted gaussian
function make_C(ci::AbstractVector{<:Real}, xi_ti::AbstractVector{<:Real})
    
    return function (x)
        
        val = sum(ci .* @. exp(-xi_ti .* x^2))
        return val
    end
end

function multiply(f::F, g::G) where {F<:Function, G<:Function}
    return x -> f(x) * g(x)
end

# Coordinate transformed gausslets (distorted gausslets)
# Using the inverse sinh mapping

function build_maps(c, s, x0)
    inv  = ntuple(i -> Inv_sinh_map(c[i], s[i], x0[i]),  length(c))
    dinv = ntuple(i -> D_inv_sinh_map(c[i], s[i], x0[i]), length(c))

    u(x)  = @inbounds sum(f(x) for f in inv)
    du(x) = @inbounds sum(f(x) for f in dinv)
    return u, du
end

function build_map_sinh(c, s, x0)
    sinh_maps  = ntuple(i -> Sinh_map(c[i], s[i], x0[i]),  length(c))
    x(u)  = @inbounds sum(f(u) for f in sinh_maps)
    return x
end

function build_map_d_inv_sinh(c, s, x0)
    d_inv_sinh_maps  = ntuple(i -> D_inv_sinh_map(c[i], s[i], x0[i]),  length(c))
    du(x)  = @inbounds sum(f(x) for f in d_inv_sinh_maps)
    return du
end

function gausslet_sinh(x,k,c,s,x0)
    u, d_u = build_maps(c,s,x0)   
    gausslet = make_G(G6_coeff_all,k)
    gausslet_distorted = multiply(sqrt∘d_u,gausslet∘u)
    
    return gausslet_distorted(x)
end

struct Gausslet_sinh{N,T<:AbstractFloat} <: Function
    k::Int
    c::NTuple{N,T}
    s::NTuple{N,T}
    x0::NTuple{N,T}
end

(f::Gausslet_sinh{N,T})(x::T) where {N,T<:AbstractFloat} = gausslet_sinh(x, f.k,f.c, f.s, f.x0)

@inline rho_single(x, c::T, s::T, X::T) where {T<:Real} = inv(s * ((x-X)^2 + c^2))

# Build ρ(x) for N nuclei.   Args:  a, s, X same length (one per nucleus)
# Optionally supply d (max spacing); d = Inf ⇒ omit 1/d term.
function build_rho(c,s,x0,d)
    N = length(c)
    @assert length(s) == N && length(x0) == N
    invd = isfinite(d) ? inv(d) : zero(eltype(c))

    return function ρ(x)
        ρ2 = zero(eltype(c))
        @inbounds for i in 1:N
            r = rho_single(x, c[i], s[i], x0[i])
            ρ2 += r*r
        end
        return sqrt(ρ2) + invd
    end
end

function build_u(ρ::Function; rtol=1e-12, atol=1e-12)
    # Cache the cumulative integral so repeated calls are cheap.
    
     memo_quad = x -> first(quadgk(ρ, 0.0, x; rtol, atol))

    return function u(x)
        if x isa AbstractArray
            return memo_quad.(x)  # element‑wise
        else
            return memo_quad(x)
        end
    end
end



function build_u_inverse(c, s, x0,d;
                         xspan::Tuple = (-100.0, 100.0), n::Int = 10_001,
                         kind::Symbol = :linear)
    xs = range(xspan[1], xspan[2], length=n)
    rho_map = build_rho(c,s,x0,d)
    u_map = build_u(rho_map)
    us = u_map.(xs)

    @assert all(diff(us) .> 0) "u_map must be strictly increasing over xspan"

    itp = begin
        if kind == :linear
            LinearInterpolation(us, xs, extrapolation_bc = Interpolations.Line())
        elseif kind == :quadratic
            QuadraticInterpolation(us, xs, extrapolation_bc = Interpolations.Line())
        else
            error("kind must be :linear or :quadratic")
        end
    end
    return (u_val -> itp(u_val))
end

function gausslet_sinh_mod(x,k,c,s,x0,d)
    d_u = build_rho(c,s,x0,d)
    u = build_u(d_u)
    gausslet = make_G(G6_coeff_all,k)
    gausslet_distorted = multiply(sqrt∘d_u,gausslet∘u)
    return gausslet_distorted(x)
end

struct Gausslet_sinh_mod{N,T<:AbstractFloat} <: Function
    k::Int
    c::NTuple{N,T}
    s::NTuple{N,T}
    x0::NTuple{N,T}
    d::T
end

(f::Gausslet_sinh_mod{N,T})(x::T) where {N,T<:AbstractFloat} = gausslet_sinh_mod(x, f.k,f.c, f.s, f.x0, f.d)

# ---------------------------------------------------------------------
# objective: Φ(c) =  Σ_i ( du(x0[i]) - sc )²
# ---------------------------------------------------------------------
function objective(c_vec::Vector{T},
                   s::NTuple{N,T},
                   x0::NTuple{N,T},
                   d::T,
                   sc::T) where {N,T}

    c = ntuple(i -> c_vec[i], N)                 # Vector → NTuple
    du = build_rho(c,s,x0,d)

    err2 = zero(T)
    @inbounds for i in 1:N

        δ = du(x0[i]) - (1.0/sc)
        err2 += δ^2
    end
    return  err2
end


# ---------------------------------------------------------------------
# driver: optimise c so that spacing ≈ sc
# ---------------------------------------------------------------------
function optimise_c(initial_c::NTuple{N,T},
                    s::NTuple{N,T},
                    x0::NTuple{N,T},
                    d::T,
                    sc::T; maxiters::Int = 200) where {N,T}

    c0_vec = collect(initial_c)                     # Optim works with Vector
    obj(v) = objective(v, s, x0, d,sc)
    opts = Optim.Options(iterations = maxiters) 
    result = optimize(obj, c0_vec, NelderMead(),opts )
    c_opt_vec  = result.minimizer
    c_opt     = ntuple(i -> c_opt_vec[i], N) 
    return c_opt, result
end


# Following gaussian quadrature coulomb operator approximation
# In appendix of "Nested Gausslet" paper
function coulomb_approx(;c=0.03,s=0.3,M=45,delta=1.0)
    inds = collect(1:M)
    ui = inds .- 1/2
    t_map = Sinh_map(c,s,0.0)
    ti = t_map.(ui)
    rho_map = D_inv_sinh_map(c,s,0.0)
    rho_ti = rho_map.(ti)
    xi_ti = ti .^2
    ci = (2.0*delta) ./ (sqrt(pi).*rho_ti)
    # integral_array = ci .* exp.(-xi_ti.*r^2)
    return ci, xi_ti
end


# functions for least squares fit of gaussian grid to distorted gausslets
function gen_grid(N)
    # k is center location of grid
    points = Float64.(collect(-N:N))
    return points
end

function gen_distorted_gausslet_coeff(c,s,x0,N)
    x_map = build_map_sinh(c,s,x0)
    density_map = build_map_d_inv_sinh(c,s,x0)
    ui= gen_grid(N)
    xi = x_map.(ui)
    rhoi = density_map.(xi)
    alphai =  (rhoi .^2)/(2.0*1.25^2)
    return alphai,xi
end

function gen_distorted_gausslet_coeff_mod(c,s,x0,d,N)
    x_map = build_u_inverse(c,s,x0,d)
    density_map = build_rho(c,s,x0,d)
    ui= gen_grid(N)
    xi = x_map.(ui)
    rhoi = density_map.(xi)
    alphai =  (rhoi .^2)/(1.25^2)
    return alphai,xi
end

function build_feature(x,α,μ)
    M = length(x)
    N = length(α)
    X = zeros(M,N)
    for i in 1:M
        for j in 1:N
            X[i,j] = exp(-α[j]*(x[i]-μ[j])^2)
        end
    end
    return X
end


function gaussian_fit(a,b,Npts,target_fun,c,s,x0,Ngrid)
    ui  = range(a, b; length=Npts)
    x_map = build_map_sinh(c,s,x0)
    xi = x_map.(ui)
    α, μ = gen_distorted_gausslet_coeff(c,s,x0,Ngrid)
    X = build_feature(xi,α,μ)
    yvec = target_fun.(xi)
    model_param = X \ yvec 
    return model_param
end

function gaussian_fit_mod(a,b,Npts,target_fun,c,s,x0,d,Ngrid)
    ui  = range(a, b; length=Npts)
    x_map = build_u_inverse(c,s,x0,d)
    xi = x_map.(ui)
    α, μ = gen_distorted_gausslet_coeff_mod(c,s,x0,d,Ngrid)
    X = build_feature(xi,α,μ)
    yvec = target_fun.(xi)
    model_param = X \ yvec 
    return model_param
end

function gaussian_fit_orb(coeffs,α, μ)
    gfs = GaussFunc.(α)
    xi_centers = [(point, ) for point in μ]
    pgs = PrimitiveOrb.(xi_centers,gfs)
    contracted_pgs = CompositeOrb(pgs, coeffs)
    return contracted_pgs
end

function gaussletInt(coeffs, exp_coeffs)
    return  sum(coeffs .* sqrt.(pi ./ exp_coeffs))
end



# ----- Custom 2 Body integrals for gaussian fit of distorted gausslets


coulomb_coeff, coulomb_exp = coulomb_approx()
#coulomb_coeff, coulomb_exp = coulomb_approx(M=115,s=0.16,a=0.01)

c_init = (0.5,0.5)
s_fixed  = (0.5, 0.5)
x0_nuc = (-0.7, 0.7)
sc_target = 0.5*0.5; 
d_fixed = 3.0
c_opt, res = optimise_c(c_init, s_fixed, x0_nuc, d_fixed,sc_target;
                        maxiters = 200)

x_coords = collect(-8:8)

g_targets_x = map(i -> Gausslet_sinh_mod(i,c_opt,s_fixed,x0_nuc,d_fixed),x_coords)


x_in = LinRange(-10,10,1000)
p = plot()
for g in g_targets_x
    plot!(p,x_in,g.(x_in))
end
display(p)


#quadgk(x -> multiply(g1_target,g2_target)(x), -Inf, Inf)
a,b = -200.0,200.0
Npts = 10000


s_gauss = (0.1,0.1)
c_gauss = (0.45,0.45)
Ngrid = 100

α_x, μ_x = gen_distorted_gausslet_coeff_mod(c_gauss,s_gauss,x0_nuc,d_fixed,Ngrid)

gc_x = map(g_target -> gaussian_fit_mod(a,b,Npts,g_target,c_gauss,s_gauss,x0_nuc,d_fixed,Ngrid), g_targets_x)

g_fit_x = map(gc -> gaussian_fit_orb(gc,α_x,μ_x),gc_x)

x_in = LinRange(-10,10,4000)
plot(x_in,g_targets_x[6].(x_in))
plot!(x_in,g_fit_x[6].(x_in))

I1,_ = quadgk(x -> g_targets_x[6](x), -Inf, +Inf)
I2 = gaussletInt(gc_x[6],α_x)
abs(I1 - I2)

for i in eachindex(g_fit_x)
    println("ind $i","overlap:",overlap(g_fit_x[i],g_fit_x[i]))
end

#S = overlaps(g_fit)
eKinetic(g_fit_x[1],g_fit_x[1])
y_coords = collect(-6:6)

c_y = (0.5,)
s_y = (0.5,)
y0 = (0.0,)

g_targets_y = map(i -> Gausslet_sinh_mod(i,c_y,s_y,y0,d_fixed),y_coords)

a,b = -200.0,200.0
Npts = 10000
s_gauss_y = (0.1,)
c_gauss_y = (0.3,)
Ngrid = 100

α_y, μ_y = gen_distorted_gausslet_coeff_mod(c_gauss_y,s_gauss_y,y0,d_fixed,Ngrid)

gc_y = map(g_target -> gaussian_fit_mod(a,b,Npts,g_target,c_gauss_y,s_gauss_y,y0,d_fixed,Ngrid), g_targets_y)

g_fit_y = map(gc -> gaussian_fit_orb(gc,α_y,μ_y),gc_y)

x_in = LinRange(-10,10,4000)
plot(x_in,g_targets_y[1].(x_in))
plot!(x_in,g_fit_y[1].(x_in))

for i in eachindex(g_fit_y)
    println("ind $i","overlap:",overlap(g_fit_y[i],g_fit_y[i]))
end
I1,_ = quadgk(x -> g_targets_y[11](x), -Inf, +Inf)
I2 = gaussletInt(gc_y[11],α_y)
abs(I1 - I2)

Ex = eKinetics(g_fit_x)
Ey = eKinetics(g_fit_y)
Nx = length(g_fit_x)
Ny = length(g_fit_y)
Ndim = Nx*Ny^2
E_core = kron(Ex,I(Ny), I(Ny)) + kron(I(Nx),Ey, I(Ny)) + kron(I(Nx),I(Ny),Ey)

@inline function nucInt(ai::Float64,aj::Float64,ar::Float64,xi::Float64,xj::Float64,xr::Float64)
    # reduced number of operations by canceling constant factors
    A = ai+aj+ar
    B = ai*xi + aj*xj + ar*xr
    C = ai*xi^2 + aj*xj^2 +ar*xr^2
    return sqrt(pi/A)*exp((B^2/A)-C)

end

@inline function nucSum(c1,c2,
    exp_coeffs_grid::Vector{Float64},
    means_grid::Vector{Float64},
    exp_coulomb,mean_coulomb)
    S = 0.0
    @inbounds for i in eachindex(c1)
        ci = c1[i]; ai = exp_coeffs_grid[i]; xi = means_grid[i]
        for j in eachindex(c2)
            cj = c2[j]; aj = exp_coeffs_grid[j]; xj = means_grid[j]
            cij = ci * cj
            S += cij*nucInt(ai,aj,exp_coulomb,xi,xj,mean_coulomb)
        end
    end
    return S
end

function build_nuc_mat(coeffs_all,exp_coeffs_grid, means_grid, exp_coulomb,mean_coulomb)
    nb  = size(coeffs_all, 2)
    nuc = Array{Float64,2}(undef, nb, nb)
    cols = collect(eachcol(coeffs_all))
    for p in 1:nb
        c_p = cols[p]
        for q in p:nb
            c_q = cols[q]
            val = nucSum(c_p,c_q,
                        exp_coeffs_grid,means_grid,
                        exp_coulomb,mean_coulomb)
            nuc[p,q] = nuc[q,p] = val
        end
    end
    return nuc
end

function build_nuc_1d(coeffs_all,exp_coeffs_grid, means_grid,
                   exp_coeffs_coulomb,mean_coulomb)
        V_m = [build_nuc_mat(coeffs_all,exp_coeffs_grid,means_grid,ar,mean_coulomb) for ar in exp_coeffs_coulomb] 
    return V_m
end

function build_nuc(coeffs_all,exp_coeffs_grid, means_grid,
                   coeffs_coulomb, exp_coeffs_coulomb,coords_coulomb)
        
        Nx = size(coeffs_all[1], 2)
        Ny = size(coeffs_all[2], 2)
        Nz = size(coeffs_all[3], 2)
        Ndim = Nx*Ny*Nz
        V_x = build_nuc_1d(coeffs_all[1],exp_coeffs_grid[1],means_grid[1],exp_coeffs_coulomb,coords_coulomb[1]) 
        V_y = build_nuc_1d(coeffs_all[2],exp_coeffs_grid[2],means_grid[2],exp_coeffs_coulomb,coords_coulomb[2]) 
        V_z = build_nuc_1d(coeffs_all[3],exp_coeffs_grid[3],means_grid[3],exp_coeffs_coulomb,coords_coulomb[3]) 
                            
        V = zeros(Float64, Ndim, Ndim)          
        for (d, V_xi,V_yi,V_zi) in zip(coeffs_coulomb, V_x,V_y,V_z)
            V .+= d * kron(V_xi, V_yi, V_zi)                  
        end
    return V 
end

nuc_coords1 = (-0.7,0.0,0.0)
nuc_coords2 = (0.7,0.0,0.0)
coeffs_all = (hcat(gc_x...),hcat(gc_y...),hcat(gc_y...))
α = [α_x,α_y,α_y]
μ = [μ_x,μ_y,μ_y]
V_n1 = -1.0 .* build_nuc(coeffs_all,α,μ,coulomb_coeff,coulomb_exp,nuc_coords1)
V_n2 = -1.0 .* build_nuc(coeffs_all,α,μ,coulomb_coeff,coulomb_exp,nuc_coords2)
V_at = V_n1 .+ V_n2
heatmap(-V_at,dpi=1200)
heatmap(E_core)

@inline function twoBodyIntDiag(ai,aj,ar,xi,xj)

    # ----- build 2×2 matrix A, its determinant and inverse -------------

    A11 = ai + ar
    A12 = -ar
    A21 = -ar
    A22 = aj + ar

    detA   = A11*A22 - A12*A21           # scalar
    invdet = 1/detA                      # reuse twice

    invA11 =  A22 * invdet
    invA12 = -A12 * invdet
    invA21 = -A21 * invdet
    invA22 =  A11 * invdet

    # ----- build B and quadratic form  Bᵀ A⁻¹ B -----------------------
    B1 = ai*xi 
    B2 = aj*xj 

    q  = (B1*(invA11*B1 + invA12*B2) +
              B2*(invA21*B1 + invA22*B2))

    # ----- constant C ---------------------------------------------------
    C  = -(ai*xi^2 + aj*xj^2)

    return (π)*sqrt(invdet) * exp(q + C)   
end


function twoBodySumDiag(c1,c2,
    exp_coeffs_grid::Vector{Float64},
    means_grid::Vector{Float64},
    exp_coulomb)

    S = 0.0
    w1 = gaussletInt(c1,exp_coeffs_grid)
    w2 = gaussletInt(c2, exp_coeffs_grid)
    
    @inbounds for i in eachindex(c1)
        ci = c1[i]; ai = exp_coeffs_grid[i]; xi = means_grid[i]
        for j in eachindex(c2)
            cj = c2[j]; aj = exp_coeffs_grid[j]; xj = means_grid[j]
            cij = ci * cj 
            S += cij*twoBodyIntDiag(ai,aj,exp_coulomb,xi,xj)
        end
    end
    return S/(w1*w2)
    
end

function build_eri_twoBodySumDiag(coeffs_all, exp_coeffs_grid, means_grid,exp_coulomb)
    nb  = size(coeffs_all, 2)
    eri_sparse = Array{Float64,2}(undef, nb, nb)

    cols = collect(eachcol(coeffs_all))   

    @inbounds for p in 1:nb
        c_p = cols[p]
        for q in p:nb
            c_q = cols[q]
            val = twoBodySumDiag(c_p, c_q,
                                    exp_coeffs_grid, means_grid,
                                    exp_coulomb)

            eri_sparse[p,q] = eri_sparse[q,p] = val
        end
    end
    return eri_sparse
end

function build_eri_diag_2d(coeffs_all,exp_coeffs_grid, means_grid,
     exp_coeffs_coulomb)
    Vee_m = [build_eri_twoBodySumDiag(coeffs_all,exp_coeffs_grid,means_grid,ar) for ar in exp_coeffs_coulomb] 
    return Vee_m
end

function build_eri_diag(coeffs_all,exp_coeffs_grid, means_grid,
    coeffs_coulomb, exp_coeffs_coulomb)
    
    Nx = size(coeffs_all[1], 2)
    Ny = size(coeffs_all[2], 2)
    Nz = size(coeffs_all[3], 2)
    Ndim = Nx*Ny*Nz

    Vee_x = build_eri_diag_2d(coeffs_all[1],exp_coeffs_grid[1],means_grid[1],exp_coeffs_coulomb) 
    Vee_y = build_eri_diag_2d(coeffs_all[2],exp_coeffs_grid[2],means_grid[2],exp_coeffs_coulomb) 
    Vee_z = build_eri_diag_2d(coeffs_all[3],exp_coeffs_grid[3],means_grid[3],exp_coeffs_coulomb) 

    Vee = zeros(Float64, Ndim, Ndim)          
    for (d, Vee_xi,Vee_yi,Vee_zi) in zip(coeffs_coulomb, Vee_x,Vee_y,Vee_z)
        Vee .+= d * kron(Vee_xi, Vee_yi, Vee_zi)                  
    end

    return Vee
end


ERI_int_diag = build_eri_diag(coeffs_all,α,μ,coulomb_coeff,coulomb_exp)



heatmap(ERI_int_diag)


@save "array_H2.jld2" V_at E_core ERI_int_diag

function rhf_diagERI(H1::AbstractMatrix{T},
                     V ::AbstractMatrix{T},
                     Ns::Integer;
                     tol::Real = 1e-10,
                     maxiter::Int = 200,
                     mix::Real = 0.30) where {T<:Real}

    N = size(H1,1)
    @assert size(H1) == (N,N)  "H1 must be N×N"
    @assert size(V)  == (N,N)  "V  must be N×N"
    @assert 0 < Ns ≤ N

    D      = zeros(T, N, N)           # density matrix  (includes factor 2)
    E_last = typemax(T)

    for iter = 1:maxiter
        ###################################################################
        # Build Fock matrix in the diagonal‑ERI approximation
        ###################################################################
        occ_diag   = diag(D)                  # vector D_kk
        J_diag_vec = V * occ_diag             # J_i  = Σ_k D_kk (ii|kk)

        # Fock: F_ij = H1_ij + δ_ij * J_diag_vec[i]  - 0.5 * D_ij * V_ij
        F = copy(H1)
        @inbounds for i=1:N
            F[i,i] += J_diag_vec[i]           # Coulomb contributes only on diag
        end
        F .-= 0.5 .* (D .* V)                 # exchange term (all elements)

        ###################################################################
        # Solve Roothaan equations  (S = I  ⇒  plain eigen‐problem)
        ###################################################################
        eig = eigen(Symmetric(F))
        C   = eig.vectors                     # MO coefficients
        ε   = eig.values

        # new density from occupied MOs, then linear mixing
        Cocc = @view C[:, 1:Ns]
        D_new = 2.0 .* (Cocc * Cocc')
        D_mix = (1 - mix) .* D .+ mix .* D_new

        # total RHF energy   E = ½ Σ_ij D_ij (H1_ij + F_ij)
        E = 0.5 * sum(D_mix .* (H1 .+ F))

        # convergence: RMS of density change
        rmsΔ = norm(D_mix - D) / N
        if rmsΔ < tol
            return (E, C, ε, iter)
        end

        D = D_mix
        E_last = E
        println("it: $iter  E: $E")
    end

    error("SCF failed to converge in $maxiter iterations (last RMS ΔP = $(norm(D_new-D)/N))")
end

@inline function build_fock_diagERI!(F::AbstractMatrix,        # out
                                     H::AbstractMatrix,
                                     V::AbstractMatrix,
                                     D::AbstractMatrix,
                                     Jtmp::AbstractVector)     # scratch N‑vector
    # J_i = Σ_k D_kk (ii|kk)
    copyto!(Jtmp, V * diag(D))

    @inbounds for i in axes(F,1), j in axes(F,2)
        Fij = H[i,j] - 0.5 * D[i,j] * V[i,j]            # exchange part
        if i == j
            Fij += Jtmp[i]                              # Coulomb only on diag
        end
        F[i,j] = Fij
    end
    return nothing
end

function rhf_diagERI(H1_ao::AbstractMatrix{T},
                     V_ao ::AbstractMatrix{T},
                     X    ::AbstractMatrix{T},
                     Ns   ::Integer;
                     tol::Real     = 1e-10,
                     maxiter::Int  = 200,
                     mix::Real     = 0.30) where {T<:Real}

    N = size(H1_ao,1)
    @assert size(H1_ao)==(N,N) && size(V_ao)==(N,N) && size(X)==(N,N)
    @assert 0 < Ns ≤ N

    # 1.  AO → orthonormal basis (‘prime’)
    H1 = X' * H1_ao * X
    Xsq = X .^ 2                             # element‑wise, N×N
    V  = Xsq' * V_ao * Xsq                  # V'_{ij} = Σ_μλ X²_{μi} (μμ‖λλ) X²_{λj}

    F   = similar(H1)                        # work array
    Jv  = zeros(T, N)                        # scratch Coulomb vector
    D   = zeros(T, N, N)                     # density (includes factor 2)

    for iter = 1:maxiter
        # ---------- build Fock from current density -----------------------
        build_fock_diagERI!(F, H1, V, D, Jv)

        # ---------- solve Roothaan‑Hall  (S=I) ----------------------------
        eig = eigen(Symmetric(F))
        Cpr = eig.vectors
        eps = eig.values

        Cocc = @view Cpr[:, 1:Ns]
        Dnew = 2 .* (Cocc * Cocc')           # closed‑shell density

        # linear mixing
        Dmix = (1 - mix) .* D .+ mix .* Dnew
        rmsΔ = norm(Dmix - D) / N

        # energy with *mixed* density (re‑use build_fock once more)
        build_fock_diagERI!(F, H1, V, Dmix, Jv)
        E = 0.5 * sum(Dmix .* (H1 .+ F))

        if rmsΔ < tol
            C_ao = X * Cpr                   # back‑transform coeffs
            return (E, C_ao, eps, iter)
        end
        D .= Dmix                           # next iteration
        println("it: $iter  E: $E")
    end
    error("SCF failed to converge in $maxiter iterations")
end

#quadgk(x-> g_targets_x[1](x)*g_targets_x[1](x),-Inf,Inf)

Sx = overlaps(g_fit_x)
Sy = overlaps(g_fit_y)
S = Symmetric(kron(Sx,Sy,Sy))
X = S^(-0.5)

Ns = 1                             # one electron pair (2 e⁻ total)

H_nuc1 = E_core + V_n1
F = eigen(Symmetric(H_nuc1))
v0 = F.vectors[:,1]
e0 = F.values[1]

H1 = E_core + V_at

E, C, eps, niter = rhf_diagERI(H1, ERI_int_diag,X, Ns; tol=1e-10, maxiter = 100,mix = 0.5);

using HCubature

@inline secpi(z) = inv(cospi(z)) 

function integrate_R2(f; rtol = 1e-10, atol = 1e-10)
    integrand_uv(uv) = begin
        u, v = uv
        x = tanpi(u - 0.5)
        y = tanpi(v - 0.5)
        J = (π^2) * secpi(u - 0.5)^2 * secpi(v - 0.5)^2
        f(x, y) * J
    end
    hcubature(integrand_uv, [0.0, 0.0], [1.0, 1.0];
              rtol = rtol, atol = atol)     # ← updated keyword names
end

function integrate_R2_rational(f; rtol=1e-10, atol=1e-10)

    # 1‑D pieces --------------------------------------------------------------
    mapcoord(t)  = (2t - 1) / (t * (1 - t))         # ℝ
    jac1d(t)     = (2t^2 - 2t + 1) / (t * (1 - t))^2  # |dx/du|

    # 2‑D integrand on the unit square ---------------------------------------
    integrand_uv(uv) = begin
        u, v = uv
        x, y = mapcoord(u), mapcoord(v)
        J    = jac1d(u) * jac1d(v)
        f(x, y) * J                    # works for scalar or vector f
    end

    hcubature(integrand_uv, [0.0, 0.0], [1.0, 1.0];
              rtol = rtol, atol = atol)
end
f_test(x,y) = exp(-(x^2+y^2))

@time integrate_R2(f_test)
@time integrate_R2_rational(f_test)

g1 = Gausslet_sinh(0,(0.5,),(0.5,),(0.0,))
g2 = Gausslet_sinh(5,(0.5,),(0.5,),(0.0,))


g_test(x,y) = g1(x)*exp(-1.0*(x-y)^2)*g2(y)
g_test2(x,y) = g1(x)*exp(-10.0^2*(x-y)^2)*g2(y)
g_test3(x,y) = g_targets_x[1](x)*exp(-0.01*(x-y)^2)*g_targets_x[3](y)

@benchmark integrate_R2(f_test)
@benchmark integrate_R2_rational(f_test)

@benchmark integrate_R2(g_test)
@benchmark integrate_R2_rational(g_test)

@benchmark integrate_R2(g_test3)
@benchmark integrate_R2_rational(g_test3)