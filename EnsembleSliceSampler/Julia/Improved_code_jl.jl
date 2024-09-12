#
# This file contains the implementation of the Ensemble Slice Sampler algorithm
# in Julia. 
#
using Distributions, Random, LinearAlgebra, StatsBase
using Combinatorics
using ProgressMeter

""" 
	tune_lengthscale(t::Int, μ::Float64, N_e::Int, N_c::Int, M_adapt::Int)

	Tune the lengthscale.

	# Arguments
	- `t::Int`: Iteration.
	- `μ::Float64`: Lengthscale.
	- `N_e::Int`: Number of evaluations.
	- `N_c::Int`: Number of constraints.
	- `M_adapt::Int`: Number of iterations to adapt the lengthscale.

	# Returns
	- `μ::Float64`: Updated lengthscale.
"""
function tune_lengthscale(t::Int64, μ::Float64, N_e::Int64, N_c::Int64, M_adapt::Int64)
	N_e = max(1, N_e)

	if t <= M_adapt
		return 2μ * N_e / (N_e + N_c)
	else
		return μ
	end
end

""" 
	DifferentialMove(μ::Float64, walker_l::AbstractVector{<:Float64}, walker_m::AbstractVector{<:Float64})

	Compute the differential move.

	# Arguments
	- `μ::Float64`: Lengthscale.
	- `walker_l::AbstractVector{<:Float64}`: Walker position.
	- `walker_m::AbstractVector{<:Float64}`: Walker position.

	# Returns
	- `direction::Array{Float64, 1}`: Direction vector.
"""
function DifferentialMove(μ::Float64, walker_l::AbstractVector{<:Float64}, walker_m::AbstractVector{<:Float64})
	return μ * (walker_l - walker_m)
end


""" 
	sample_ESS(rng::Random.AbstractRNG, logdens::Function, walkers::AbstractMatrix{<:Float64}, N_iter::Int, n_walkers::Int, ndim::Int, M_adapt::Int, max_steps::Int, μ)

	Ensemble Slice Sampler.

	# Arguments
	- `rng::Random.AbstractRNG`: Random number generator.
	- `logdens::Function`: Log density function.
	- `walkers::AbstractMatrix{<:Float64}`: Initial walker positions.
	- `N_iter::Int`: Number of iterations.
	- `n_walkers::Int`: Number of walkers.
	- `ndim::Int`: Dimension of the problem.
	- `M_adapt::Int`: Number of iterations to adapt the lengthscale.
	- `max_steps::Int`: Maximum number of steps.
	- `μ`: Lengthscale.

	# Returns
	- `S_save::Array{Float64, 3}`: Array of walker positions.
"""
function sample_ESS(rng::Random.AbstractRNG, logdens::Function, walkers::AbstractMatrix{<:Float64}, N_iter::Int64, n_walkers::Int64, ndim::Int64, M_adapt::Int64 = 100, max_steps::Int64 = 10^4, μ = 1.0)

	# walkers is of size

	# num of walkers divided by 2
	n_wdiv2 = div(n_walkers, 2)

	logdensities = [logdens(walkers[:, i]) for i in 1:n_walkers]

	# Randomly shuffle the walkers
	walker_indexes = randperm(rng, n_walkers)
	# get the first half of the shuffled walkers

	wl_ind = 1:n_wdiv2

	subset_a = @view walker_indexes[wl_ind]
	# get the second half of the shuffled walkers
	subset_b = @view walker_indexes[n_wdiv2+1:end]
	# get the two subsets
	sets = [[subset_a, subset_b], [subset_b, subset_a]]

	S_save = Array{Float64}(undef, ndim, n_walkers, N_iter)
	# initialise the log densities for the left and right stepping points

	@inbounds begin
		@showprogress for t in 1:N_iter
			R, L, N_e, N_c = 0, 0, 0, 0
			logdensities_left, logdensities_right = Vector{Float64}(undef, n_wdiv2), Vector{Float64}(undef, n_wdiv2)
			position_left, position_right = Matrix{Float64}(undef, ndim, n_wdiv2), Matrix{Float64}(undef, ndim, n_wdiv2)

			Widths = Vector{Float64}(undef, n_wdiv2)
			logdensities_shrink = Vector{Float64}(undef, n_wdiv2)
			positions_shrink = Matrix{Float64}(undef, ndim, n_wdiv2)
			# loop over the walkers
			for set in sets
				active, inactive = set

				### DifferentialMove ###
				# get all the permutations of the inactive walkers
				permuts = collect(permutations(inactive, 2))
				# get the number of permutations
				pairs = sample(rng, permuts, n_wdiv2, replace = false)
				# iterate over the pairs
				η = hcat([DifferentialMove(μ, view(walkers, :, p[1]), view(walkers, :, p[2])) for p in pairs]...) # (ndim, n_wdiv2)

				# masks the for the left and right stepping
				mask_left = fill(true, n_wdiv2)
				mask_right = fill(true, n_wdiv2)


				# get the move to the new position
				# draw y position 
				δ = rand(rng, Exponential(1), n_wdiv2)
				Y = logdensities[active] - δ

				# interval for the left stepping point
				L = -rand(rng, n_wdiv2)
				# interval for the right stepping point
				R = L .+ 1
				# initialise the counter
				l = 0

				J = floor.(Int, max_steps .* rand(rng, n_wdiv2))
				K = max_steps - 1 .- J

				# stepping out procedure
				while size(mask_left[mask_left], 1) > 0 || size(mask_right[mask_right], 1) > 0

					if size(mask_left[mask_left], 1) > 0
						l += 1
					end
					if size(mask_right[mask_right], 1) > 0
						l += 1
					end
					if l > max_steps
						error("Max steps reached in stepping out")
					end

					for j in wl_ind[mask_left]
						if J[j] < 1
							mask_left[j] = false
						end
					end
					for j in wl_ind[mask_right]
						if K[j] < 1
							mask_right[j] = false
						end
					end
					# println(size(position_left[:, mask_left])," L[mask_left]: ", size(L[mask_left]), " η[:, mask_left]: ", size(η[:, mask_left]), " walkers[:, active]: ", size(walkers[:, active][:, mask_left]))
					position_left[:, mask_left] = view(L, mask_left)' .* view(η, :, mask_left) + view(walkers, :, active)[:, mask_left]
					position_right[:, mask_right] = view(R, mask_right)' .* view(η, :, mask_right) + view(walkers, :, active)[:, mask_right]

					if size(view(position_left, :, mask_left), 1) + size(view(position_right, :, mask_right), 1) < 0
						logdensities_left[mask_left] = []
						logdensities_right[mask_right] = []
						l -= 1
					else
						nl = size(position_left[:, mask_left], 2)
						nr = size(position_right[:, mask_right], 2)
						logdensities_left[mask_left] = [logdens(view(view(position_left, :, mask_left), :, i)) for i in 1:nl]
						logdensities_right[mask_right] = [logdens(view(view(position_right, :, mask_right), :, i)) for i in 1:nr]
					end
					for j in wl_ind[mask_left]
						if Y[j] < logdensities_left[j]
							L[j] -= 1
							N_e += 1
							J[j] -= 1
						else
							mask_left[j] = false
						end
					end
					for j in wl_ind[mask_right]
						if Y[j] < logdensities_right[j]
							R[j] += 1
							N_e += 1
							K[j] -= 1
						else
							mask_right[j] = false
						end
					end
				end


				## shrink the interval##
				mask = fill(true, n_wdiv2)
				l = 0
				while size(mask[mask], 1) > 0

					Widths[mask] = rand(rng, Uniform(), size(mask[mask])) .* (view(R, mask) - view(L, mask)) .+ view(L, mask)

					positions_shrink[:, mask] = view(Widths, mask)' .* view(η, :, mask) + view(view(walkers, :, active), :, mask)
					logdensities_shrink[mask] = [logdens(view(view(positions_shrink, :, mask), :, i)) for i in 1:size(view(positions_shrink, :, mask), 2)]

					for j in wl_ind[mask]
						if Y[j] < logdensities_shrink[j]
							mask[j] = false
						else
							if Widths[j] < 0.0
								L[j] = Widths[j]
								N_c += 1
							else
								R[j] = Widths[j]
								N_c += 1
							end
						end
					end
					l += 1
					if l > max_steps
						error("Max steps reached in shrink")
					end
				end

				# update the walker
				walkers[:, active] = positions_shrink
				S_save[:, active, t] = positions_shrink
				logdensities[active] = logdensities_shrink
			end
			μ = tune_lengthscale(t, μ, N_e, N_c, M_adapt)

		end
	end
	return permutedims(S_save, [3, 1, 2])
end


### Application to the 8 schools problem
using Plots, Turing, StatsPlots

rng = MersenneTwister(1234)

n_schools = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0] # estimated treatment effects
σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

@model function school_reparam(y, σ, n_schools = 8)
	μ ~ Normal(0, 5)
	τ ~ truncated(Cauchy(0, 5), lower = 0)
	θ ~ filldist(Normal(0, 1), n_schools)
	for i in 1:n_schools
		y[i] ~ Normal(μ + τ * θ[i], σ[i])
	end
end

logd(X::AbstractVector{Float64}) = (logjoint(school_reparam(y, σ), (μ = X[1], τ = X[2], θ = X[3:10])))

sample_dict = rand(rng, OrderedDict, school_reparam(y, σ)).vals
flatten(w) = mapreduce(x -> x, vcat, w)
sample_dict = flatten(sample_dict);

n_dims = length(sample_dict)
n_walkers = 50
init_values = [flatten(rand(rng, OrderedDict, school_reparam(y, σ)).vals) for i in 1:n_walkers]
init_values = hcat(init_values...)

c = sample_ESS(rng, logd, S, 20_000, n_walkers, n_dims)

# using Profile
# @profview sample_ESS(rng, logd, S, 20_0, n_walkers, n_dims)

chn = Chains(c, ["μ", "τ", "θ[1]", "θ[2]", "θ[3]", "θ[4]", "θ[5]", "θ[6]", "θ[7]", "θ[8]"])

plot(chn[100:end, :, :])