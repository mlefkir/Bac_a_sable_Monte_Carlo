using Distributions
using Random: Random
using AdvancedMH
using AbstractMCMC
using LogDensityProblems
using AdvancedMH: logdensity
using Turing
using Turing.Inference: SampleFromPrior, AbstractVarInfo, Sampler, Model, setindex!!, _params_to_array, VarInfo, getparams, InferenceAlgorithm, get_transition_extras, getlogevidence, getlogp, getspace

using Combinatorics, StatsBase

struct ZeusProposal{P, F <: AbstractFloat} <: AdvancedMH.Proposal{P}
	proposal::P
	μ::F
end

struct Transition{T, L <: Real} <: AdvancedMH.AbstractTransition
	params::T
	lp::L
	accepted::Bool
	μ::L
	t::Int
end

struct EnsembleSliceSampler{T <: Float64, A <: Int64} <: AdvancedMH.MHSampler
	"initial length scale"
	μ_init::T
	"number of adapation steps"
	M_adapt::A
	"number of walkers"
	n_walkers::A
	"max number of attempts"
	max_steps::A

	function EnsembleSliceSampler(μ_init::T, M_adapt::A, n_walkers::A, max_steps::A) where {T, A}
		if n_walkers % 2 != 0
			throw(ArgumentError("The number of walkers must be even."))
		elseif max_steps < 100
			throw(ArgumentError("The maximum number of steps must be at least 100."))
		end

		new{T, A}(μ_init, M_adapt, n_walkers, max_steps)
	end
end

struct EnSS{space, E <: EnsembleSliceSampler} <: InferenceAlgorithm
	ensemble::E
end

function EnSS(n_walkers::Int, μ_init = 1.0, M_adapt = 50, max_steps = 10000)
	ensemble = EnsembleSliceSampler(μ_init, M_adapt, n_walkers, max_steps)
	return EnSS{(), typeof(ensemble)}(ensemble)
end
Turing.Inference.getspace(::EnSS) = ()


struct EnSSState{V <: AbstractVarInfo, S}
	vi::V
	states::S
end

function AbstractMCMC.step(
	rng::Random.AbstractRNG,
	model::Model,
	spl::Sampler{<:EnSS};
	resume_from = nothing,
	initial_params = nothing,
	kwargs...)
	if resume_from !== nothing
		state = loadstate(resume_from)
		return AbstractMCMC.step(rng, model, spl, state; kwargs...)
	end

	# Sample from the prior
	n = spl.alg.ensemble.n_walkers
	vis = [VarInfo(rng, model, SampleFromPrior()) for _ in 1:n]

	# Update the parameters if provided.
	if initial_params !== nothing
		length(initial_params) == n ||
			throw(ArgumentError("initial parameters have to be specified for each walker"))
		vis = map(vis, initial_params) do vi, init
			vi = DynamicPPL.initialize_parameters!!(vi, init, spl, model)

			# Update log joint probability.
			last(DynamicPPL.evaluate!!(model, rng, vi, SampleFromPrior()))
		end
	end

	# Compute initial transition and states.
	# transition = map(Base.Fix1(Turing.Inference.Transition, model), vis)
	transition = [Turing.Inference.Transition(getparams(model, vi), getlogp(vi), (accepted = true, length_scale = spl.alg.ensemble.μ_init, t = 1)) for vi in vis]
	# TODO: Make compatible with immutable `AbstractVarInfo`.
	state = EnSSState(
		vis[1],
		map(vis) do vi
			vi = DynamicPPL.link!!(vi, spl, model)
			Transition(vi[spl], getlogp(vi), false, spl.alg.ensemble.μ_init, 1)
		end,
	)

	return transition, state
end

function AbstractMCMC.step(
	rng::Random.AbstractRNG, model::Model, spl::Sampler{<:EnSS}, state::EnSSState; kwargs...)
	# Generate a log joint function.
	vi = state.vi
	densitymodel = AdvancedMH.DensityModel(
		Base.Fix1(LogDensityProblems.logdensity, Turing.LogDensityFunction(model, vi)),
	)

	# Compute the next states.
	states = last(AbstractMCMC.step(rng, densitymodel, spl.alg.ensemble, state.states))

	# Compute the next transition and state.
	transition = map(states) do _state
		vi = setindex!!(vi, _state.params, spl)
		t = Turing.Inference.Transition(getparams(model, vi), _state.lp, (accepted = _state.accepted, length_scale = _state.μ, t = _state.t))#, false, spl.alg.ensemble.μ_init, 1)
		return t
	end
	newstate = EnSSState(vi, states)

	return transition, newstate
end

function AbstractMCMC.bundle_samples(
	samples::Vector{<:Vector},
	model::AbstractMCMC.AbstractModel,
	spl::Sampler{<:EnSS},
	state::EnSSState,
	chain_type::Type{MCMCChains.Chains};
	save_state = false,
	sort_chain = false,
	discard_initial = 0,
	thinning = 1,
	kwargs...)
	# Convert transitions to array format.
	# Also retrieve the variable names.
	params_vec = map(Base.Fix1(_params_to_array, model), samples)

	# Extract names and values separately.
	varnames = params_vec[1][1]
	varnames_symbol = map(Symbol, varnames)
	vals_vec = [p[2] for p in params_vec]

	# Get the values of the extra parameters in each transition.
	extra_vec = map(get_transition_extras, samples)

	# Get the extra parameter names & values.
	extra_params = extra_vec[1][1]
	extra_values_vec = [e[2] for e in extra_vec]

	# Extract names & construct param array.
	nms = [varnames_symbol; extra_params]
	# `hcat` first to ensure we get the right `eltype`.
	x = hcat(first(vals_vec), first(extra_values_vec))

	# Pre-allocate to minimize memory usage.
	parray = Array{eltype(x), 3}(undef, length(vals_vec), size(x, 2), size(x, 1))
	for (i, (vals, extras)) in enumerate(zip(vals_vec, extra_values_vec))
		parray[i, :, :] = transpose(hcat(vals, extras))
	end

	# Get the average or final log evidence, if it exists.
	le = getlogevidence(samples, state, spl)

	# Set up the info tuple.
	info = (varname_to_symbol = OrderedDict(zip(varnames, varnames_symbol)),)
	if save_state
		info = merge(info, (model = model, sampler = spl, samplerstate = state))
	end

	# Concretize the array before giving it to MCMCChains.
	parray = MCMCChains.concretize(parray)

	# Chain construction.
	chain = MCMCChains.Chains(
		parray,
		nms,
		(internals = extra_params,);
		evidence = le,
		info = info,
		start = discard_initial + 1,
		thin = thinning,
	)

	return sort_chain ? sort(chain) : chain
end

# Store the new draw, its log density, and draw information
Transition(model::AdvancedMH.DensityModelOrLogDensityModel, params, accepted, μ, t) = Transition(params, logdensity(model, params), accepted, μ, t)
function Transition(model::AbstractMCMC.LogDensityModel, params, accepted, μ, t)
	return Transition(params, LogDensityProblems.logdensity(model.logdensity, params), accepted, μ, t)
end
function Transition(model::DynamicPPL.Model, vi::AbstractVarInfo, accepted::Bool, μ, t)
	θ = getparams(model, vi)
	lp = getlogp(vi)
	return Transition(θ, lp, accepted, μ, t)
end

function AbstractMCMC.step(
	rng::Random.AbstractRNG,
	model::AdvancedMH.DensityModelOrLogDensityModel,
	spl::EnsembleSliceSampler,
	params_prev::Vector{<:Transition};
	kwargs...)
	transitions = propose(rng, spl, model, params_prev)
	return transitions, transitions
end

function DifferentialMove(μ::Float64, walker_l::AbstractVector{<:Float64}, walker_m::AbstractVector{<:Float64})
	return μ * (walker_l - walker_m)
end


function propose(
	rng::Random.AbstractRNG,
	spl::EnsembleSliceSampler,
	model::AdvancedMH.DensityModelOrLogDensityModel,
	walkers_t::Vector{<:Transition})

	new_walkers = similar(walkers_t)

	T = eltype(walkers_t[1].params)

	t = walkers_t[1].t


	# get the number of walkers
	n_walkers = spl.n_walkers
	# num of walkers divided by 2
	n_wdiv2 = div(n_walkers, 2)
	M_adapt = spl.M_adapt
	max_steps = spl.max_steps
	ndim = length(walkers_t[1].params)
	μ = walkers_t[1].μ

	logdens(x::AbstractVector) = logdensity(model, x)
	## collect the walkers's positions in a matrix
	walkers = Matrix{T}(undef, ndim, n_walkers)
	for i in 1:n_walkers
		walkers[:, i] = walkers_t[i].params
	end
	# walkers .= [walker.params for walker in walkers_t]
	# wal = hcat([walker.params for walker in walkers_t]...)

	# set the contraction and expansion counts to zero
	R, L, N_e, N_c = 0.0, 0.0, 0, 0

	# get the log density all the walkers
	logdensities = [walker.lp for walker in walkers_t]



	# Randomly shuffle the walkers
	walker_indexes = randperm(rng, n_walkers)
	# get the first half of the shuffled walkers
	wl_ind = 1:n_wdiv2

	subset_a = @view walker_indexes[wl_ind]
	# get the second half of the shuffled walkers
	subset_b = @view walker_indexes[n_wdiv2+1:end]
	# get the two subsets
	sets = [[subset_a, subset_b], [subset_b, subset_a]]


	Widths = Vector{T}(undef, n_wdiv2)
	
	logdensities_shrink = Vector{T}(undef, n_wdiv2)
	positions_shrink = Matrix{T}(undef, ndim, n_wdiv2)
	# initialise the log densities for the left and right stepping points
	logdensities_left, logdensities_right = Vector{T}(undef, n_wdiv2), Vector{T}(undef, n_wdiv2)
	position_left, position_right = Matrix{T}(undef, ndim, n_wdiv2), Matrix{T}(undef, ndim, n_wdiv2)

	# iterate over the two sets
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
		while size(mask_left[mask_left],1) > 0 || size(mask_right[mask_right],1) > 0

			if size(mask_left[mask_left],1) > 0
				l += 1
			end
			if size(mask_right[mask_right],1) > 0
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
				nl = size(view(position_left,:, mask_left), 2)
				nr = size(view(position_right,:, mask_right), 2)
				logdensities_left[mask_left] = [logdens(view(view(position_left, :, mask_left), :, i)) for i in 1:nl]
				logdensities_right[mask_right] = [logdens(view(view(position_right, :, mask_right), :, i)) for i in 1:nr]
			end
			for j in wl_ind[mask_left]
				if  Y[j]< logdensities_left[j]
					L[j] -= 1
					N_e += 1
					J[j] -= 1
				else
					mask_left[j] = false
				end
			end
			for j in wl_ind[mask_right]
				if  Y[j]< logdensities_right[j]
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
		while size(mask[mask],1)>0

			Widths[mask] = rand(rng, Uniform(), size(mask[mask])) .* (view(R, mask) - view(L, mask)) .+ view(L, mask)
			

			positions_shrink[:, mask] = view(Widths, mask)' .* view(η, :, mask) + view(view(walkers, :, active), :, mask)
			logdensities_shrink[mask] = [logdens(view(view(positions_shrink, :, mask), :, i)) for i in 1:size(view(positions_shrink, :, mask), 2)]

			for j in wl_ind[mask]
				if Y[j] < logdensities_shrink[j]
					mask[j] = false
				else
					if Widths[j] <0.
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
		logdensities[active] = logdensities_shrink
		
		# update the transition
		# println(positions_shrink[i], logdensities_shrink[i], true, μ, t + 1)
		new_walkers[active] = [Transition(positions_shrink[:,i], logdensities_shrink[i], true, μ, t + 1) for i in wl_ind]
	end

	N_e = max(1, N_e)
	# tune_lengthscale(t, μ, N_e, N_c, M_adapt)

	if t <= M_adapt
		new_μ = 2μ * N_e / (N_e + N_c)
		new_walkers_2 = similar(new_walkers)
		for i in 1:n_walkers
			new_walkers_2[i] = Transition(model, new_walkers[i].params, true, new_μ, t)
		end
		return new_walkers_2
	else
		return new_walkers
	end
end

## Example

n_schools = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0] # estimated treatment effects
σ = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0] # standard error of the effect estimate


@model function school_reparam(y::AbstractVector{<:Float64}, σ::AbstractVector{<:Float64}, n_schools::Int64=8)
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5), lower = 0)
    θ ~ filldist(Normal(0, 1), n_schools)
    for i in 1:n_schools
        y[i] ~ Normal(μ + τ * θ[i], σ[i])
    end
end
mymod = school_reparam(y, σ)

spl = EnSS(50)
rng = Random.MersenneTwister(0)
using Profile

chain = sample(rng,mymod, spl, 200,progress=true)

using StatsPlots
chain[100:end,:,:]
plot(chain[100:end,:,:])
@profview chain = sample(rng,mymod, spl, 2_000,progress=true)