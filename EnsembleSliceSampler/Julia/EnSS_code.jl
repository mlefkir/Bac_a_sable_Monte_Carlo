using Distributions
using Random: Random
using AdvancedMH
using AbstractMCMC
using LogDensityProblems
using AdvancedMH: logdensity
using Turing.Inference: SampleFromPrior, AbstractVarInfo, Sampler, Model, setindex!!, _params_to_array, VarInfo, getparams, InferenceAlgorithm, get_transition_extras, getlogevidence, getlogp


function get_complementary(i, N)
	indices = collect(1:N)
	deleteat!(indices, i)
	return indices
end

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
end

struct EnSS{space, E <: EnsembleSliceSampler} <: InferenceAlgorithm
	ensemble::E
end

function EnSS(n_walkers::Int, μ_init = 1.0, M_adapt = 50, max_steps = 10000)
	ensemble = EnsembleSliceSampler(μ_init, M_adapt, n_walkers, max_steps)
	return EnSS{(), typeof(ensemble)}(ensemble)
end

DynamicPPL.getspace(::EnSS) = ()


struct EnSSState{V <: AbstractVarInfo, S}
	vi::V
	states::S
end

function AbstractMCMC.step(
	rng::Random.AbstractRNG,
	model::Model,
	spl::Sampler{<:EnSS};
	kwargs...)

	# Sample from the prior
	n = spl.alg.ensemble.n_walkers
	vis = [VarInfo(rng, model, SampleFromPrior()) for _ in 1:n]

	# Compute initial transition and states.
	# transition = map(Base.Fix1(Turing.Inference.Transition, model), vis)
	transition = [Turing.Inference.Transition(getparams(model, vi), getlogp(vi), (accepted = true, length_scale = spl.alg.ensemble.μ_init, t = 1)) for vi in vis]
	# println(transition[1])
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

function propose(
	rng::Random.AbstractRNG,
	spl::EnsembleSliceSampler,
	model::AdvancedMH.DensityModelOrLogDensityModel,
	walkers::Vector{<:Transition})
	new_walkers = similar(walkers)
	n_walkers = spl.n_walkers
	M_adapt = spl.M_adapt
	μ = walkers[1].μ

	R, L, N_e, N_c = 0.0, 0.0, 0, 0
	for i in 1:n_walkers
		walker = walkers[i]

		indices = get_complementary(i, n_walkers)
		l, m = sample(rng, indices, 2, replace = false)

		new_walkers[i], R, L, N_e, N_c = move(rng, spl, model, walker, walkers[l], walkers[m], R, L, N_e, N_c)
	end

	t = new_walkers[1].t
	N_e = max(1, N_e)

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

function DifferentialMove(μ, walker_l, walker_m)
	return μ * (walker_l - walker_m)
end

function move(
	rng::Random.AbstractRNG,
	spl::EnsembleSliceSampler,
	model::AdvancedMH.DensityModelOrLogDensityModel,
	walker::Transition,
	walker_l::Transition,
	walker_m::Transition, R::Float64, L::Float64, N_e::Int, N_c::Int)
	max_steps = spl.max_steps
	μ = walker.μ

	X′ = 0.0
	# get the move to the new position
	ηₖ = DifferentialMove(μ, walker_l.params, walker_m.params)
	# draw y position 
	δ = rand(rng, Exponential(1))
	Y = walker.lp - δ


	L = -rand(rng)
	R = L + 1
	l = 0
	# println("attention!")
	Y′ = Transition(model, walker.params + L .* ηₖ, false, μ, walker.t).lp

	while Y < Y′
		L = L - 1
		N_e = N_e + 1
		l += 1
		if l == max_steps
			error("Max steps reached", " iteration: ", t, " walker: ", k)
		end
		Y′ = Transition(model, walker.params + L .* ηₖ, false, μ, walker.t).lp
	end
	l = 0
	Y′ = Transition(model, walker.params + R .* ηₖ, false, μ, walker.t).lp
	while Y < Y′
		R = R + 1
		N_e = N_e + 1
		l += 1
		if l == max_steps
			error("Max steps reached")
		end
		Y′ = Transition(model, walker.params + R .* ηₖ, false, μ, walker.t).lp
	end

	l = 0
	while true
		l += 1
		X′ = rand(rng, Uniform(L, R))
		Y′ = Transition(model, walker.params + X′ .* ηₖ, false, μ, walker.t).lp
		if Y < Y′
			break
		end
		if X′ < 0
			L = X′
			N_c = N_c + 1
		else
			R = X′
			N_c = N_c + 1
		end
		if l == max_steps
			error("Max steps reached")
		end
	end

	Xₖ = X′ .* ηₖ + walker.params

	return Transition(model, Xₖ, true, μ, walker.t + 1), R, L, N_e, N_c

end
