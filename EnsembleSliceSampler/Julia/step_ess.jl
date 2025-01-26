function AbstractMCMC.step(
	rng::Random.AbstractRNG,
	model_wrapper::AbstractMCMC.LogDensityModel,
	sampler::EnsembleSliceSampler,
	state::ESState)

	model = model_wrapper.logdensity

	# extract the sampler parameters
	μ = sampler.μ_init
	M_adapt = sampler.M_adapt
	n_walkers = sampler.n_walkers
	max_steps = sampler.max_steps

	# extract current state 
	# walker must ndim* n_walkers
	walkers, μ, t = state.x, state.μ, state.t
	ndim = size(walkers, 1)
	n_wdiv2 = div(n_walkers, 2)

	T = eltype(walkers)
	logdens(x::AbstractVector) = LogDensityProblems.logdensity(model, x)

	# sets 
	next_walkers = Matrix{T}(undef, ndim, n_walkers)
	# set the contraction and expansion counts to zero
	R, L, N_e, N_c = 0.0, 0.0, 0, 0

	# get the log density all the walkers
	logdensities = [logdens(x) for x in eachcol(walkers)]


	# Randomly shuffle the walkers
	walker_indexes = Random.randperm(rng, n_walkers)
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
				nl = size(view(position_left, :, mask_left), 2)
				nr = size(view(position_right, :, mask_right), 2)
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
		logdensities[active] = logdensities_shrink

		next_walkers[:, active] = positions_shrink
	end


	μ = tune_lengthscale(t, μ, N_e, N_c, M_adapt)
	t += 1
	state_new = ESState(next_walkers, μ, t)
	return ESSample(next_walkers), state_new
end