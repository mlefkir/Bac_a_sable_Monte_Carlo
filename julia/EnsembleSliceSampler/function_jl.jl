using Distributions, Random, LinearAlgebra, StatsBase
using Combinatorics
using ProgressMeter

function tune_lengthscale(t, μ, N_e, N_c, M_adapt)
	N_e = max(1, N_e)

	if t <= M_adapt
		return 2μ * N_e / (N_e + N_c)
	else
		return μ
	end
end

function DifferentialMove(μ, walker_l, walker_m)
	return μ * (walker_l - walker_m)
end

function sampleMything(rng, logdens, walkers, N_iter, M_adapt = 100, max_steps = 10^4, μ = 1.0)

	# walkers is of size

	ndim, n_walkers = size(walkers)
	# num of walkers divided by 2
	n_wdiv2 = div(n_walkers, 2)

	logdensities = [logdens(walkers[:, i]) for i in 1:n_walkers]

	# Randomly shuffle the walkers
	walker_indexes = randperm(rng, n_walkers)
	# get the first half of the shuffled walkers
	subset_a = walker_indexes[1:n_wdiv2]
	# get the second half of the shuffled walkers
	subset_b = walker_indexes[n_wdiv2+1:end]
	# get the two subsets
	sets = [[subset_a, subset_b], [subset_b, subset_a]]


	S_save = zeros(ndim,n_walkers, N_iter)

	@showprogress for t in 1:N_iter
		R, L, N_e, N_c = 0, 0, 0, 0

		# loop over the walkers
		for set in sets
			active, inactive = set

			### DifferentialMove ###
			# get all the permutations of the inactive walkers
			permuts = collect(permutations(inactive, 2))
			# get the number of permutations
			pairs = sample(rng, permuts, n_wdiv2, replace = false)
			# iterate over the pairs
			η = hcat([DifferentialMove(μ, walkers[:, p[1]], walkers[:, p[2]]) for p in pairs]...) # (ndim, n_wdiv2)

			# masks the for the left and right stepping
			mask_left = fill(true, n_wdiv2)
			mask_right = fill(true, n_wdiv2)

			# initialise the log densities for the left and right stepping points
			logdensities_left, logdensities_right = Vector{Float64}(undef, n_wdiv2), Vector{Float64}(undef, n_wdiv2)
			position_left, position_right = Matrix{Float64}(undef, ndim, n_wdiv2), Matrix{Float64}(undef, ndim, n_wdiv2)

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

				for j in (1:n_wdiv2)[mask_left]
					if J[j] < 1
						mask_left[j] = false
					end
				end
				for j in (1:n_wdiv2)[mask_right]
					if K[j] < 1
						mask_right[j] = false
					end
				end
				# println(size(position_left[:, mask_left])," L[mask_left]: ", size(L[mask_left]), " η[:, mask_left]: ", size(η[:, mask_left]), " walkers[:, active]: ", size(walkers[:, active][:, mask_left]))
				position_left[:, mask_left] = L[mask_left]' .* η[:, mask_left] + walkers[:, active][:, mask_left]
				position_right[:, mask_right] = R[mask_right]' .* η[:, mask_right] + walkers[:, active][:, mask_right]

				if size(position_left[:, mask_left],1) + size(position_right[:, mask_right],1) < 0
					logdensities_left[mask_left] = []
					logdensities_right[mask_right] = []
					l -= 1
				else
					logdensities_left[mask_left] = [f(position_left[:, mask_left][:, i]) for i in 1:size(position_left[:, mask_left],2)]
					logdensities_right[mask_right] = [f(position_right[:, mask_right][:, i]) for i in 1:size(position_right[:, mask_right],2)]
				end
				for j in (1:n_wdiv2)[mask_left]
					if  Y[j]< logdensities_left[j]
						L[j] -= 1
						N_e += 1
						J[j] -= 1
					else
						mask_left[j] = false
					end
				end
				for j in (1:n_wdiv2)[mask_right]
					if  Y[j]< logdensities_right[j]
						R[j] += 1
						N_e += 1
						K[j] -= 1
					else
						mask_right[j] = false
					end
				end
			end
			# println("N_e: ", N_e)


			## shrink the interval##
			Widths = Vector{Float64}(undef, n_wdiv2)
			logdensities_shrink = Vector{Float64}(undef, n_wdiv2)
			positions_shrink = Matrix{Float64}(undef, ndim, n_wdiv2)
			mask = fill(true, n_wdiv2)
			l = 0
			while size(mask[mask],1)>0

				Widths[mask] = rand(rng,Uniform(), size(mask[mask])) .* (R[mask] - L[mask]) .+ L[mask]
				
				positions_shrink[:, mask] = Widths[mask]' .* η[:, mask] + walkers[:, active][:, mask]
				logdensities_shrink[mask] = [logdens(positions_shrink[:, mask][:, i]) for i in 1:size(positions_shrink[:, mask],2)]
			
				for j in (1:n_wdiv2)[mask]
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
			S_save[:, active, t] = positions_shrink
			logdensities[active] = logdensities_shrink
		end
		μ = tune_lengthscale(t, μ, N_e, N_c, M_adapt)

	end
	# chains = Chains(permutedims(S_save, [3, 2, 1]))
	return Chains(permutedims(S_save, [3, 1, 2]))
end
