using Distributions, Random, LinearAlgebra, StatsBase
using ProgressMeter

function tune_lengthscale(t, μ, N_e, N_c, M_adapt)
	N_e = max(1, N_e)

	if t <= M_adapt
		return 2μ * N_e / (N_e + N_c)
	else
		return μ
	end
end

function get_complementary(i, N)
	indices = collect(1:N)
	deleteat!(indices, i)
	return indices
end

function get_direction_vector(S, l, m, μ)
	return μ * (S[l] - S[m])
end

""" 
	DifferentialMove(rng, k, μ, S, N)

	Perform a differential move for walker k.

	# Arguments
	- `rng::AbstractRNG`: Random number generator.
	- `k::Int`: Index of the walker.
	- `μ::Float64`: Lengthscale.
	- `S::Array{Float64, 2}`: Array of walker positions.
	- `N::Int`: Number of walkers.

	# Returns
	- `ηₖ::Array{Float64, 1}`: Differential move.
"""
function DifferentialMove(rng, k, μ, S, N)
	# work on walker k
	indices = get_complementary(k, N)
	# draw two random indices from the complementary set, without replacement
	l, m = sample(rng, indices, 2, replace = false)
	return get_direction_vector(S, l, m, μ)
end

# sample_dict = rand(OrderedDict, inferenceModel(y)).vals
# N2 = convert(Int, N / 2)

function sampleMything(rng, f, S, D, N_iter, N, M_adapt = 100, max_steps = 10^4, μ = 1.0)

	S_save = zeros(N, D, N_iter)
	μ_values = zeros(N_iter)
	R_values = zeros(N_iter)
	L_values = zeros(N_iter)
	N_e_values = zeros(N_iter)
	N_c_values = zeros(N_iter)

	@showprogress for t in 1:N_iter
		R, L, N_e, N_c = 0, 0, 0, 0
		X′ = 0

		# loop over the walkers
		@showprogress for k in 1:N

			Xₖ = S[k] # get the current position of walker k
			ηₖ = DifferentialMove(rng, k, μ, S, N) # get the differential move

			δ = rand(rng, Exponential(1))
			Y = f(Xₖ) - δ

			L = -rand(rng)
			R = L + 1
			l = 0
			while Y < f(L .* ηₖ + Xₖ)
				L = L - 1
				N_e = N_e + 1
				l += 1
				if l == max_steps
					println("L: ", L, " Y: ", Y, " f(L): ", f(L .* ηₖ + Xₖ))
					error("Max steps reached", " iteration: ", t, " walker: ", k)
				end
			end
			l = 0
			while Y < f(R .* ηₖ + Xₖ)
				R = R + 1
				N_e = N_e + 1
				l += 1
				if l == max_steps
					println("L: ", R, " Y: ", Y, " f(R): ", f(R .* ηₖ + Xₖ))
					error("Max steps reached")
				end
			end

			l = 0
			while true
				l += 1
				X′ = rand(rng, Uniform(L, R))
				Y′ = f(X′ .* ηₖ + Xₖ)
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
					println("L: ", R, " Y: ", Y, " f(R): ", f(R .* ηₖ + Xₖ))

					error("Max steps reached")
				end
			end
			# println("Steps: ", l)
			Xₖ = X′ .* ηₖ + Xₖ
			S_save[k, :, t] = Xₖ
			S[k] = Xₖ
		end
		μ_values[t] = μ
		R_values[t] = R
		L_values[t] = L
		N_e_values[t] = N_e
		N_c_values[t] = N_c
		# println("R: ", R, " L: ", L, " N_e: ", N_e, " N_c: ", N_c, " μ: ", μ)
		μ = tune_lengthscale(t, μ, N_e, N_c, M_adapt)
	end

	chains = Chains(permutedims(S_save, [3, 2, 1]))
	return chains
end


# rng = MersenneTwister(0)
# S = [rand(rng, OrderedDict, inferenceModel(y)).vals for i in 1:N]
# plot(chains)
# plot(chains[500:end, :, [1,3,5,2,6]])
