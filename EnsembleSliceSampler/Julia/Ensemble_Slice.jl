using Distributions, Plots, Random, LinearAlgebra, Turing, StatsPlots, StatsBase
using ProgressMeter

fun(x, α, β, γ) = @. α * exp(-β * x^2) * cos(γ * x)
x = 0.0:0.05:5.0
y = fun(x, 3.0, 0.30, -4)
yerr = randn(length(x)) * 0.5
y = randn(length(x)) .* yerr + y
plot(x, y, seriestype = :scatter, yerr = yerr, label = "data")
t = LinRange(0, 5, 1000)
plot!(t, fun(t, 3.0, 0.30, -4), label = "true function", lw = 4)

using DelimitedFiles

writedlm("data.txt", [x y abs.(yerr)])

function logprior(α, β, γ)
	α_prior = LogNormal(2, 1)
	β_prior = LogNormal(0, 1)
	γ_prior = Normal(0, 1)
	return logpdf(α_prior, α) + logpdf(β_prior, β) + logpdf(γ_prior, γ)
end

function loglikelihood(y, yerr, α, β, γ)
	return -0.5 * sum((fun(x, α, β, γ) - y) .^ 2 ./ yerr .^ 2)
end

function logposterior(y, yerr, α, β, γ)
	return loglikelihood(y, yerr, α, β, γ) + logprior(α, β, γ)
end

function lpost(α, β, γ)
	return logposterior(y, yerr, α, β, γ)
end
Σ = diagm(yerr .^ 2)
@model function inferenceModel(y)
	α ~ LogNormal(2, 1)
	β ~ LogNormal(0, 1)
	γ ~ Normal(-2, 1)

	y ~ MvNormal(fun(x, α, β, γ), 1)
end


sample_dict = rand(OrderedDict, inferenceModel(y)).vals

# f(X) = (logjoint(inferenceModel(y), (α = X[1], β = X[2], γ = X[3])))
f(X) = lpost(X[1], X[2], X[3])

D = 3 # number of parameters
N = 10 * D # number of walkers

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

function DifferentialMove(rng, k, μ, S)
	# work on walker k
	indices = get_complementary(k, N)
	# draw two random indices from the complementary set, without replacement
	l, m = sample(rng, indices, 2, replace = false)
	return get_direction_vector(S, l, m, μ)
end

# sample_dict = rand(OrderedDict, inferenceModel(y)).vals
N2 = convert(Int, N / 2)


N_iter = 10000

bad = []
μ = 1.0
M_adapt = 1000
max_steps = 10^4
S_save = zeros(N, D, N_iter);
μ_values = zeros(N_iter)
R_values = zeros(N_iter)
L_values = zeros(N_iter)
N_e_values = zeros(N_iter)
N_c_values = zeros(N_iter)

rng = MersenneTwister(0)
S = [rand(rng, OrderedDict, inferenceModel(y)).vals for i in 1:N]
@showprogress for t in 1:N_iter
	# println(stdout,"Iteration: ", t);flush(stdout)
	R, L, N_e, N_c = 0, 0, 0, 0
	X′ = 0

	# loop over the walkers
	@showprogress for k in 1:N
		# println(stdout,"Walker: ", k);flush(stdout)

		Xₖ = S[k] # get the current position of walker k
		ηₖ = DifferentialMove(rng, k, μ, S) # get the differential move
		# Y  = rand(rng) * f(Xₖ)
		# println("Xₖ: ", Xₖ)
		# println("f(Xₖ)", f(Xₖ))
		delta = rand(rng, Exponential(1))
		# println("delta: ", delta)
		Y = f(Xₖ) - delta

		L = -rand(rng)
		R = L + 1
		l = 0
		while Y < f(L .* ηₖ + Xₖ)
			L = L - 1
			N_e = N_e + 1
			l += 1
			if l == max_steps
				println("L: ", L, " Y: ", Y, " f(L): ", f(L .* ηₖ + Xₖ))
				error("Max steps reached"," iteration: ", t, " walker: ", k)
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
				error("Max steps reached")
			end
		end
		# println("Steps: ", l)
		if l == max_steps
			println("Max steps reached")
			push!(bad, [t, k])
		end
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


using MCMCChains

chains = Chains(permutedims(S_save, [3, 2, 1]))
plot(chains)
# plot(chains[500:end, :, [1,3,5,2,6]])