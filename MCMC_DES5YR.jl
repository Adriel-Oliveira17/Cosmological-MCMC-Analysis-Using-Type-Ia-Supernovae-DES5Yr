using Turing, Distributions, StatsPlots, MCMCChains, QuadGK, CSV, DataFrames, LinearAlgebra, Statistics, DelimitedFiles, StatsBase, DataFrames

# Base path
base_path = "path"

# Reading files
data = CSV.read(joinpath(base_path, "DES5YR/DES-SN5YR_HD+MetaData.csv"), DataFrame; delim = ',', ignorerepeated=true)

zcmb = data[:,:zCMB]
mu = data[:,:MU]
mu_error = data[:,:MUERR_FINAL]

# Open the file for reading
f = open("path/DES5YR/covsys_000.txt", "r")

# Read the first line and parse it as an integer (matrix size)
n = parse(Int, readline(f))

# Read the remaining data as Float64 numbers and flatten into a vector
values = readdlm(f, Float64) |> vec

# Close the file
close(f)

# Reshape the vector into an n x n matrix
sys_errors = reshape(values, n, n)

# Build diagonal matrix of variances (square of errors)
est_errors_mat = Diagonal(mu_error.^ 2)

# Sum the covariance matrices
Cij = sys_errors + est_errors_mat

# Calculate the inverse of the total covariance matrix
InvCovTot = inv(Cij)

#println(InvCovTot)

c = 299792.458  # speed of light in km/s

# Cosmological H(z) function - Global
function H(z, om, H0)
    #Or = om/(1 +((2.5*10^4)*om*(H0/100)^2*(T0cmb/2.7)^(-4)))
    Or = 0
    return H0 * sqrt(Or* (1.0 + z)^4 + om * (1 + z)^3 + (1 - om -Or))
end

# Luminosity distance function dL for supernova model
function dL(z, om, H0)
    integrando(zp) = 1 / (sqrt(om * (1.0 + zp)^3 + (1.0 - om)))
    χ = [quadgk(integrando, 0, zi)[1] for zi in z]  # numerical integration over redshift
    dl = (1 .+ z) .* (c/H0) .* χ                     # luminosity distance calculation
    return dl
end

# Define the log-likelihood function for supernova data
function log_likelihood_sn(om, H0, zcmb, mu, InvCovTotal)
    mu_model = 5 .* log10.(dL(zcmb, om, H0)) .+ 25  # theoretical distance modulus
    diff = mu_model .- mu                            # residuals
    chi2sn = dot(diff, InvCovTot * diff)             # chi-squared with covariance matrix
    return -0.5 * chi2sn                             # log-likelihood value
end

# Define the cosmological model using Turing
@model function cosmological_model(zcmb, mu, InvCovTot)
    # Priors
    om ~ Uniform(0.1, 0.5)
    H0 ~ Uniform(60.0, 80.0)

    # Log-likelihood — explicitly register it
    ll = log_likelihood_sn(om, H0, zcmb, mu, InvCovTot)
    Turing.@addlogprob! ll
end

# MCMC sampler configuration: sample from the posterior
chain = sample(cosmological_model(zcmb, mu, InvCovTot), NUTS(), 10000)

# Display results
println(chain)

# Plot results
plot(chain)
