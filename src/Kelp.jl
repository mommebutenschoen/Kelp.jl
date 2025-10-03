module Kelp

# Write your package code here.

# ==============================================================
# KELP MODEL
# ==============================================================
# A model simulating kelp growth based on Broch et al. 2012
# ==============================================================

using Dates
using Interpolations
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Debugger
using Plots

export simulate_annual_cycle, day_length_hours, day_length_change, broch2012_params, test_plot

break_on(:error)

"""
Approximate daylight duration (hours) on *day* at *lat_deg*.

The formula uses a standard astronomical approximation:

    δ  =  23.44° · sin[ 2π (day + 284) / 365 ]       (solar declination)
    H0 =  arccos( –tan φ · tan δ )                   (hour angle at sunrise)
    L  =  (2 / 15) · H0 (in degrees)                  (day length in hours)

where φ is latitude. When |tan φ · tan δ| ≥ 1, day‑length is clipped to
0 h or 24 h (polar night/midnight sun).
"""
function day_length_hours(day::Integer, lat_deg::Real)
    # Wrap *day* to [1, 365]
    day = ((day - 1) % 365) + 1

    # Solar declination in radians
    delta_rad = deg2rad(23.44) * sin(2 * π * (day + 284) / 365)
    phi_rad = deg2rad(lat_deg)

    # cos H0 may fall outside [‑1, 1] near the poles
    cosH0 = -tan(phi_rad) * tan(delta_rad)
    if cosH0 >= 1
        return 0.0  # polar night
    elseif cosH0 <= -1
        return 24.0 # 24‑hour daylight
    else
        H0 = acos(cosH0) # radians
        return 2 * H0 * 24 / (2 * π) # hours
    end
end

function day_length_change(n::Integer, lat::Real)
    l = day_length_hours(n, lat)
    l_yesterday = day_length_hours(n - 1, lat)
    return l - l_yesterday
end

"""
Solve Eq. 12 for β using Newton’s method.

Parameters
----------
P_max : float
    Target P_max value (left‑hand side of Eq. 12).
alpha : float
    Photosynthetic efficiency parameter α.
I_sat : float
    Saturating light intensity I_sat.
beta0 : float, optional
    Initial guess for β (default 1 × 10⁻⁹).
max_iter : int, optional
    Maximum Newton iterations (default 10).
tol : float, optional
    Convergence tolerance on |f(β)| (default 1 × 10⁻¹²).

Returns
-------
float
    Approximate root β.
"""
function newton_beta(P_max::Real, alpha::Real, I_sat::Real;
                     beta0::Real=1e-9, max_iter::Integer=10, tol::Real=1e-12)
    function f(beta::Real)::Real
        # Eq. 12 right‑hand side minus target P_max
        term1 = alpha * I_sat / log(1 + alpha / beta)
        frac_alpha = alpha / (alpha + beta)
        frac_beta = beta / (alpha + beta)
        term2 = frac_alpha * (frac_beta ^ (beta / alpha))
        return term1 * term2 - P_max
    end

    beta = beta0
    for _ in 1:max_iter
        f_val = f(beta)
        if abs(f_val) < tol
            break
        end
        # Numerical derivative via central difference
        eps = 1e-6 * abs(beta) + 1e-12
        f_prime = (f(beta + eps) - f(beta - eps)) / (2 * eps)
        if isapprox(f_prime, 0.0, atol=1e-15)
            error("Derivative vanished; try different beta0")
        end
        beta_new = beta - f_val / f_prime
        # Keep β positive to satisfy domain of log and fractions
        beta = beta_new > 0 ? beta_new : beta * 0.5
    end
    return beta
end

function broch2012_params()
    params = Dict{Symbol, Any}(
        :A_0 => 6,
        :alpha => (3.75e-5) * (24 * 1e6) / (24 * 60 * 60),
        :C_min => 0.01,
        :C_struct => 0.2,
        :gamma => 0.5,
        :epsilon => 0.22,
        :I_sat => 90 * 24 * 60 * 60 / 1e6,
        :J_max => 1.4e-4 * 24,
        :k_a => 0.6,
        :k_dw => 0.0785,
        :k_c => 2.1213,
        :k_n => 2.72,
        :u_max => 0.18,
        :N_min => 0.01,
        :N_max => 0.022,
        :N_struct => 0.01,
        :P1 => 1.22e-3,
        :P2 => 1.3e-3,
        :Tp1 => 285,
        :Tp2 => 288,
        :a1 => 0.85,
        :a2 => 0.3,
        :R1 => 2.785e-4,
        :R2 => 5.429e-4,
        :Tr1 => 285,
        :Tr2 => 290,
        :Taph => 25924,
        :Tapl => 27774,
        :Tpl => 271,
        :Tph => 296,
        :U_65 => 0.03,
        :Kx => 4,
        :kd => 0.1,
        :a_1 => 0.85,
        :a_2 => 0.3
    )

    params[:Tap] = (1 / params[:Tp1] - 1 / params[:Tp2])^(-1) * log(params[:P2] / params[:P1])
    params[:Tar] = (1 / params[:Tr1] - 1 / params[:Tr2])^(-1) * log(params[:R2] / params[:R1])
    params[:m_2] = 0.039 / (2 * (1 - params[:N_min] / params[:N_max]))
    params[:m_1] = 0.18 / (2 * (1 - params[:N_min] / params[:N_max])) - params[:m_2]

    return params
end

function load_environmental_data(filename=nothing)
    if filename !== nothing
        # Load from file - implementation depends on format
    end

    # Generate mock data for testing
    days = 365 * 24
    t = 1:days
    par = 100 * ones(days)
    temp = 8 * ones(days)
    n = 4 * ones(days)
    u = 0.06 * ones(days)
    depth = 10 * ones(days)

    return Dict(
        :temp => temp,
        :par => par,
        :depth => depth,
        :days => t,
        :X => n,
        :U => u
    )
end

function interpolate_to_dates(orig_dates, orig_values, target_dates, kind="linear")
    orig_num = Float64.(Dates.value.(orig_dates))
    target_num = Float64.(Dates.value.(target_dates))

    itp = linear_interpolation(orig_num, orig_values, extrapolation_bc=Flat())
    return [itp(t) for t in target_num]
end

function load_broch_test_mb()
    tn = DataFrame(CSV.File("temperature_no3_data.csv"))
    tn."Date" = Date.(tn."Month", DateFormat("u-y")) .+ Day(14)

    ir = DataFrame(CSV.File("irradiance_daily_data.csv"))
    ir."Date" = Date.(ir."Date")

    start_date = DateTime(2010, 9, 15)
    run_days = 365
    days = [start_date + Hour(d) for d in 0:(run_days * 24 - 1)]

    par = interpolate_to_dates(ir."Date", ir."Irradiance_mol_photons_m2_day", days)
    temp = interpolate_to_dates(tn."Date", tn."Temperature_C", days)
    n = interpolate_to_dates(tn."Date", tn."NO3_mmol_N_m3", days)
    u = 0.06 * ones(length(days))
    depth = 10 * ones(length(days))
    t = 1:length(days)

    return Dict(
        :temp => temp,
        :par => par,
        :depth => depth,
        :days => days,
        :X => n,
        :U => u
    )
end

function load_broch_2012_mb()
    t_df = DataFrame(CSV.File("temperature_daily_2012.csv"))
    t_df."Date" = Date.(t_df."Date")

    x_df = DataFrame(CSV.File("no3_daily_2012.csv"))
    x_df."Date" = Date.(x_df."Date")

    ir_df = DataFrame(CSV.File("irradiance_daily_2012.csv"))
    ir_df."Date" = Date.(ir_df."Date")

    start_date = DateTime(2012, 2, 1)
    run_days = 364
    days = [start_date + Hour(d) for d in 0:(run_days * 24 - 1)]

    par = interpolate_to_dates(ir_df."Date", ir_df."Irradiance_mol_m2_day", days)
    temp = interpolate_to_dates(t_df."Date", t_df."Temperature_C", days)
    n = interpolate_to_dates(x_df."Date", x_df."NO3_mmol_N_m3", days)
    u = 0.06 * ones(length(days))
    depth = 10 * ones(length(days))

    return Dict(
        :temp => temp,
        :par => par,
        :depth => depth,
        :days => days,
        :X => n,
        :U => u
    )
end

function load_broch_test()
    origin = DateTime("2010-08-15")
    temp_file = DataFrame(CSV.File("./bs2012/temp.csv"))
    no3_file = DataFrame(CSV.File("./bs2012/no3.csv"))
    irr_file = DataFrame(CSV.File("./bs2012/irr.csv"))

    sort!(temp_file)
    sort!(no3_file)
    sort!(irr_file)

    temp_file."Date" = origin .+ Second.(convert_float_days_to_seconds.(temp_file."day"))
    no3_file."Date" = origin .+ Second.(convert_float_days_to_seconds.(no3_file."day"))
    irr_file."Date" = origin .+ Second.(convert_float_days_to_seconds.(irr_file."day"))

    start_date = DateTime(2010, 9, 10)
    run_days = 365
    days = [start_date + Hour(d) for d in 0:(run_days * 24 - 1)]

    par = interpolate_to_dates(irr_file."Date", irr_file."irr", days)
    temp = interpolate_to_dates(temp_file."Date", temp_file."temp", days)
    n = interpolate_to_dates(no3_file."Date", no3_file."no3", days)
    u = 0.06 * ones(length(days))
    depth = 10 * ones(length(days))

    return Dict(
        :temp => temp,
        :par => par,
        :depth => depth,
        :days => days,
        :X => n,
        :U => u
    )
end

function convert_float_days_to_seconds(days::Float64)
    days = round(Int64,days*86400)
    return Second(days)
end

"""
Simulate kelp growth over one year using provided environmental data

Parameters:
env_data: Dictionary with environmental time series (temp, PAR, depth)
params: Model parameters
"""
function simulate_annual_cycle(env_data=nothing ; params=nothing, lat=60.0, debug=false)
    if params === nothing
        params = broch2012_params()
    end

    if env_data === nothing
        env_data = load_environmental_data()
    elseif env_data == "Broch"
        env_data = load_broch_test()
    end

    days = length(env_data[:days])

    max_day_length_change = maximum([day_length_change(i, lat) for i in 1:366])

    # Initialize arrays
    states = Dict{Symbol, Vector{Float64}}(
        :A => zeros(days),
        :C => zeros(days),
        :N => zeros(days),
        :W_s => zeros(days),
        :W_d => zeros(days),
        :W_w => zeros(days),
        :C_total => zeros(days),
        :N_total => zeros(days)
    )

    if debug
        states[:Respiration] = zeros(days)
        states[:Exudation] = zeros(days)
        states[:Gross_photosynthesis] = zeros(days)
        states[:beta] = zeros(days)
        states[:A_lost] = zeros(days)
        states[:mu] = zeros(days)
        states[:v] = zeros(days)
        states[:J] = zeros(days)
        states[:mu_scale] = zeros(days)
        states[:f_area] = zeros(days)
        states[:f_temp] = zeros(days)
        states[:f_photo] = zeros(days)
    end

    # Initial conditions
    states[:A][1] = 30
    states[:C][1] = 0.35
    states[:N][1] = 0.01

    force_dt = env_data[:days]
    timestep = (force_dt[2] - force_dt[1]).value / (24 * 60 * 60 * 1000)
    day_number = [dayofyear(d) for d in force_dt]

    # Simulation loop
    for t in 2:length(force_dt)
        # Light attenuation with depth
        I = env_data[:par][t]

        # Specific growth rate
        # Effect of area on growth rate
        f_area = params[:m_1] * exp(-(states[:A][t-1] / params[:A_0])^2) + params[:m_2]

        # Effect of temperature on growth rate
        temp_val = env_data[:temp][t]
        if temp_val < 10
            f_temp = 0.08 * temp_val + 0.2
        elseif temp_val > 15
            if temp_val > 19
                f_temp = 0.0
            else
                f_temp = 19/4 - temp_val/4
            end
        else
            f_temp = 1.0
        end

        # Seasonal influence on growth rate
        lambda_val = day_length_change(day_number[t], lat) / max_day_length_change
        f_photo = params[:a_1] * (1 + sign(lambda_val) * (abs(lambda_val)^0.5)) + params[:a_2]

        # Growth rate
        mu = f_area * f_photo * f_temp * min(1 - params[:N_min] / states[:N][t-1], 1 - params[:C_min] / states[:C][t-1])

        # Apical frond loss
        v = (1e-6 * exp(params[:epsilon] * states[:A][t-1])) / (1 + 1e-6 * (exp(params[:epsilon] * states[:A][t-1]) - 1))

        # Area change
        da_dt = (mu - v) * states[:A][t-1]
        states[:A][t] = states[:A][t-1] + timestep * da_dt # update euler

        # Nutrient Change
        # Nutrient uptake rate per unit area
        J = (params[:J_max] * (1 - exp(-env_data[:U][t] / params[:U_65])) *
             ((params[:N_max] - states[:N][t-1]) / (params[:N_max] - params[:N_min])) *
             (env_data[:X][t] / (params[:Kx] + env_data[:X][t])))

        dn_dt = (params[:k_a]^-1) * J - mu * (states[:N][t-1] + params[:N_struct])
        states[:N][t] = states[:N][t-1] + timestep * dn_dt

        # Carbon Change
        # Photosynthesis
        p1 = params[:Tap] / params[:Tp1] - params[:Tap] / (env_data[:temp][t-1] + 273.15)
        p2 = params[:Tapl] / (env_data[:temp][t-1] + 273.15) - params[:Tapl] / params[:Tpl]
        p3 = params[:Taph] / params[:Tph] - params[:Taph] / (env_data[:temp][t-1] + 273.15)

        p_max = (params[:P1] * exp(p1)) / (1 + exp(p2) + exp(p3))
        beta = newton_beta(p_max, params[:alpha], params[:I_sat])

        # Gross photosynthetic rate
        p_s = (params[:alpha] * params[:I_sat]) / (log(1 + params[:alpha] / beta))
        p = p_s * (1 - exp(-(params[:alpha] * I) / p_s)) * exp(-(beta * I) / p_s)

        # Respiration
        r = params[:R1] * exp((params[:Tar] / params[:Tr1]) - (params[:Tar] / (env_data[:temp][t-1] + 273.15))) # In Kelvin

        # Exudation
        e = 1 - exp(params[:gamma] * (params[:C_min] - states[:C][t-1]))

        dc_dt = (1 / params[:k_a]) * (p * (1 - e) - r) - mu * (states[:C][t-1] + params[:C_struct])
        states[:C][t] = states[:C][t-1] + timestep * dc_dt

        # Extreme carbon limitation
        A_lost = 0.0
        if states[:C][t] < params[:C_min]
            A_lost = (states[:A][t] * (params[:C_min] - states[:C][t])) / params[:C_struct]
            states[:A][t] = states[:A][t] - A_lost
            states[:C][t] = params[:C_min]
        end

        # Diagnostics
        states[:W_s][t] = params[:k_a] * states[:A][t] # Structural weight
        states[:W_d][t] = params[:k_a] * (1 + params[:k_n] * (states[:N][t] - params[:N_min]) + params[:N_min] +
                                      params[:k_c] * (states[:C][t] - params[:C_min]) + params[:C_min]) * states[:A][t] # Dry weight

        states[:W_w][t] = params[:k_a] * ((1 / params[:k_dw]) + params[:k_n] * (states[:N][t] - params[:N_min]) + params[:N_min] +
                                      params[:k_c] * (states[:C][t] - params[:C_min]) + params[:C_min]) * states[:A][t] # Dry weight

        states[:C_total][t] = (states[:C][t] + params[:C_struct]) * states[:W_s][t] # Total carbon
        states[:N_total][t] = (states[:N][t] + params[:N_struct]) * states[:W_s][t] # Total nitrogen

        # Debug
        if debug
            states[:Respiration][t] = r
            states[:Exudation][t] = e
            states[:Gross_photosynthesis][t] = p
            states[:beta][t] = beta
            states[:A_lost][t] = A_lost
            states[:mu][t] = mu
            states[:v][t] = v
            states[:J][t] = J
            states[:mu_scale][t] = min(1 - params[:N_min] / states[:N][t-1], 1 - params[:C_min] / states[:C][t-1])
            states[:f_area][t] = f_area
            states[:f_temp][t] = f_temp
            states[:f_photo][t] = f_photo
        end
    end

    return states, env_data
end

function test_plot()

    results,env = simulate_annual_cycle("Broch", debug=true)
    params = broch2012_params()
    results[:time] = [n for n = 1:length(results[:A])]/24
    @bp

    # paper data
    offset = 0
    c = [0.312,0.349,0.292,0.195,0.224,0.266,0.281,0.307,0.363,0.373]
    c_t = [10.016,100.156,130.973,164.872,192.607,224.965,266.568,291.222,343.611,409.097] .+ offset
    n_t = [10.255,96.239,135.681,160.924,195.633,222.454,269.785,288.717,345.514,406.255] .+ offset
    n = [0.007,0.014,0.021,0.024,0.025,0.022,0.023,0.017,0.012,0.013]
    area_file = CSV.File("./bs2012/area.csv")
    c2_file = CSV.File("./bs2012/carbon.csv")
    n2_t = [14.8696,32.8696,57.913,75.1304,90.7826,104.8696,125.2174,145.5652,156.5217,162.7826,170.6087,178.4348,184.6957,194.087,200.3478,214.4348,237.913,248.8696,259.8261,267.6522,270.7826,273.913,278.6087,288,297.3913,308.3478,316.1739,327.1304,336.5217,352.1739,364.6957,378.7826,396.7826] .+ offset
    n2 = [0.0088,0.0095,0.0109,0.0124,0.0146,0.017,0.0209,0.0249,0.027,0.0284,0.0291,0.0285,0.0281,0.0265,0.025,0.0229,0.0189,0.0175,0.0165,0.0154,0.0145,0.0136,0.0127,0.0112,0.0102,0.0095,0.0091,0.0086,0.0083,0.0083,0.0084,0.0086,0.0088]

    # structural to dry weight conversion (paper plots g/g dry weight where as g/g structural weight is used in calculations), I think they are also plotting the total carbon not just the reserve
    n_factor = (results[:N] .- params[:N_min]) .* params[:k_n]
    c_factor = (results[:C] .- params[:C_min]) .* params[:k_c]
    w_factor = 1 .+ n_factor .+ c_factor .+ params[:C_min] .+ params[:N_min]

    #plt.figure()
    plt1=plot(results[:time],results[:A], c="red")
    plot!(plt1,area_file.day,area_file.area, c="darkblue")


    plt2=plot(results[:time],(results[:N] .+ params[:N_struct]) ./ w_factor, c="red")
    scatter!(plt2,n_t,n, c="lightblue")
    plot!(plt2,n2_t, n2, c="darkblue")

    plt3=plot(results[:time],(results[:C] .+ params[:C_struct]) ./ w_factor, c="red")
    scatter!(plt3,c_t,c,c="lightblue")
    plot!(plt3,c2_file.day, c2_file.carbon, c="darkblue")

end

end
