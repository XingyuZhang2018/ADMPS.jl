# Lagacy code for understandings

# An equivalent substitution for loss_eig
function loss_power(AL)
    AL = reshape(AL,(D,2,D))
    AR = getAR(AL)
    FLp,FRp = HL,HR

    for i = 1:3000 # large number
        FLp = ein"((adf,abc),dgeb),fgh -> ceh"(FLp,AL,Hm,conj(AL))
        FRp = ein"((ceh,abc),dgeb),fgh -> adf"(FRp,AR,Hm,conj(AR))
    end

    h_eff_0, h_eff_1 = effective_hamiltonian_01(FLp,FRp,Hm,AL,AR)
    f_per_site = free_energy_persite(h_eff_0,h_eff_1)
    @show f_per_site
    
    # end
#             # end
    return f_per_site
    # return f0
end

#get AR
function getAR(AL)
    CR, AR, λR = rightorth(reshape(AL,(D,2,D)))
    # ein"ij,jkl,lm->ikm"(inv(CR),AL,((CR))) ./ AR .== 1.0
    return AR
end