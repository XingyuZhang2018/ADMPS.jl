set terminal png
set output "./plot/tttt.png"
set term png size 1200, 800

set multiplot layout 2,2 columns
    set logscale y
    set xlabel "beta,(L=6)"
    set format y "10^{%L}"

    # set xr [1:400]
    # set yr [1.6:1.8]

    plot './data/tttt.dat' using ($1):($4) title "|A|",
    plot './data/tttt.dat' using ($1):($2) title "|OO^T-O^TO|"
    plot './data/tttt.dat' using ($1):($3) title "|A|-rho(A)"
    plot './data/tttt.dat' using ($1):($5) title "error of log(Z)"

unset multiplot