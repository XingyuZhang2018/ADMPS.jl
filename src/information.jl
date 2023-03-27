function print_header(verbosity::Int)
    if verbosity > 0
        message = "time  iter   loss           grad_norm\n"
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end
end