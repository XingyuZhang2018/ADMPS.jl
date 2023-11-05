nohup  julia --project=@. ./run/triising.jl 4 cpu > ./data/triising-chi4.dat &
nohup  julia --project=@. ./run/triising.jl 8 cpu > ./data/triising-chi8.dat &
nohup  julia --project=@. ./run/triising.jl 12 cpu > ./data/triising-chi12.dat &
nohup  julia --project=@. ./run/triising.jl 16 cpu > ./data/triising-chi16.dat &

nohup  julia --project=@. ./run/triisingAT.jl 4 cpu > ./data/triisingAT-chi4.dat &
nohup  julia --project=@. ./run/triisingAT.jl 8 cpu > ./data/triisingAT-chi8.dat &
nohup  julia --project=@. ./run/triisingAT.jl 12 cpu > ./data/triisingAT-chi12.dat &
nohup  julia --project=@. ./run/triisingAT.jl 16 cpu > ./data/triisingAT-chi16.dat &

nohup  julia --project=@. ./run/triisingto4.jl 4 cpu > ./data/triisingto4-chi4.dat &
nohup  julia --project=@. ./run/triisingto4.jl 8 cpu > ./data/triisingto4-chi8.dat &
nohup  julia --project=@. ./run/triisingto4.jl 12 cpu > ./data/triisingto4-chi12.dat &
nohup  julia --project=@. ./run/triisingto4.jl 16 cpu > ./data/triisingto4-chi16.dat &

nohup  julia --project=@. ./run/dcprec.jl 4 cpu > ./data/dcprec-chi4.dat &
nohup  julia --project=@. ./run/dcprec.jl 8 cpu > ./data/dcprec-chi8.dat &
nohup  julia --project=@. ./run/dcprec.jl 12 cpu > ./data/dcprec-chi12.dat &
nohup  julia --project=@. ./run/dcprec.jl 16 cpu > ./data/dcprec-chi16.dat &

nohup  julia --project=@. ./run/triisingPR6.jl 4 cpu > ./data/triisingPR6-chi4.dat &
nohup  julia --project=@. ./run/triisingPR6.jl 8 cpu > ./data/triisingPR6-chi8.dat &
nohup  julia --project=@. ./run/triisingPR6.jl 12 cpu > ./data/triisingPR6-chi12.dat &
nohup  julia --project=@. ./run/triisingPR6.jl 16 cpu > ./data/triisingPR6-chi16.dat &

nohup  julia --project=@. ./run/J1J2.jl 4 cpu > ./data/J1J2-chi4.dat &
nohup  julia --project=@. ./run/J1J2.jl 8 cpu > ./data/J1J2-chi8.dat &
nohup  julia --project=@. ./run/J1J2.jl 12 cpu > ./data/J1J2-chi12.dat &
nohup  julia --project=@. ./run/J1J2.jl 16 cpu > ./data/J1J2-chi16.dat &

nohup  julia --project=@. ./run/triisingPR9.jl 4 cpu > ./data/triisingPR9-chi4.dat &
nohup  julia --project=@. ./run/triisingPR9.jl 8 cpu > ./data/triisingPR9-chi8.dat &
nohup  julia --project=@. ./run/triisingPR9.jl 12 cpu > ./data/triisingPR9-chi12.dat &
nohup  julia --project=@. ./run/triisingPR9.jl 16 cpu > ./data/triisingPR9-chi16.dat &

nohup  julia --project=@. ./run/triising099.jl 4 cpu > ./data/triising099-chi4.dat &
nohup  julia --project=@. ./run/triising099.jl 8 cpu > ./data/triising099-chi8.dat &
nohup  julia --project=@. ./run/triising099.jl 12 cpu > ./data/triising099-chi12.dat &
nohup  julia --project=@. ./run/triising099.jl 16 cpu > ./data/triising099-chi16.dat &

nohup srun --pty --partition=v100 --gres=gpu:1 julia --project=@. ./run/triising.jl 32 gpu > ./data/triising-chi32.dat &
nohup srun --pty --partition=v100 --gres=gpu:1 julia --project=@. ./run/triising.jl 64 gpu > ./data/triising-chi64.dat &

nohup  julia --project=@. ./run/triisingU.jl 4 cpu > ./data/triisingU-chi4.dat &
nohup  julia --project=@. ./run/triisingU.jl 8 cpu > ./data/triisingU-chi8.dat &
nohup  julia --project=@. ./run/triisingU.jl 12 cpu > ./data/triisingU-chi12.dat &
nohup  julia --project=@. ./run/triisingU.jl 16 cpu > ./data/triisingU-chi16.dat &

nohup  julia --project=@. ./run/triisingU.jl 4 cpu > ./data/triisingU-chi4.dat &
nohup  julia --project=@. ./run/triisingU.jl 8 cpu > ./data/triisingU-chi8.dat &
nohup  julia --project=@. ./run/triisingLoad.jl 12 cpu > ./data/triisingload-chi12.dat &
nohup  julia --project=@. ./run/triisingU.jl 16 cpu > ./data/triisingU-chi16.dat &


nohup  julia --project=@. ./run/ising.jl 4 cpu > ./data/ising-chi4.dat &
nohup  julia --project=@. ./run/ising.jl 8 cpu > ./data/ising-chi8.dat &
nohup  julia --project=@. ./run/ising.jl 12 cpu > ./data/ising-chi12.dat &
nohup  julia --project=@. ./run/ising.jl 16 cpu > ./data/ising-chi16.dat &

# nohup srun --pty --partition=a800 --gres=gpu:1 julia --project=@. ./run/ising.jl 16 gpu > ./data/ising-chi16.dat &
# nohup srun --pty --partition=a800 --gres=gpu:1 julia --project=@. ./run/ising.jl 32 gpu > ./data/ising-chi32.dat &
# nohup srun --pty --partition=a800 --gres=gpu:1 julia --project=@. ./run/ising.jl 64 gpu > ./data/ising-chi64.dat &

nohup julia --project=@. ./run/randn.jl 4 cpu > ./data/randc103-chi4.dat &
nohup julia --project=@. ./run/randn.jl 8 cpu > ./data/randc103-chi8.dat &
nohup julia --project=@. ./run/randn.jl 12 cpu > ./data/randc103-chi12.dat &
nohup julia --project=@. ./run/randn.jl 16 cpu > ./data/randc103-chi16.dat &

nohup julia --project=@. ./run/rand.jl 4 cpu > ./data/rand105-chi4.dat &
nohup julia --project=@. ./run/rand.jl 8 cpu > ./data/rand105-chi8.dat &
nohup julia --project=@. ./run/rand.jl 12 cpu > ./data/rand105-chi12.dat &

nohup srun --pty --partition=a800 --gres=gpu:1 julia --project=@. ./run/randn.jl 16 cpu > ./data/randn-chi16.dat &
# nohup srun --pty --partition=a800 --gres=gpu:1 julia --project=@. ./run/randn.jl 32 gpu > ./data/randn-chi32.dat &
# nohup srun --pty --partition=a800 --gres=gpu:1 julia --project=@. ./run/randn.jl 64 gpu > ./data/randn-chi64.dat &