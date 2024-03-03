cd algorithm/To_Neural_Network/

for alg in NN
do
  for jammer in det1
  do
    for env_seed in 3003
    do
      for alg_seed in {5009..5010}
      do
        echo Running $alg-$jammer-$env_seed-$alg_seed
        nohup python3 -m code_radar_level --alg $alg --iter 10000 --env_seed $env_seed --alg_seed $alg_seed --jammer $jammer --cuda 0 &
      done
    done
  done
done












