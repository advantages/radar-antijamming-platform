cd algorithm/To_Bandit/

for alg in BL
do
  for jammer in det1
  do
    for env_seed in 3003
    do
      for alg_seed in {5009..5010}
      do
        echo Running $alg-$jammer-$env_seed-$alg_seed
        nohup python3 code_radar_level.py --alg $alg --iter 10000 --env_seed $env_seed --alg_seed $alg_seed --jammer $jammer --cuda 0 &
      done
    done
  done
done












