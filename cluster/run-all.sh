mkdir -p out/

printf 'Deleting previous outputs...\n'
rm -f /out/*.out

printf 'Deleting cached datasets...\n'
rm -f /tmp/*.dataset.tmp

printf 'Submitting all tasks...\n'
(cd out/ && for FILE in ../tasks/*.slurm; do printf ' %s -> ' "$FILE"; sbatch -p long_gpu $FILE; sleep 10s; done;)

printf '\n'
./status-cluster.sh

printf '\n'
./tail-all.sh