for CMU_SLICE in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
python3 run.py --dataset cmu --mode sparse_to_dense --cmu_slice $CMU_SLICE
done
