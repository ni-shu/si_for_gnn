test_class_1_only=False
lower_thres=0.3
xai_type=cam
default_num_nodes=256
default_num_features=5

for nongauss_type in skewnorm exponnorm gennormsteep gennormflat t; do
    for nongauss_wd in 0.01 0.02 0.03 0.04 0.05 0.10 0.15; do
        python experiment/experiment.py \
            --num_results 1000 \
            --num_worker 1 \
            --num_nodes $default_num_nodes \
            --num_features $default_num_features \
            --corr False \
            --signal 0.0 \
            --lower_thres $lower_thres \
            --upper_thres $(echo "scale=2; 1 - $lower_thres" | bc) \
            --test_class_1_only $test_class_1_only\
            --xai_type $xai_type\
            --nongauss True\
            --nongauss_type $nongauss_type\
            --nongauss_wd $nongauss_wd
        if [ $? -ne 0 ]; then
            echo "An error occurred. The script will be terminated."
            exit 1
        fi
    done
done


