
test_class_1_only=False
lower_thres=0.3
xai_type=cam
num_features=50

for num_nodes in 32 64 128 256; do
    python experiment/experiment.py \
        --num_results 10000 \
        --num_worker 1 \
        --num_nodes $num_nodes \
        --num_features 5 \
        --corr False \
        --signal 0.0 \
        --lower_thres $lower_thres \
        --upper_thres $(echo "scale=2; 1 - $lower_thres" | bc) \
        --test_class_1_only $test_class_1_only\
        --xai_type $xai_type\
        --covariance_estimation True
    if [ $? -ne 0 ]; then
        echo "An error occurred. The script will be terminated."
        exit 1
    fi
done

