
test_class_1_only=False
lower_thres=0.3
xai_type=cam
default_num_nodes=256
default_num_features=5

# signal
for corr in False True; do
    for signal in 1.0 1.5 2.0 2.5; do
        python experiment/experiment.py \
            --num_results 1000 \
            --num_worker 1 \
            --num_nodes $default_num_nodes \
            --num_features $default_num_features \
            --corr $corr \
            --signal $signal \
            --lower_thres $lower_thres \
            --upper_thres $(echo "scale=2; 1 - $lower_thres" | bc) \
            --test_class_1_only $test_class_1_only\
            --xai_type $xai_type

        if [ $? -ne 0 ]; then
            echo "An error occurred. The script will be terminated."
            exit 1
        fi
    done
done


# num nodes
for corr in False True; do
    for num_nodes in 32 64 128 256; do
        python experiment/experiment.py \
            --num_results 1000 \
            --num_worker 1 \
            --num_nodes $num_nodes \
            --num_features $default_num_features \
            --corr $corr \
            --signal 0.0 \
            --lower_thres $lower_thres \
            --upper_thres $(echo "scale=2; 1 - $lower_thres" | bc) \
            --test_class_1_only $test_class_1_only\
            --xai_type $xai_type

        if [ $? -ne 0 ]; then
            echo "An error occurred. The script will be terminated."
            exit 1
        fi
    done
done


# num features
for num_features in 5 10 15 20; do
    for corr in False True; do
        python experiment/experiment.py \
            --num_results 1000 \
            --num_worker 1 \
            --num_nodes $default_num_nodes \
            --num_features $num_features \
            --corr $corr \
            --signal 0.0 \
            --lower_thres $lower_thres \
            --upper_thres $(echo "scale=2; 1 - $lower_thres" | bc) \
            --test_class_1_only $test_class_1_only\
            --xai_type $xai_type
            
        if [ $? -ne 0 ]; then
            echo "An error occurred. The script will be terminated."
            exit 1
        fi
    done
done
