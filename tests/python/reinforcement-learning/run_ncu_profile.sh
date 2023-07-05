ncu --target-processes all --clock-control none --set full -f -o bert_base_1_128_workload_0-ncu python3 load_database_and_train_cost_model.py

# Collect all metrics
ncu --target-processes all \
    --clock-control none \
    --set full \
    -f -o bert_base_1_128_workload_0-ncu --csv \
    python3 load_database_and_train_cost_model.py

# Export the ncu profile to csv file
# ncu -i feature_extractor_test-ncu.ncu-rep --csv --page raw > feature_extractor_test-ncu.csv
# ncu -i bert_base_1_128_workload_3-ncu.ncu-rep --csv --page raw > bert_base_1_128_workload_3-ncu.csv
ncu --target-processes all --clock-control none --set full  --csv -o bert_base_1_128_workload_0-ncu python3 load_database_and_train_cost_model.py
