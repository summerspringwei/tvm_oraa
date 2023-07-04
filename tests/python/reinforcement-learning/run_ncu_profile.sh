
# Collect all metrics
ncu --target-processes all \
    --clock-control none \
    --set full \
    -f -o feature_extractor_test-ncu --csv \
    python3 load_database_and_train_cost_model.py

# Export the ncu profile to csv file
ncu -i feature_extractor_test-ncu.ncu-rep --csv --page raw > feature_extractor_test-ncu.csv

