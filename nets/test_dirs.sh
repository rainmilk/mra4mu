# Directory for saving machine unlearning results
BASE_SAVE_DATA_DIR="/nvme/szh/data/3ai/lips/saved_data"
LIPS_SAVE_DATA_DIR="/nvme/szh/data/3ai/lips/saved_data/lipnet"

if [ ! -d "$BASE_SAVE_DATA_DIR" ]; then
    echo "Directory for saving MU data directory not found: $BASE_SAVE_DATA_DIR"
    exit 1
fi

if [ ! -d "$LIPS_SAVE_DATA_DIR" ]; then
    echo "Directory for saving lipschitz data directory not found: $LIPS_SAVE_DATA_DIR"
    exit 1
fi

# Directory for saving the training outputs like Models and Logs
BASE_OUTPUT_DIR="/nvme/szh/data/3ai/lips/outputs"

if [ ! -d "$BASE_OUTPUT_DIR" ]; then
    echo "Directory for saving outputs not found: $BASE_OUTPUT_DIR"
    exit 1
fi

# Directory for saving the training outputs
LIPS_OUTPUT_DIR="/nvme/szh/data/3ai/lips/outputs/lipnet/resnet18"

if [ ! -d "$LIPS_OUTPUT_DIR" ]; then
    echo "Directory for saving lipschitz models not found: $LIPS_OUTPUT_DIR"
    exit 1
fi

