DISEASE_CONFIGS = {
    "diabetes": {
        "data_url": "data/diabetes.csv",
        "target_column": "Outcome",
        "positive_class": 1,
        "test_size": 0.2
    },
    "heart_disease": {
        "data_url": "data/heart.csv",
        "target_column": "target",
        "positive_class": 1,
        "test_size": 0.2
    },
    "cancer": {
        "data_url": "data/cancer.csv",
        "target_column": "diagnosis",
        "positive_class": "M",
        "test_size": 0.2
    },
    "alzheimers": {
        "data_url": "data/alzheimers.csv",
        "target_column": "Group",
        "positive_class": "Demented",
        "test_size": 0.2
    },
    # Add configurations for other diseases as needed
}