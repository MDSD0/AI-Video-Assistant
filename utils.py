# Placeholder for utility functions like metrics analysis
def save_metrics(metrics: list, filename: str = "metrics.json"):
    """Save metrics to a file for analysis."""
    import json
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)