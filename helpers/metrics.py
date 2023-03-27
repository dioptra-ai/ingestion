from numpy import unique

def segmentation_distribution(segmentation_class_mask, class_names = None):
    values, counts = unique(segmentation_class_mask, return_counts=True)
    values = values.astype(int).tolist()
    counts = counts.astype(int).tolist()
    class_names = class_names or [str(v) for v in range(max(values) + 1)]
    distribution = {}
    for value, count in zip(values, counts):
        distribution[class_names[value]] = count
    
    return distribution
