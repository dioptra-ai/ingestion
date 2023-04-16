from numpy import unique

def segmentation_class_distribution(segmentation_class_mask, class_names = None):
    values, counts = unique(segmentation_class_mask, return_counts=True)
    values = values.astype(int).tolist()
    counts = counts.astype(int).tolist()
    class_names = class_names or [str(v) for v in range(max(values) + 1)]
    num_classes = len(class_names)
    distribution = {}
    for value, count in zip(values, counts):
        distribution[class_names[value] if value < num_classes else str(value)] = count
    
    return distribution
