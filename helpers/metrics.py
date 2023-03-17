from numpy import unique, ravel

def segmentation_distribution(segmentation_class_mask, class_names = None):
    values, counts = unique(ravel(segmentation_class_mask), return_counts=True)
    class_names = class_names or [str(i) for i in range(len(values))]
    distribution = {}
    for value, count in zip(values, counts):
        distribution[class_names[value]] = int(count)
    
    return distribution
