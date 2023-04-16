from helpers.eventprocessor.utils import compute_argmax, compute_entropy

def process_confidences(confidences, class_names=None):
    
    if confidences is None:
        
        return {
            'metrics': None,
            'confidence': None,
            'class_name': None
        }
    else:
        max_index = compute_argmax(confidences)

        return {
            'metrics': {
                'entropy': compute_entropy(confidences)
            },
            'confidence': confidences[max_index],
            'class_name': class_names[max_index] if class_names else None
        }
