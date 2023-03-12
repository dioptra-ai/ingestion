import torch
from eventprocessor.utils import (
    process_logits
)
import numpy as np

# main function
if __name__ == '__main__':
    eps = 1e-4
    # create a 1d array of logits with random numbers of shape [num_classes]
    logits = np.random.rand(3).tolist()
    confidences, _, _, _ = process_logits(logits)
    # calculate confidences using pytorch
    logits = torch.tensor(logits)
    # softmax over the first dimension
    probabilities = torch.clamp(torch.nn.functional.softmax(logits, dim=0), min=eps)
    # calculate the confidence of the argmax class
    test_confidences = probabilities.tolist()
    print('confidences of 1D: ', all([abs(confidences[i] == test_confidences[i]) < eps] for i in range(0, len(confidences))))
   

    # create a 2d array of logits with random numbers of shape [num_classes, num_inferences]
    logits = np.random.rand(3, 2).tolist()
    confidences, _ ,_, _ = process_logits(logits)
    # calculate confidences using pytorch
    logits = torch.tensor(logits)
    # softmax over the first dimension
    probabilities = torch.clamp(torch.nn.functional.softmax(logits, dim=0), min=eps)
    # calculate the confidence of the argmax class
    test_confidences = probabilities.tolist()
    print('confidences of 2D: ', all([abs(confidences[i] == test_confidences[i]) < eps] for i in range(0, len(confidences))))

    # create a 3D array of logits with random numbers of shape [num_classes, height, width]
    logits = np.random.rand(3, 2, 2).tolist()
    confidences, segmentation_class_mask, m_entropy, _ = process_logits(logits)
    # calculate the confidence, metrics and segmentation mask without using the process_logits function and instead using pytorch
    # begin with calculation a probability mask
    logits = torch.tensor(logits)
    # softmax over the first dimension
    probabilities = torch.clamp(torch.nn.functional.softmax(logits, dim=0), min=eps)
    # print('probabilities')
    # print(probabilities)
    # calculate the entropy for each class
    entropy = torch.sum(torch.special.entr(probabilities), dim=(1,2))

    # print('entropy')
    # print(entropy)
    # print(entropy2)
    # entropy = torch.sum(-probabilities * torch.log2(probabilities), dim=(1,2))
    # calculate the mean entropy
    entropy = torch.mean(entropy)
    # print('mean entropies')
    # print(entropy)
    # print(metrics['entropy'])

    # calculated the class mask
    class_mask = torch.argmax(logits, dim=0).tolist()
    # calculate the confidence for each class in the class mask
    test_confidences = [0 for _ in range(0, len(logits))]
    for i in range(0, len(logits)):
        if any(i in mask for mask in class_mask):
            # find the pixels in the mask that equal i
            # compute the mean of the probabilities of those pixels
            # assign the confidence of that class to that mean
            test_confidences[i] = torch.mean(torch.tensor([probabilities[i][j][k] for j in range(0, len(logits[0])) for k in range(0, len(logits[0][0])) if class_mask[j][k] == i]))
    print('entropy calculation is correct: ', abs(entropy -  m_entropy) < eps)
    print('segmentation_class_mask calculation is correct: ', class_mask == segmentation_class_mask)
    print('confidences calculation is correct: ', all([abs(confidences[i] - test_confidences[i]) < eps for i in range(0, len(confidences))]))


    # create a 4D array of logits with random numbers of shape [num_inferences, num_classes, height, width]
    logits = np.random.rand(5, 4, 2, 2).tolist()
    confidences, segmentation_class_mask, m_entropy, m_variance = process_logits(logits)
    # calculate the confidence, metrics and segmentation mask without using the process_logits function and instead using pytorch
    # begin with calculation a probability mask
    logits = torch.tensor(logits)
    # softmax over the first dimension
    probabilities = torch.clamp(torch.nn.functional.softmax(logits, dim=1), min=eps)
    # calculate the mean probabilities for each class for each inference over the image
    probability_means = torch.mean(probabilities, dim=(2,3))
    # print(probability_means)
    variance = torch.mean(torch.var(probability_means, dim=0))
    # print(torch.var(probability_means, dim=0))
    probabilities = torch.mean(probabilities, dim=0)
    entropy = torch.sum(torch.special.entr(probabilities), dim=(1,2))
    entropy = torch.mean(entropy)
    class_mask = torch.argmax(probabilities, dim=0).tolist()
    # calculate the confidence for each class in the class mask
    test_confidences = [0 for _ in range(0, len(logits[0]))]
    for i in range(0, len(logits[0])):
        if any(i in mask for mask in class_mask):
            # find the pixels in the mask that equal i
            # compute the mean of the probabilities of those pixels
            # assign the confidence of that class to that mean
            test_confidences[i] = torch.mean(torch.tensor([probabilities[i][j][k] for j in range(0, len(logits[0][0])) for k in range(0, len(logits[0][0][0])) if class_mask[j][k] == i]))
    print('variance calculation is correct: ', abs(m_variance - variance) < eps)
    print('entropy calculation is correct: ', abs(entropy - m_entropy) < eps)
    print('segmentation_class_mask calculation is correct: ', class_mask == segmentation_class_mask)
    print('confidences calculation is correct: ', all([abs(confidences[i] - test_confidences[i]) < eps for i in range(0, len(confidences))]))


