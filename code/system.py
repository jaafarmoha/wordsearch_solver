"""Word Search Classification System.

Solution to the COM2004/3004 assignment.
Authour: Jaafar Mohammed
Date: Dec 2025

version: v1.0
"""

from typing import List

import numpy as np
import scipy.linalg
from utils import utils
from utils.utils import Puzzle

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20

def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)

def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    Takes the raw feature vectors and reduces them down to the required number of
    dimensions using PCA.

    The model is passed as an argument so that you can pass information from the 
    training stage.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    if "reduction_matrix" not in model:
        # Compute and store mean
        data_mean = np.mean(data, axis=0)
        model["mean_vector"] = data_mean.tolist()
        
        # Centring data
        data_c = data - data_mean
        
        # Compute covariance matrix
        covx = np.cov(data_c, rowvar=0)
        N = covx.shape[0]
        
       # Eigenvectors and eigenvalues
        w, v = scipy.linalg.eigh(covx, subset_by_index=(N - N_DIMENSIONS, N - 1))
        v = np.fliplr(v)  
        
        model["reduction_matrix"] = v.tolist()
    else:
        # Using stored mean and reduction matrix
        data_mean = np.array(model["mean_vector"])
        v = np.array(model["reduction_matrix"])
        data_c = data - data_mean 
    
    # Apply dimensionality reduction
    reduced_data = data_c @ v
    return reduced_data

def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Stores the labels and the dimensionally reduced trainingvectors. These needed to 
    store if using a non-parametric classifier such as a nearest neighbour or k-nearest 
    neighbour classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    model = {}
    model["labels_train"] = labels_train.tolist()
    # Store reduced training feature vectors
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model

def classify_squares(fvectors_test: np.ndarray, model: dict, K = 5) -> List[str]:
    """ KNN classifier using euclidean distance

    A list of unlabelled feature vectors are passed and the model parameters learn during 
    the training stage. This function classifies each feature vector and returns a list of labels.

    Args:
        fvectors_test (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.
        K (int): number of nearest neighbors to consider

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """
    fvectors_train = np.array(model["fvectors_train"])
    train_labels = np.array(model["labels_train"])
    predicted_labels = []

    for test_vector in fvectors_test:
        # Calculate Euclidean distances to all training vectors
        distances = np.sqrt(np.sum((fvectors_train - test_vector) ** 2, axis=1))
        
        # Get indices of K nearest neighbors using argsort
        nearest_indices = np.argsort(distances)[:K]
        
        # Get the labels of the K nearest neighbors
        nearest_labels = train_labels[nearest_indices]
        
        # Find the most common label
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predicted_labels.append(predicted_label)
    
    return predicted_labels

def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """ Function to solve word seach puzzle

    The function searches for the words in the grid of classified letter labels.
    It is passed the letter labels as a 2-D array and a list of words to search for.
    This will return a position for each word. The word position should be
    represented as tuples of the form (start_row, start_col, end_row, end_col).

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """

    # All 8 directions
    DIRECTIONS = [
        (0, 1),    # Right
        (1, 0),    # Down
        (1, 1),    # Down and Right
        (1, -1),   # Down and Left
        (0, -1),   # Left
        (-1, 0),   # Up
        (-1, -1),  # Up and Left
        (-1, 1),   # Up and Right
    ]
    
    rows, cols = labels.shape
    result = []
    
    for word in words:
        word = word.upper() 
        word_len = len(word)
        best_score = -1
        final_position = (0, 0, 0, 0)
        
        
        # Search through each cell in the grid
        for i in range(rows):
            for j in range(cols):
                for di, dj in DIRECTIONS:

                    # Find end position
                    end_i = i + (word_len - 1) * di
                    end_j = j + (word_len - 1) * dj
                    
                    # Check if word fits in this direction
                    if not(0 <= end_i < rows and 0 <= end_j < cols):
                        continue
                    
                    # Calculates a score for each matching letter
                    score = 0
                    for k in range(word_len):
                        current_char = labels[i + k*di, j + k*dj]
                        if k < word_len and current_char == word[k]:
                            score += 1
                    
                    # Update to best match
                    if score > best_score:
                        best_score = score
                        final_position = (i, j, end_i, end_j)
                
        result.append(final_position)
    
    return result
