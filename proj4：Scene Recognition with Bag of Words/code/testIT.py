import numpy as np
import os

from helpers import get_image_paths

if __name__ == "__main__":
    from student import get_tiny_images, build_vocabulary, get_bags_of_words, \
        svm_classify, nearest_neighbor_classify
    from create_results_webpage import create_results_webpage

    # FEATURE = 'tiny image'
    # FEATURE = 'bag of words'
    FEATURE = 'bag of words'

    # CLASSIFIER = 'nearest neighbor'
    # CLASSIFIER = 'support vector machine'
    CLASSIFIER = 'nearest neighbor'

    # This is the path the script will look at to load images from.
    data_path = '../data/'

    # This is the list of categories / directories to use. The categories are
    # somewhat sorted by similarity so that the confusion matrix looks more
    # structured (indoor and then urban and then rural).
    categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
                  'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
                  'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

    # This list of shortened category names is used later for visualization.
    abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                       'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For']

    # Number of training examples per category to use. Max is 100. For
    # simplicity, we assume this is the number of test cases per category as
    # well.
    num_train_per_cat = 100

    # This function returns string arrays containing the file path for each train
    # and test image, as well as string arrays with the label of each train and
    # test image. By default all four of these arrays will be 1500x1 where each
    # entry is a string.
    print('Getting paths and labels for all train and test data.')
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(data_path, categories, num_train_per_cat)
    #   train_image_paths  1500x1   list
    #   test_image_paths   1500x1   list
    #   train_labels       1500x1   list
    #   test_labels        1500x1   list

    print('Using %s representation for images.' % FEATURE)

    if FEATURE.lower() == 'tiny image':
        print('Loading tiny images...')
        # YOU CODE get_tiny_images (see student.py)
        train_image_feats = get_tiny_images(train_image_paths)
        test_image_feats = get_tiny_images(test_image_paths)
        print('Tiny images loaded.')

    elif FEATURE.lower() == 'bag of words':
        # Because building the vocabulary takes a long time, we save the generated
        # vocab to a file and re-load it each time to make testing faster. If
        # you need to re-generate the vocab (for example if you change its size
        # or the length of your feature vectors), simply delete the vocab.npy
        # file and re-run main.py

        if not os.path.isfile('vocab.npy'):
            print('No existing visual word vocabulary found. Computing one from training images.')
            # Larger values will work better (to a point), but are slower to compute
            vocab_size = 200

            # YOU CODE build_vocabulary (see student.py)
            vocab = build_vocabulary(train_image_paths, vocab_size)
            np.save('vocab.npy', vocab)

        # YOU CODE get_bags_of_words.m (see student.py)
        train_image_feats = get_bags_of_words(train_image_paths)

        # You may want to write out train_image_features here as a *.npy and
        # load it up later if you want to just test your classifiers without
        # re-computing features
        test_image_feats = get_bags_of_words(test_image_paths)
        # Same goes here for test image features.

    elif FEATURE.lower() == 'placeholder':
        train_image_feats = []
        test_image_feats = []
    else:
        raise ValueError('Unknown feature type!')

    print('Using %s classifier to predict test set categories.' % CLASSIFIER)

    if CLASSIFIER.lower() == 'nearest neighbor':
        # YOU CODE nearest_neighbor_classify (see student.py)
        predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)

    elif CLASSIFIER.lower() == 'support vector machine':
        # YOU CODE svm_classify (see student.py)
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

    elif CLASSIFIER.lower() == 'placeholder':
        # The placeholder classifier simply predicts a random category for every test case
        random_permutation = np.random.permutation(len(test_labels))
        predicted_categories = [test_labels[i] for i in random_permutation]

    else:
        raise ValueError('Unknown classifier type')

    create_results_webpage(train_image_paths, \
                           test_image_paths, \
                           train_labels, \
                           test_labels, \
                           categories, \
                           abbr_categories, \
                           predicted_categories)