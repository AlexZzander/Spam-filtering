import math
import numpy as np


class SpamClassifier:
    def __init__(self, training_data):
        """
        :param training_data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                     the first column contains the binary response (coded as 0s and 1s).
        :param log_class_priors: a numpy array of length 2.
        :param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
            theta[j, i] corresponds to the logarithm of the probability of feature i appearing
            in a sample belonging to class j.
        """
        self.training_data = training_data
        self._log_class_priors = None
        self._log_class_conditional_likelihoods = None

    def estimate_log_class_priors(self):
        """
        :return log_class_priors: a numpy array of length two
        """
        spam_number = self.training_data[:, 0].sum()
        ham_number = self.training_data.shape[0] - spam_number
        spam = math.log(spam_number / self.training_data.shape[0])
        ham = math.log(ham_number / self.training_data.shape[0])
        self._log_class_priors = np.array([spam, ham])

    def estimate_log_class_conditional_likelihoods(self, alpha=1.0):
        """
        :param alpha:
        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

        :return theta:
            a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging
            to class j.
        """
        class_spam = []
        class_ham = []
        for i in self.training_data:
            if i[0] == 1:
                class_spam.append(i[1:])
            else:
                class_ham.append(i[1:])
        spam_words = np.array(class_spam)
        ham_words = np.array(class_ham)

        theta_spam_sum = spam_words.sum(axis=0)
        theta_spam = self.calculate_theta_for_class(alpha, theta_spam_sum)

        theta_ham_sum = ham_words.sum(axis=0)
        theta_ham = self.calculate_theta_for_class(alpha, theta_ham_sum)

        self._log_class_conditional_likelihoods = np.vstack((theta_spam, theta_ham))

    @staticmethod
    def calculate_theta_for_class(alpha, class_theta):
        total_keyword = class_theta.sum()
        words_count = class_theta.shape[0]
        for word_index in range(words_count):
            class_theta[word_index] = math.log(
                (class_theta[word_index] + alpha) / (total_keyword + words_count * alpha))

        return class_theta

    def train(self):
        self.estimate_log_class_priors()
        self.estimate_log_class_conditional_likelihoods()

    def predict(self, new_data):
        """
        :param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
        :return class_predictions: a numpy array containing the class predictions for each row
            of new_data.
        """
        class_predictions = []
        for data_row in new_data:
            data_row_c0 = data_row * self._log_class_conditional_likelihoods[0]
            data_row_c1 = data_row * self._log_class_conditional_likelihoods[1]
            sum_c0 = self._log_class_priors[0] + data_row_c0.sum()
            sum_c1 = self._log_class_priors[1] + data_row_c1.sum()
            class_predictions.append(np.array([sum_c1, sum_c0]).argmax())

        return np.array(class_predictions)


def create_classifier(training_data):
    res_classifier = SpamClassifier(training_data)
    res_classifier.train()
    return res_classifier


if __name__ == '__main__':
    training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
    classifier = create_classifier(training_spam)
    testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
    test_data = testing_spam[:, 1:]
    test_labels = testing_spam[:, 0]

    predictions = classifier.predict(test_data)
    accuracy = np.count_nonzero(predictions == test_labels) / test_labels.shape[0]
    print(f"Accuracy on test data is: {accuracy}")
