import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores

        n_features = self.X.shape[1]
        logN = np.log(self.X.shape[0])

        best_n_components = self.n_constant
        best_score = float('inf')
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_components)
            try:
                logL = model.score(self.X, self.lengths)
            except (ValueError, AttributeError):
                logL = -1e9
            n_params = n_components * (n_components - 1) + 2 * n_features * n_components
            bic = -2 * logL + n_params * logN
            if bic < best_score:
                best_score = bic
                best_n_components = n_components
        return self.base_model(best_n_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on DIC scores
        scores = {}

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(n_components)
            scores[n_components] = {}
            for word in self.words:
                try:
                    scores[n_components][word] = model.score(*self.hwords[word])
                except (ValueError, AttributeError):
                    scores[n_components][word] = -1e9

        best_n_components = self.n_constant
        best_score = float('-inf')        
        for n_components in scores.keys():
            dic = scores[n_components][self.this_word] - np.mean([scores[n_components][word] \
                                    for word in self.words if word != self.this_word])
            if dic > best_score:
                best_score = dic
                best_n_components = n_components
        return self.base_model(best_n_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV
        
        n_splits = 3
        best_n_components = self.n_constant
        best_score = float('-inf')
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            sum_logL = 0            
            if len(self.sequences) < n_splits: 
                break
            split_method = KFold(n_splits)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                model = GaussianHMM(n_components=n_components, n_iter=1000)
                model.fit(*combine_sequences(cv_train_idx, self.sequences))
                try:
                    logL = model.score(*combine_sequences(cv_test_idx, self.sequences))
                except ValueError:
                    logL = -1e9
                sum_logL += logL
            # print(f'Score for n_components={n_components}: {sum_logL / n_splits}')
            if sum_logL / n_splits > best_score:
                best_n_components = n_components
                best_score = sum_logL / n_splits
        # print(f'Best Score for {self.this_word}: {best_score}')
        return self.base_model(best_n_components)