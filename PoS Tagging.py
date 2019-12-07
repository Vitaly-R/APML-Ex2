import pickle
import numpy as np
import matplotlib.pyplot as plt

START_STATE = '*STS*'
START_WORD = '*STW*'
END_STATE = '*ENDS*'
END_WORD = '*ENDW*'
RARE_WORD = '*RARE_WORD*'


class Baseline(object):
    """
    The baseline model.
    """

    def __init__(self, pos_tags, words):
        """
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        """
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.i2pos = {i: pos for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.emission_probabilities = None
        self.multinomial_probabilities = None

    def train(self, training_set):
        """
        Trains the baseline model on the given training data set.
        Assumes that the training set was pre-processed to handle rare words, and start/end of sentences.
        Also assumes that the sentences are constructed from the words in the list given to the constructor,
        and that the tags are from the list of pos tags given to the constructor.
        :param training_set: a numpy array of shape (num_examples, 2) in which each row is a list of pos tags,
        and a sentence corresponding to it as a lit of words of the same length.
        """
        sentences = training_set[:, 1]
        tag_lists = training_set[:, 0]

        # calculating the probability of each tag - Pr(y)
        tags_histogram = np.zeros(self.pos_size)
        for tag_list in tag_lists:
            for tag in tag_list:
                tags_histogram[self.pos2i[tag]] += 1
        self.multinomial_probabilities = tags_histogram / np.sum(tags_histogram)

        # calculating the emission probability of each word in the sentences - e(x,y)=Pr(X=x|Y=y)
        word_tag_table = np.zeros((self.words_size, self.pos_size))
        for i in range(sentences.shape[0]):
            for j in range(len(sentences[i])):
                word = sentences[i][j]
                tag = tag_lists[i][j]
                word_tag_table[self.word2i[word], self.pos2i[tag]] += 1  # this is the number of times the word 'word' got the tag 'tag'
        self.emission_probabilities = word_tag_table / tags_histogram

    def MAP(self, sentences):
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        """
        results = list()
        # in this naive model, the result for each word in a sentence is argmax(P(y)P(x|y))
        for sentence in sentences:
            result = list()
            for word in sentence:
                if word not in self.words:
                    p_x = self.emission_probabilities[self.word2i[RARE_WORD]]
                else:
                    p_x = self.emission_probabilities[self.word2i[word]]
                res = self.multinomial_probabilities * p_x  # calculating P(y)P(x|y) for each y
                tag = np.argmax(res)  # getting the tag index
                result.append(self.i2pos[tag])  # getting the tag itself
            results.append(result)
        return np.array(results)


class HMM(object):
    """
    The basic HMM_Model with multinomial transition functions.
    """

    def __init__(self, pos_tags, words):
        """
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        """

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.i2pos = {i: pos for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.transition_probabilities = None
        self.emission_probabilities = None

    def train(self, training_set):
        """
        Trains the HMM over the given training set.
        Assumes that all words in the set were given to the constructor when creating the model, and that all PoS tags were given as well.
        If the model was previously trained, it will re-train the model (erasing the probabilities learned thus far).
        :param training_set: A numpy.ndarray object with shape (training set size, 2), such that training_set[j] = [sentence, tags] where sentence and tags are list objects.
        """
        sentences = training_set[:, 1]
        tag_lists = training_set[:, 0]

        # calculating transition probabilities - t(i, j) = Pr(y_i|y_j) = #(y_j -> y_i) / sum_j'(#(y_j' -> y_i))
        transition_histogram = np.zeros((self.pos_size, self.pos_size))  # transition_histogram[i, j] = #(y_j -> y_i), meaning the sum of a row is: sum_j'(#(y_j' -> y_i))
        tags_histogram = np.zeros(self.pos_size)  # tags_histogram[j] = sum_x(#(x->j)) (number of appearances of tag j)
        for tag_list in tag_lists:
            tags_histogram[self.pos2i[tag_list[0]]] += 1
            for j in range(1, len(tag_list)):
                transition_histogram[self.pos2i[tag_list[j]], self.pos2i[tag_list[j-1]]] += 1
                tags_histogram[self.pos2i[tag_list[j]]] += 1
        probabilities = list()

        for transition_vec in transition_histogram.T:
            s = np.sum(transition_vec)
            if s > 0:
                probabilities.append(list(transition_vec / s))
            else:
                probabilities.append(list(transition_vec))
        self.transition_probabilities = np.array(probabilities).T

        # calculating the emission probability of each word - e(x, i) = Pr(x|y) = #(x->y)/ sum_x'(#(x'->y))
        word_tag_histogram = np.zeros((self.words_size, self.pos_size))  # word_tag_histogram[i, j] = #(i -> j), meaning that the sum of each column is sum_x'(#(x'->j)) = tags_histogram[j]
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                word_tag_histogram[self.word2i[sentences[i][j]], self.pos2i[tag_lists[i][j]]] += 1
        self.emission_probabilities = word_tag_histogram / tags_histogram

    def sample(self, n):
        """
        Sample n sequences of words from the HMM.
        :return: A list of word sequences, and their tags.
        """
        sequences = list()
        tags = list()
        for _ in range(n):
            words_sequence = list()
            tags_sequence = list()
            generated_state = START_STATE
            while True:
                generated_word = np.random.choice(self.words, p=self.emission_probabilities[:, self.pos2i[generated_state]])
                words_sequence.append(generated_word)
                tags_sequence.append(generated_state)
                if generated_state != END_STATE:
                    generated_state = np.random.choice(self.pos_tags, p=self.transition_probabilities[:, self.pos2i[generated_state]])
                else:
                    break
            sequences.append(words_sequence)
            tags.append(tags_sequence)
        return list(zip(sequences, tags))

    def viterbi(self, sentences):
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        """
        results = list()
        for sequence in sentences:
            ptable = np.zeros((len(sequence), self.pos_size, 2))
            for i in range(ptable.shape[0]):
                word = sequence[i]
                if word not in self.words:
                    word = RARE_WORD
                l_emission_probs = np.log(self.emission_probabilities[self.word2i[word]])
                if i == 0:
                    ptable[i, :, 0] = l_emission_probs
                else:
                    for j in range(ptable.shape[1]):
                        incoming = ptable[i-1, :, 0] + np.log(self.transition_probabilities[j, :]) + l_emission_probs[j]
                        ptable[i, j] = np.array([incoming.max(), incoming.argmax()])
            tags = list()
            itag = np.argmax(ptable[-1, :, 0])
            for i in range(ptable.shape[0] - 1, -1, -1):
                tags.insert(0, self.i2pos[itag])
                itag = int(ptable[i, itag, 1])
            results.append(tags)
        return results


class MEMM(object):
    """
    The base Maximum Entropy Markov Model with log-linear transition functions.
    """

    def __init__(self, pos_tags, words, phi_func, w_shape: [int, tuple] = None):
        """
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param phi_func: the feature mapping function, which accepts two PoS tags and a word, and returns a list of indices that have a "1" in the binary feature vector.
        """
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.i2pos = {i: pos for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.phi = phi_func
        self.w = np.zeros(self.pos_size ** 2 + self.pos_size * self.words_size) if w_shape is None else np.zeros(w_shape)

    def viterbi(self, sentences):
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of POS tag sequences.
        """
        results = list()
        for sequence in sentences:
            ptable = np.zeros((len(sequence), self.pos_size, 2))
            for i in range(ptable.shape[0]):
                word = sequence[i]
                if word not in self.words:
                    word = RARE_WORD
                for j in range(ptable.shape[1]):
                    if i == 0:
                        ptable[0, j, 0] = self.w[self.phi(self.i2pos[j], START_STATE, word, self)].sum()
                    else:
                        incoming = np.array([self.w[self.phi(self.i2pos[j], self.i2pos[k], word, self)].sum() for k in range(self.pos_size)]) + ptable[i-1, :, 0]
                        ptable[i, j] = np.array([incoming.max(), incoming.argmax()])
            tags = list()
            itag = np.argmax(ptable[-1, :, 0])
            for i in range(ptable.shape[0]-1, -1, -1):
                tags.insert(0, self.i2pos[itag])
                itag = int(ptable[i, itag, 1])
            results.append(tags)
        return results

    def perceptron(self, training_set, eta=0.1, epochs=1):
        """
        learn the weight vector of a log-linear model according to the training set.
        :param training_set: iterable sequence of sentences and their parts-of-speech.
        :param eta: the learning rate for the perceptron algorithm.
        :param epochs: the amount of times to go over the entire training data (default is 1).
        """
        print('-------- training model over', training_set.shape[0], 'sentences --------')
        factor = epochs * training_set.shape[0] + 1
        final_w = self.w / factor
        w = self.w
        for e in range(1, epochs + 1):
            print('----- epoch', e, '-----') if epochs > 1 else None
            i = 1
            for tags, sentence in training_set:
                predictions = self.viterbi([sentence])[0]

                step = np.zeros(w.shape)
                step[self.phi(tags[0], START_STATE, sentence[0], self)] += eta
                step[self.phi(predictions[0], START_STATE, sentence[0], self)] -= eta
                for j in range(1, len(sentence)):
                    step[self.phi(tags[j], tags[j - 1], sentence[j], self)] += eta
                    step[self.phi(predictions[j], predictions[j - 1], sentence[j], self)] -= eta

                w += step
                final_w += (w / factor)

                print(str((int(10000 * (i / training_set.shape[0]))) / 100) + '% complete') if i == 1 or not i % (training_set.shape[0] // 10) or i == (training_set.shape[0] - 1) else None
                i += 1
        self.w = final_w.copy()


def phi_1(tag, p_tag, word, model: MEMM):
    """
    Basic feature mapping function for a MEMM.
    Assumes that the size of the feature vector is (model.pos_size ** 2 + model.pos_size * model.words_size)
    :param tag: tag of the given word
    :param p_tag: ag of the previous word
    :param word: current word
    :param model: model for which the indices are created.
    :return: indices in the binary feature vector for which the value is 1.
    """
    return [model.pos2i[tag] * model.pos_size + model.pos2i[p_tag], model.pos_size ** 2 + model.pos_size * model.word2i[word] + model.pos2i[tag]]


def phi_2(tag, p_tag, word, model: MEMM):
    """
    Basic feature mapping function with basic additional features for a MEMM.
    Assumes that the size of the feature vector is (model.pos_size ** 2 + model.pos_size * model.words_size + 6)
    :param tag: tag of the given word
    :param p_tag: ag of the previous word
    :param word: current word
    :param model: model for which the indices are created.
    :return: indices in the binary feature vector for which the value is 1.
    """
    res = [model.pos2i[tag] * model.pos_size + model.pos2i[p_tag], model.pos_size ** 2 + model.pos_size * model.word2i[word] + model.pos2i[tag]]

    if word.endswith('ing') or word.endswith('ed') or word.endswith('ate') or word.endswith('ify'):
        res.append(model.w.shape[0] - 6)  # suggesting a verb
    if word.endswith('ion') or word.endswith('age') or word.endswith('nce') or word.endswith('dom') or word.endswith('er') or word.endswith('hood') or word.endswith('ism') or word.endswith('ship') \
            or word.endswith('ness') or word.endswith('ment'):
        res.append(model.w.shape[0] - 5)  # suggesting a noun
    if word.endswith('ful') or word.endswith('ble') or word.endswith('ive') or word.endswith('ese') or word.endswith('less') or word.endswith('ous') or word.endswith('ic'):
        res.append(model.w.shape[0] - 4)  # suggesting an adjective
    if len(word) <= 3:
        res.append(model.w.shape[0] - 3)  # suggesting a linking word (and, a, the...)
    if word[0].isupper():
        res.append(model.w.shape[0] - 2)  # suggesting beginning of a sentence, or a name
    if word.isdigit():
        res.append(model.w.shape[0] - 1)  # number

    return res


def compare_models(pos, words, training_ds, test_ds):
    """
    Preform an evaluation of different models for the PoS tagging task.
    :param pos: list of possible PoS tags.
    :param words: list of words in the training set
    :param training_ds: training set of shape (training set size, 2), in which training_set[j] = [sentence, tags] where sentence and tags are list objects.
    :param test_ds: test set of shape (test set size, 2) with a structure like the training set.
    """

    # Creating the models to evaluate.
    baseline_model = Baseline(pos, words)
    hmm_model = HMM(pos, words)
    memm_model_1 = MEMM(pos, words, phi_1)
    memm_model_2 = MEMM(pos, words, phi_2, (len(pos) ** 2 + len(words) * len(pos) + 6))

    # Splitting the set set to sentences and tags for evaluation.
    test_size = len(test_ds)  # To be able to test on a smaller number of sentences as necessary
    test_tags = test_ds[:test_size, 0]
    test_sentences = test_ds[:test_size, 1]

    # Defining parameters for evaluating Baseline and HMM
    baseline_accuracies = list()
    hmm_accuracies = list()
    factors = [20, 10, 5, 2, 1]
    x_axis = [100 // factor for factor in factors]

    # Evaluation of Baseline Model, and HMM
    for factor in factors:
        # Training the models.
        print('--- Training Baseline model and HMM over ', 100 / factor, '% of the training data ---', sep='')
        baseline_model.train(training_ds[: training_ds.shape[0] // factor])
        hmm_model.train(training_ds[: training_ds.shape[0] // factor])

        # Testing Baseline Model
        print('--- Testing Baseline model ---')
        baseline_predictions = baseline_model.MAP(test_sentences)
        correct = 0
        overall = 0
        for i in range(len(baseline_predictions)):
            for j in range(len(baseline_predictions[i])):
                overall += 1
                if baseline_predictions[i][j] == test_tags[i][j]:
                    correct += 1
        baseline_accuracies.append(100 * correct / overall)

        # Testing HMM
        print('--- Testing HMM ---')
        hmm_predictions = hmm_model.viterbi(test_sentences)
        correct = 0
        overall = 0
        for i in range(len(hmm_predictions)):
            for j in range(len(hmm_predictions[i])):
                overall += 1
                if hmm_predictions[i][j] == test_tags[i][j]:
                    correct += 1
        hmm_accuracies.append(100 * correct / overall)
        print()

    # Sampling example from the HMM
    print('--- Sampling example from the HMM ---')
    sample = hmm_model.sample(1)[0]
    print('Generated tag list', sample[1], '\nGenerated sentence', sample[0], '\n')

    # Evaluation of each MEMM over 10% of training data, and the entire test set (which is slightly larger)
    print('--- Training MEMM over 10% of the training data ---')
    print('(With no additional features)')
    memm_model_1.perceptron(training_ds[: training_ds.shape[0] // 10])
    print('--- Testing model ---')
    memm_1_predictions = memm_model_1.viterbi(test_sentences)
    correct = 0
    overall = 0
    for i in range(len(memm_1_predictions)):
        for j in range(len(memm_1_predictions[i])):
            overall += 1
            if memm_1_predictions[i][j] == test_tags[i][j]:
                correct += 1
    print('MEMM (With no additional features) accuracy: ', 100 * correct / overall, '%', sep='')

    print('--- Training MEMM over 10% of the training data ---')
    print('(With basic additional features)')
    memm_model_2.perceptron(training_ds[: training_ds.shape[0] // 10])
    print('--- Testing model ---')
    memm_2_predictions = memm_model_2.viterbi(test_sentences)
    correct = 0
    overall = 0
    for i in range(len(memm_2_predictions)):
        for j in range(len(memm_2_predictions[i])):
            overall += 1
            if memm_2_predictions[i][j] == test_tags[i][j]:
                correct += 1
    print('MEMM (With basic additional features) accuracy: ', 100 * correct / overall, '%', sep='')

    # Plotting the accuracies of the Baseline Model and HMM
    plt.figure()
    plt.title('Model accuracies')
    plt.plot(x_axis, baseline_accuracies, label='Baseline model accuracies')
    plt.plot(x_axis, hmm_accuracies, label='HMM accuracies')
    plt.xlabel('% of training data')
    plt.ylabel('accuracy (%)')
    plt.legend()
    # plt.show()  # Uncomment to show plot.


def load_data(test_set_fraction=0.1, rare_threshold=5):
    """
    Loading and pre-processing the data for the PoS tagging task.
    :param test_set_fraction: How much of the data should be used for testing, a number between 0 and 1.
    :param rare_threshold: Up to how many times should a word appear in the training set to be considered rare.
    :return:
    pos - a list of a possible PoS tags
    words - a list of words of the training set
    training_ds - a numpy.ndarray of shape (N, 2) (N - size of training set), in which training_ds[i] = [sentence (list object), tags (list object)]
    test_ds - a numpy.ndarray of shape (T, 2) (T - size of test set), with the same structure as training_ds.
    """
    # loading from files
    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    words.append(RARE_WORD)
    words.append(START_WORD)
    words.append(END_WORD)
    pos.append(START_STATE)
    pos.append(END_STATE)

    # adding start/end words, and start/end tags to each sentence:
    for (tag_list, sentence) in data:
        sentence.insert(0, START_WORD)
        sentence.append(END_WORD)
        tag_list.insert(0, START_STATE)
        tag_list.append(END_STATE)

    # splitting into training set and test set:
    data = np.array(data)
    inds = np.arange(data.shape[0])
    np.random.shuffle(inds)

    training_ds = data[int(test_set_fraction * inds.shape[0]):]
    test_ds = data[:int(test_set_fraction * inds.shape[0])]

    # finding rare words:
    word_count = {word: 0 for word in words}
    for (_, sentence) in training_ds:
        for word in sentence:
            word_count[word] += 1
    for (_, sentence) in training_ds:
        for i in range(len(sentence)):
            if sentence[i] not in word_count.keys():
                sentence[i] = RARE_WORD
            elif word_count[sentence[i]] <= rare_threshold:
                word_count[RARE_WORD] += word_count[sentence[i]]
                word_count.pop(sentence[i])
                sentence[i] = RARE_WORD
    word_count = {word: count for (word, count) in word_count.items() if count > 0}
    words = np.array(list(word_count.keys()))
    pos = np.array(pos)
    return pos, words, training_ds, test_ds


def main():
    pos, words, training_ds, test_ds = load_data()
    compare_models(pos, words, training_ds, test_ds)


if __name__ == '__main__':
    main()
