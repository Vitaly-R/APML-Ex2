import pickle
import numpy as np

START_STATE = '*STS*'
START_WORD = '*STW*'
END_STATE = '*ENDS*'
END_WORD = '*ENDW*'
RARE_WORD = '*RARE_WORD*'
EPSILON = 1e-10


def data_example(data_path='PoS_data.pickle',
                 words_path='all_words.pickle',
                 pos_path='all_PoS.pickle'):
    """
    An example function for loading and printing the Parts-of-Speech data for
    this exercise.
    Note that these do not contain the "rare" values and you will need to
    insert them yourself.

    :param data_path: the path of the PoS_data file.
    :param words_path: the path of the all_words file.
    :param pos_path: the path of the all_PoS file.
    """

    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    print("The number of sentences in the data set is: " + str(len(data)))
    print("\nThe tenth sentence in the data set, along with its PoS is:")
    print(data[10][1])
    print(data[10][0])

    print("\nThe number of words in the data set is: " + str(len(words)))
    print("The number of parts of speech in the data set is: " + str(len(pos)))

    print("one of the words is: " + words[34467])
    print("one of the parts of speech is: " + pos[17])

    print(pos)


class Baseline(object):
    """
    The baseline model.
    """

    def __init__(self, pos_tags, words, training_set):
        """
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
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
        self.train(training_set)

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
        # this is simply the number of times tag y appeared for word x divided by the total number of times that tag y appeared.
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
                res = self.multinomial_probabilities * p_x  # calculating P(y)*P(x|y) for each y
                tag = np.argmax(res)  # getting the tag index
                result.append(self.i2pos[tag])  # getting the tag itself
            results.append(result)
        return np.array(results)


def baseline_mle(training_set, model: Baseline):
    """
    a function for calculating the Maximum Likelihood estimation of the
    multinomial and emission probabilities for the baseline model.
    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial baseline model with the pos2i and word2i mappings among other things.
    :return: a mapping of the multinomial and emission probabilities. The multinomial probabilities are given
    as a numpy array with shape (|POS|, ), and the emission probabilities are given as a numpy array with shape (|words|, |POS|).
    """
    model.train(training_set)
    return model.multinomial_probabilities, model.emission_probabilities


class HMM(object):
    """
    The basic HMM_Model with multinomial transition functions.
    """

    def __init__(self, pos_tags, words, training_set):
        """
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
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
        self.train(training_set)

    def train(self, training_set):
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
        :return: A list of word sequences.
        """
        sequences = list()
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
        return sequences

    def __predict_sequence(self, sequence):
        """
        Given a sequence of words, predict its most likely PoS assignment.
        :param sequence: sequence to predict.
        :return: sequence of PoS tags.
        """
        # Creating a probability table representing a similar graph to the one learned in class.
        # Instead of simply holding the probabilities for the corresponding node in a layer, each cell [i, j] holds a list of sums of the maximal paths leading to the cell [i, j]
        # from the cell [i, k] in the previous layer in the graph. This way, the cell [i, j, k] holds the weight of the maximal path leading to the j'th node in he i'th layer from the k'th node in
        # the i-1'th layer.
        probability_table = np.zeros((len(sequence), self.pos_size, self.pos_size)) + EPSILON
        for i in range(len(sequence)):
            if sequence[i] not in self.words:
                emissions = self.emission_probabilities[self.word2i[RARE_WORD]]
            else:
                emissions = self.emission_probabilities[self.word2i[sequence[i]]]
            for j in range(self.pos_size):
                emission_p = emissions[j]
                if i == 0:
                    res = np.log(emission_p)
                    probability_table[i, j] = res
                else:
                    for k in range(self.pos_size):
                        probability_table[i, j, k] = probability_table[i-1, k, np.argmax(probability_table[i-1, k])] + np.log(self.transition_probabilities[j, k] * emission_p)
        # Extracting the list of tags is done as follows:
        # 1. We extract the cell with the highest value from the last layer. The list index is the index of the PoS tag, and the position within the list represents the index of the list in the
        # previous layer from which the maximal path proceeded. We insert the tag received to the list of tags.
        # 2. We iterate over the probability table in reverse order , each time we get the index of the list in the previous layer (which is the index of the highest value
        # in the current list), and insert the PoS tag corresponding to the index of the current list to the beginning of the list of tags.
        tags = list()
        (tag_index, prev_list_index) = np.unravel_index(np.argmax(probability_table[-1]), probability_table[-1].shape)
        tags.insert(0, self.i2pos[tag_index])
        i = len(sequence) - 2
        while 0 <= i:
            tag_index, prev_list_index = prev_list_index, np.argmax(probability_table[i, prev_list_index])
            tags.insert(0, self.i2pos[tag_index])
            i -= 1
        return tags

    def viterbi(self, sentences):
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        """
        results = list()
        print('Running HMM on', len(sentences), 'sentences')
        c = 1
        for sequence in sentences:
            if c == 1 or not c % 50:
                print('Starting round', c)
            results.append(self.__predict_sequence(sequence))
            c += 1
        return results


def hmm_mle(training_set, model: HMM):
    """
    a function for calculating the Maximum Likelihood estimation of the
    transition and emission probabilities for the standard multinomial HMM.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial HMM with the pos2i and word2i mappings among other things.
    :return: a mapping of the transition and emission probabilities. You may implement
            the probabilities in |PoS|x|PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """

    model.train(training_set)
    return model.transition_probabilities, model.emission_probabilities


class MEMM(object):
    """
    The base Maximum Entropy Markov Model with log-linear transition functions.
    """

    def __init__(self, pos_tags, words, training_set):
        """
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
        :param training_set: A training set of sequences of POS-tags and words.
        """
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.i2pos = {i: pos for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.feature_vector_shape = (self.pos_size * self.pos_size + self.pos_size * self.words_size, )
        self.w = np.zeros(self.feature_vector_shape)
        self.w = perceptron(training_set, self, self.w)

    def phi(self, tag, prev_tag, word):
        curr_ind = self.pos2i[tag]
        prev_ind = self.pos2i[prev_tag]
        word_ind = self.word2i[word]

        ind1 = prev_ind + self.pos_size * curr_ind
        ind2 = (self.pos_size * self.pos_size) + (self.pos_size * word_ind + curr_ind)
        return np.array([ind1, ind2])

    def __predict_sequence(self, sequence, w):
        """
        Given a sequence of words, predict its most likely PoS assignment.
        :param sequence: sequence to predict.
        :param w: weight vector according to which to predict.
        :return: sequence of PoS tags.
        """
        # Creating a probability table representing a similar graph to the one learned in class.
        # Instead of simply holding the probabilities for the corresponding node in a layer, each cell [i, j] holds a list of sums of the maximal paths leading to the cell [i, j]
        # from the cell [i, k] in the previous layer in the graph. This way, the cell [i, j, k] holds the weight of the maximal path leading to the j'th node in he i'th layer from the k'th node in
        # the i-1'th layer.
        probability_table = np.zeros((len(sequence), self.pos_size, self.pos_size))
        for i in range(len(sequence)):
            word = sequence[i]
            if word not in self.words:
                word = RARE_WORD
            for k in range(self.pos_size):
                if i == 0:
                    probability_table[i, k, :] = 0
                else:
                    z = np.log(np.sum(np.array([np.exp(np.sum(w[self.phi(self.i2pos[j], self.i2pos[k], word)])) for j in range(self.pos_size)])))
                    m = probability_table[i-1, k, np.argmax(probability_table[i-1, k])]
                    for j in range(self.pos_size):
                        probability_table[i, j, k] = m + np.sum(w[self.phi(self.i2pos[j], self.i2pos[k], word)]) - z
        # Extracting the list of tags is done as follows:
        # 1. We extract the cell with the highest value from the last layer. The list index is the index of the PoS tag, and the position within the list represents the index of the list in the
        # previous layer from which the maximal path proceeded. We insert the tag received to the list of tags.
        # 2. We iterate over the probability table in reverse order , each time we get the index of the list in the previous layer (which is the index of the highest value
        # in the current list), and insert the PoS tag corresponding to the index of the current list to the beginning of the list of tags.
        tags = list()
        (tag_index, prev_list_index) = np.unravel_index(np.argmax(probability_table[-1]), probability_table[-1].shape)
        tags.insert(0, self.i2pos[tag_index])
        i = len(sequence) - 2
        while 0 <= i:
            tag_index, prev_list_index = prev_list_index, np.argmax(probability_table[i, prev_list_index])
            tags.insert(0, self.i2pos[tag_index])
            i -= 1
        return tags

    def viterbi(self, sentences, w):
        """
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        """
        results = list()
        for sequence in sentences:
            results.append(self.__predict_sequence(sequence, w))
        return results


def perceptron(training_set, initial_model, w0, eta=0.1, epochs=2):
    """
    learn the weight vector of a log-linear model according to the training set.
    :param training_set: iterable sequence of sentences and their parts-of-speech.
    :param initial_model: an initial MEMM object, containing among other things
            the phi feature mapping function.
    :param w0: an initial weights vector.
    :param eta: the learning rate for the perceptron algorithm.
    :param epochs: the amount of times to go over the entire training data (default is 1).
    :return: w, the learned weights vector for the MEMM.
    """
    w = [w0]
    for e in range(1, epochs + 1):
        print('epoch', e)
        for i in range(1, len(training_set) + 1):
            if i == 1 or not i % (len(training_set) // 10):
                print('round', i)
            tags, sentence = training_set[i-1]
            predictions = initial_model.viterbi([sentence], w[i-1])[0]
            w.append(w[i-1] + eta * np.sum([initial_model.phi(tags[j], tags[j-1], sentence[j]) - initial_model.phi(predictions[j], predictions[j-1], sentence[j]) for j in range(len(tags))]))
    return np.average(np.array(w), axis=0)


def load_data(test_set_fraction=0.1, rare_threshold=5):
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


def run_baseline(pos, words, training_ds, test_ds):
    bl_model = Baseline(pos, words, training_ds)
    test_sentences = test_ds[:, 1]
    test_tags = test_ds[:, 0]
    test_predictions = bl_model.MAP(test_sentences)
    correct_tags = 0
    overall_tags = 0
    for i in range(len(test_tags)):
        for j in range(len(test_tags[i])):
            overall_tags += 1
            if test_predictions[i][j] == test_tags[i][j]:
                correct_tags += 1
    print('Baseline model accuracy: ', 100 * correct_tags / overall_tags, '%', sep='')


def run_hmm(pos, words, training_ds, test_ds):
    hmm_model = HMM(pos, words, training_ds)
    test_sentences = test_ds[:, 1]
    test_tags = test_ds[:, 0]
    predictions = hmm_model.viterbi(test_sentences[:500])
    correct_tags = 0
    overall_tags = 0
    for i in range(len(predictions)):
        for t in range(len(predictions[i])):
            overall_tags += 1
            if predictions[i][t] == test_tags[i][t]:
                correct_tags += 1
    print('HMM accuracy: ', 100 * correct_tags / overall_tags, '%', sep='')


def run_memm(pos, words, training_ds, test_ds):
    memm_model = MEMM(pos, words, training_ds[: 1000])
    test_sentences = test_ds[:, 1]
    predictions = memm_model.viterbi(test_sentences[:100], memm_model.w)
    test_tags = test_ds[:, 0]
    correct_tags = 0
    overall_tags = 0
    for i in range(len(predictions)):
        for t in range(len(predictions[i])):
            overall_tags += 1
            if predictions[i][t] == test_tags[i][t]:
                correct_tags += 1
    print('MEMM accuracy: ', 100 * correct_tags / overall_tags, '%', sep='')


def main():
    pos, words, training_ds, test_ds = load_data()
    run_memm(pos, words, training_ds, test_ds)


if __name__ == '__main__':
    main()
