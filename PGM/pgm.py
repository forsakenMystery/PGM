import nltk
from sklearn_crfsuite import metrics
from nltk.tag.util import untag
from sklearn_crfsuite import CRF

# nltk.download('treebank')

tagged_sentences = nltk.corpus.treebank.tagged_sents()

print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))


def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        X.append([features(untag(tagged), index) for index in range(len(tagged))])
        y.append([tag for _, tag in tagged])

    return X, y


X_train, y_train = transform_to_dataset(training_sentences)
X_test, y_test = transform_to_dataset(test_sentences)

print(len(X_train))
print(len(X_test))
print(X_train[0])
print(y_train[0])


model = CRF()
model.fit(X_train, y_train)

sentence = ['I', 'am', 'Bob', '!']


def pos_tag(sentence):
    sentence_features = [features(sentence, index) for index in range(len(sentence))]
    return list(zip(sentence, model.predict([sentence_features])[0]))


print(pos_tag(sentence))  # [('I', 'PRP'), ('am', 'VBP'), ('Bob', 'NNP'), ('!', '.')]


y_pred = model.predict(X_test)
print(metrics.flat_accuracy_score(y_test, y_pred))


import pystruct.models as crf
from pystruct import learners
from pystruct.utils import SaveLogger
model = crf.EdgeFeatureGraphCRF(inference_method='qpbo')
ssvm = learners.NSlackSSVM(
    model, verbose=2, C=0.01, max_iter=100000, n_jobs=-1,
    tol=0.0001, show_loss_every=5,
    logger=SaveLogger(experiment_name + ".pickle", save_every=100),
    inactive_threshold=1e-3, inactive_window=10, batch_size=100)
ssvm.fit(X_train, y_train)
y_pred = ssvm.predict(X_test)
print(metrics.flat_accuracy_score(y_test, y_pred))
