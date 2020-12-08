from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
import numpy as np

docs = ['Well done!', 'Good work', 'Great effort', 'Nice work', 'Excellent!', 'Good job', 'Very good', 'Nice', 'Good for you!',
        'That\'s really nice', 'Superb', 'That\'s the best ever', 'You did that very well', 'That\'s great!', 'You\'ve got it made',
        'Way to go!', 'Terrific', 'That\'s the way to do it!', 'That\'s not bad!', 'That\'s quite an improvement',
        'Couldn\'t have done it better myself', 'Good thinking', 'Marvelous',
        'That\'s bad', 'That\'s too bad', 'That\'s not good', 'You are failed', 'That\'s not nice', 'Terrible', 'Error', 'Bad',
        'Not nice', 'Worst']  # 23 + 10
#Representation for if a word is good or bad.
labels = np.concatenate((np.ones(23), np.zeros(10)))
print(labels)
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
x = token.texts_to_sequences(docs)
print(x)
# moderator max phrase length for the given doc.
max_phrase_length = 6
padded_x = pad_sequences(x, max_phrase_length)
print(padded_x)

word_size = len(token.word_index) + 1
model = Sequential()
model.add(Embedding(word_size, 8, input_length=max_phrase_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
model.fit(padded_x, labels, epochs=20)




