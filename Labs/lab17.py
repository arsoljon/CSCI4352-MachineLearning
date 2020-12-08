from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)
text = 'IN THE LATE SUMMER of that year we lived in a house in a village that looked across the river and the ploain to the mountains.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)