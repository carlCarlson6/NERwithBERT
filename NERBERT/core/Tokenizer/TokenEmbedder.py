from keras.preprocessing.sequence import pad_sequences


class TokenEmbedder:

    def SetTokenEmbedding(self, Tokenizer, TagToIndex, TokenizedText: list, TokenizedLabels: list, MaxLength: int):
        inputIds = self.__GetInputIds(Tokenizer, TokenizedText, MaxLength)
        tags = self.__GetTags(TagToIndex, TokenizedLabels, MaxLength)
        return inputIds, tags

    def __GetInputIds(self, Tokenizer, TokenizedText: list, MaxLength: int, DTpye='long', Truncating='post', Padding='post'):
        inputIds = pad_sequences(
            [Tokenizer.convert_tokens_to_dis(text) for text in TokenizedText],
            maxlen=MaxLength, dtype=DTpye,
            truncating=Truncating, padding=Padding
        )
        return inputIds

    def __GetTags(self, TagToIndex, TokenizedLabels: list, MaxLength: int, DTpye='long', Truncating='post', Padding='post'):
        tags = pad_sequences(
            [[TagToIndex.get(lbl) for lbl in label] for label in TokenizedLabels],
            maxlen = MaxLength, value = TagToIndex['O'],
            dtype = DTpye, padding = Padding, truncating = Truncating
        )
        return tags