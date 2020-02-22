from transformers import BertTokenizer

class DataTokenizer:

    def TokenizeData(self, Tokenizer, Sentences, Labels):
        tokenizedTexts = []
        tokenizedLabels = []

        for wordList, label in zip(Sentences, Labels):
            tempLabel = []
            tempToken = []

            tempLabel.append('[CLS]')
            tempToken.append('[CLS]')

            for word, lbl in zip(wordList, label):
                tokenList = Tokenizer.tokenize(word)

                for index, token in enumerate(tokenList):
                    tempToken.append(token)
                    if index == 0:
                        tempLabel.append(lbl)
                    else:
                        tempLabel.append('X')
            
            tempLabel.append('[SEP]')
            tempToken.append('[SEP]')

            tokenizedTexts.append(tempToken)
            tokenizedLabels.append(tempLabel)
        
        return tokenizedTexts, tokenizedLabels