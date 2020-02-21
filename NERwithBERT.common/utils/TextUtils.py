class TextUtils:

    @staticmethod
    def GetMaxLength(inputs):
        lengths = map(len, inputs)
        maxLength = max(lengths)

        return maxLength