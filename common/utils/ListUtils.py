class ListUtils():
    @staticmethod
    def FlattenList(listOfLists):
        flattenList = [item for sublist in listOfLists for item in sublist]
        return flattenList