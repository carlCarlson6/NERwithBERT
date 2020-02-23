class TagGetter:
    def GetUniqueTagValues(self, DataFrame):
        uniqueTags = list(set(DataFrame['Tag'].values))
        uniqueTags.append('X')
        uniqueTags.append('[CLS]')
        uniqueTags.append('[SEP]')
        
        tagsValues = set(uniqueTags)

        return tagsValues