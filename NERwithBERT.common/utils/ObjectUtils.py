import joblib

class ObjectUtils():

    @staticmethod
    def SaveObject(objectToSave, filename, path):
        try:
            joblib.dump(
                objectToSave,
                open(path+'//'+filename+'.joblib', 'wb')
            )
            return True

        except:
            return False

    @staticmethod
    def LoadObject(filename, path):
        try:
            loadedObject = joblib.load(open(path+'//'+filename+'.joblib', 'rb'))
            return loadedObject

        except:
            return False
