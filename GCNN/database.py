class database:
    def __init__(self, jsontrain='data_train.json', jsonCV='data_CV.json', jsontest ='data_test.json'):
        self.jsontrain= jsontrain
        self.jsonCV = jsonCV
        self.jsontest = jsontest
    def getTrainName(self):
        return self.jsontrain
    def getCVName(self):
        return self.jsonCV
    def getTestName(self):
        return self.jsontest
class CodeChef():
    # def __init__(self, problem='', jsontrain='_train_AstGraph.json', jsonCV='_CV_AstGraph.json', jsontest ='_test_AstGraph.json'):
    def __init__(self, problem='', jsontrain='_CFG_train.json', jsonCV='_CFG_CV.json', jsontest='_CFG_test.json'):
        self.problem = problem
        self.jsontrain= jsontrain
        self.jsonCV = jsonCV
        self.jsontest = jsontest

    def getTrainName(self):
        return self.problem + self.jsontrain

    def getCVName(self):
        return self.problem + self.jsonCV

    def getTestName(self):
        return self.problem +self.jsontest