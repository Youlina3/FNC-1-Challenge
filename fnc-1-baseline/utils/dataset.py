from csv import DictReader
import pandas as pd

class DataSet():
    def __init__(self, name="train", path="fnc-1-data"):
        self.path = path
        self.test_df = None
        print("Reading dataset")
        bodies = "body_table.csv"
        stances = name+"_data.csv"

        #generate dataframe of unlabled test set
        if (name == "test"):
            self.test_df = self.generate_df(stances)

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))



    def read(self,filename):
        rows = []
        with open(self.path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)
        return rows

    def generate_df(self,filename):
        test_df = pd.read_csv(self.path + "/" + filename)
        print ("my test_df:")
        print (test_df)
        return test_df