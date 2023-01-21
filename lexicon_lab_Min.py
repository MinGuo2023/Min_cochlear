import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from switch import switch_simdrop

"""The Code takes several minutes to execute."""
class similarity:

    def __init__(self):
        pass

    """This function takes two words and returns the cosine similarity between them based on a model"""
    def cosine_similarity (self, word1, word2, model):
        sim = 0
        vec1, vec2 = self.vecgenerator(word1, word2, model)
        if vec1 == 0:
            return 0
        norm_vec1 = self.normalize(vec1)
        norm_vec2 = self.normalize(vec2)
        """Calculate the dot product of the two normal vectors"""
        for i in range(len(vec1)):
            sim += (norm_vec1[i] * norm_vec2[i])
        return sim

    """This function takes two words and convert them to vectors based on a given model"""
    def vecgenerator(self, word1, word2, model):
        vec1 = []
        vec2 = []
        f = open(model + ".txt", "r")
        r = f.readlines()
        for line in r:
            line_list = line.split()
            if line_list[0] == word1:
                vec1 = line_list[1:]
            if line_list[0] == word2:
                vec2 = line_list[1:]
        if vec1 == [] or vec2 == []:
            return 0, 0
        for i in range(50):
            vec1[i] = float(vec1[i])
            vec2[i] = float(vec2[i])
        return vec1, vec2

    """This function converts a given vector into a normal vector of the same direction"""
    def normalize(self, vec):
        normal_vec = vec
        norm_squared = 0
        for num in vec:
            norm_squared += (num*num)
        norm = np.sqrt(norm_squared)
        for i in range(len(vec)):
            normal_vec[i] = vec[i]/norm
        return normal_vec

    """This function visualizes the words produced by a given participant in a TSNE plot"""
    def visualize_items(self, ID):
        data_speech, data_word = self.retrieve(ID)
        tsne = TSNE(n_components=2, perplexity=10.0, verbose=1, random_state=123)
        z = tsne.fit_transform(data_speech)
        y = np.zeros(len(data_speech))
        df = pd.DataFrame()
        df["y"] = y
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", 1),
                        data=df).set(title="Cochlear data T-SNE projection")
        tsne = TSNE(n_components=2, perplexity=10.0, verbose=1, random_state=123)
        z = tsne.fit_transform(data_word)
        y = np.zeros(len(data_word))
        for i in range(len(y)):
            y[i] = 1
        df = pd.DataFrame()
        df["y"] = y
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=dict({1: "green"}),
                        data=df).set(title="Cochlear data T-SNE projection：\nred dots: speech2vec; green dots:word2vec")
        plt.show()

    """This function returns the list of words produced by a participant in word2vec and speech2vec"""
    def retrieve(self, ID):
        f0 = open("data-cochlear.txt", "r")
        r0 = f0.readlines()
        words = []
        for line in r0:
            line_list = line.split()
            if line_list[0] == ID:
                words.append(line_list[1])
        data_speech = []
        data_word = []
        f = open("speech2vec.txt", "r")
        r = f.readlines()
        f2 = open("word2vec.txt", "r")
        r2 = f2.readlines()
        for word in words:
            for line in r:
                line_list = line.split()
                if line_list[0] == word:
                    vec = line_list[1:]
                    data_speech.append(vec)
            for line in r2:
                line_list = line.split()
                if line_list[0] == word:
                    vec = line_list[1:]
                    data_word.append(vec)
        arr = np.array(data_speech)
        arr2 = np.array(data_word)
        return arr, arr2

    """This function returns the consecutive pairwise similarity of the data list in the pandas dataframe format"""
    def pairwise_similarity(self, data, model):
        r = data.readlines()
        col1 = []
        col2 = []
        col3 = []
        present_participant_ID = ""
        previous_word= ""
        index = 0
        for line in r:
            print(index)
            index += 1
            line_list = line.split()
            col2.append(line_list[1])
            if line_list[0] != present_participant_ID:
                present_participant_ID = line_list[0]
                col1.append(line_list[0])
                col3.append(2)
            else:
                col1.append(line_list[0])
                col3.append(self.cosine_similarity(line_list[1], previous_word, model))
            previous_word = line_list[1]
        d = {'ID': col1, "Item": col2, "Similarity": col3}
        df = pd.DataFrame(data=d)
        return df, col1, col2, col3


class cluster:

    def __init__(self):
        pass


    """
        Args:
            data: the data file 
            model: either word2vec or speech2vec
        Returns:
            a dictionary in the following form:
            {Participant ID: [number of clusters, number of switches]}"""
    def compute_clusters(self, data, model):
        sim = similarity()
        pd, col1, col2, col3 = sim.pairwise_similarity(data, model)
        dict = {}
        present_id = col1[0]
        fluency_list = []
        similarity_list = []
        for num in range(1, len(col1)):
            print(num)
            if present_id != col1[num]:
                simdrop_list = switch_simdrop(fluency_list, similarity_list)
                print(simdrop_list)
                num_switches = 0
                for i in simdrop_list:
                    if i == 1:
                        num_switches += 1
                num_clusters = num_switches + 1
                if simdrop_list[1] == 1:
                    num_clusters -= 1
                if simdrop_list[-2] == 1:
                    num_clusters -= 1
                dict[present_id] = [num_clusters, num_switches]
                fluency_list = []
                similarity_list = []
                present_id = col1[num]
            elif num == len(col1) - 1:
                simdrop_list = switch_simdrop(fluency_list, similarity_list)
                print(simdrop_list)
                num_switches = 0
                for i in simdrop_list:
                    if i == 1:
                        num_switches += 1
                num_clusters = num_switches + 1
                if simdrop_list[1] == 1:
                    num_clusters -= 1
                if simdrop_list[-2] == 1:
                    num_clusters -= 1
                dict[present_id] = [num_clusters, num_switches]
            else:
                fluency_list.append(col2[num])
                similarity_list.append(col3[num])
        return dict


    """plots a bar graph showing the number of clusters and switches for a given participant using a given model."""
    def visualize_clusters(self, ID, model):

        f = open("data-cochlear.txt", "r")
        dict= self.compute_clusters(f, model)
        num_of_participants = len(dict.keys())
        total_clusters = 0
        total_switches = 0
        for i in dict.values():
            total_clusters += i[0]
            total_switches += i[1]
        ave_clu = total_clusters/num_of_participants
        ave_swi = total_switches / num_of_participants
        dict_plot = {"# of clusters for " + ID: dict[ID][0], "# of switches for " + ID: dict[ID][1],
                     "Average # of clusters": ave_clu, "Average # of switches": ave_swi}
        x = list(dict_plot.keys())
        y = list(dict_plot.values())
        ave_clu = total_clusters / num_of_participants
        fig = plt.figure(figsize=(10, 5))
        plt.bar(x, y, color='maroon',
                width=0.4)

        plt.ylabel("Number")
        plt.title("number of clusters and switches for Participant " + ID)
        plt.show()


s = similarity()
print(s.cosine_similarity("goose", "duck", "speech2vec"))
print(s.cosine_similarity("of", "of", "word2vec"))
s.visualize_items("CBS-711")
f = open("data-cochlear.txt", "r")
pd, col1, col2, col3 = s.pairwise_similarity(f, "speech2vec")
print(pd)
clu = cluster()
print(clu.compute_clusters(f, "speech2vec"))
clu.visualize_clusters("CAF-657", "speech2vec")




