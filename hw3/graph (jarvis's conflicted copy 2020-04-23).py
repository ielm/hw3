import networkx as nx
import nltk
from textblob import TextBlob
from networkx.algorithms.centrality import degree_centrality
# import pylab as plt
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv
import xml.etree.ElementTree as ET
from wordcloud import WordCloud
import re


from hw3 import User, Post, Topic, avg, build_wordcloud, run_stats
from data import load_nps_data
from manage import NPS_DATA_DIR

_sentinel = object()

FILES = ["10-19-20s_706posts.xml",
         "10-19-30s_705posts.xml",
         "10-19-40s_686posts.xml",
         "10-19-adults_706posts.xml",
         "10-24-40s_706posts.xml",
         "10-26-teens_706posts.xml",
         "11-06-adults_706posts.xml",
         "11-08-20s_705posts.xml",
         "11-08-40s_706posts.xml",
         "11-08-adults_705posts.xml",
         "11-08-teens_706posts.xml",
         "11-09-20s_706posts.xml",
         "11-09-40s_706posts.xml",
         "11-09-adults_706posts.xml",
         "11-09-teens_706posts.xml"]

def build_nps_graph(_users: list, _posts: list):
    g = nx.DiGraph()

    for i, p in enumerate(_posts):
        post = Post(p, i)
        user = User(post.user)
        g.add_node(post)
        g.add_node(user)
        g.add_edge(user, post, tag=f"{post.index}")
    return g



if __name__ == '__main__':
    usr, pst = load_nps_data()

    # build_wordcloud(pst)
    # G = build_nps_graph(usr, pst)
    # A = to_agraph(G)
    # print(A)
    # A.layout('dot')
    # A.draw('abcde.png')

    # run_stats()

    # nx.spring_layout(G)
    # nx.draw_networkx(G, label_pos=0.3, font_size=7)
    # edge_labels=dict([((u,v,),d['tag'])
    #          for u,v,d in G.edges(data=True)])
    plt.show()
