# import networkx as nx
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.tokenize import RegexpTokenizer
# from textblob import TextBlob
# # import pylab as plt
# import matplotlib.pyplot as plt
# import seaborn as sns
# from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
# from networkx.algorithms.centrality import degree_centrality
# import pygraphviz as pgv
# import xml.etree.ElementTree as ET
# from wordcloud import WordCloud
# import threading
# import re
#
# from utils import User, Post, Topic, TopicGraph, avg, build_wordcloud, analyze_graph, build_csv, plot, FILES
# from data import load_nps_data
#
# _sentinel = object()
#
#
# def build_nps_graph(_users: list, _posts: list):
#     g = TopicGraph()
#     g.add_node(_sentinel, rank="source")
#     _seen_users = []
#     for i, p in enumerate(_posts):
#         post = Post(p, i)
#         user = User(post.user, _users.index(post.user))
#         post.user = user.id
#         g.add_node(post, label=f"{post.id}")
#         if user.id not in _seen_users:
#             g.add_node(user, rank="min", label=f"{user.id}")
#             g.add_edge(user, _sentinel, label="PART_OF", style="dotted")
#             _seen_users.append(user.id)
#         g.add_edge(post, user, label=f"POSTED_BY")
#
#         punctokenizer = RegexpTokenizer(r'\w+')
#         ptokens = punctokenizer.tokenize(post.text)
#         # ptokens = word_tokenize(post.text)
#
#         swords = set(stopwords.words("english"))
#         system = ["ACTION", "PM", "PART", "PST", "JOIN", "ROOM"]
#         topic_cats = ["NN", "NNP", "NNS", "NNPS", "PRP"]  # "VB", "VBG", "VBD", "VBN", "VBP"
#         misc_words = ["im", "na", "yr", "gon", "yer", "ya"]
#
#         ptopics = [t for t in nltk.pos_tag(ptokens) if t[1] in topic_cats and
#                    t[0] not in swords and t[0] not in system and t[0] not in misc_words]
#         topic = Topic(ptopics, user, g.context.get_next_index(), cutoff=g.context.n) if ptopics else None
#         update = g.context.update(topic, debug=False)
#         if not update["LINK"]:
#             if topic:
#                 g.add_node(topic, label=f"{topic.id}")
#                 g.add_edge(topic, user, label="INTRODUCED_BY")
#         if update["LINK"]:
#             g.add_edge(update["TOPIC"], topic, label="REFERENCED_BY")
#             g.add_edge(topic, user, label="CONTRIBUTED_BY")
#     return g
#
#
# def run_stats_old():
#     sid = SentimentIntensityAnalyzer()
#     for f in FILES:
#         users, posts = load_nps_data(f)
#
#         graph = build_nps_graph(users, posts)
#         cent = degree_centrality(graph)
#         cent_list = []
#
#         for k in cent.keys():
#             if cent[k] != 0.0:
#                 # print(cent[k])
#                 cent_list.append(cent[k])
#
#         sentiment = {"POS": [], "NEG": [], "NEU": []}
#
#         for p in posts:
#             scores = sid.polarity_scores(p[2])
#             sentiment["POS"].append(scores["pos"])
#             sentiment["NEG"].append(scores["neg"])
#             sentiment["NEU"].append(scores["neu"])
#
#         print(f"\nFILE: {f}\n\t\
#             USERS: {len(users)}\n\t\
#             POSTS: {len(posts)}\n\t\
#             POSITIVE AVG: {avg(sentiment['POS'])}\n\t\
#             NEGATIVE AVG: {avg(sentiment['NEG'])}\n\t\
#             NEUTRAL AVG: {avg(sentiment['NEU'])}\n\t\
#             AVG CENTRALITY: {avg(cent_list)}")
#
#
#
#
# if __name__ == '__main__':
#     usr, pst = load_nps_data()
#     # build_nps_graph(usr, pst)
#
#     # build_wordcloud(pst, to_file=True)
#     G = build_nps_graph(usr, pst)
#
#     # ug = G.subgraph([n for n in G.nodes() if isinstance(n, User)])
#     # ug =  G.to_undirected()
#     # ua = to_agraph(ug)
#     # ua.graph_attr.update(directed=True, ranksep=4, overlap="scale", ratio="auto", clusterrank="global", center=True)
#     # print(ua)
#     # ua.layout('dot')
#     # ua.draw('ua1.png')
#
#     # stats = analyze_graph(G)
#     # plt.scatter("Sentiment", "Influence", data={"Sentiment": [x[1]["SENTIMENT"] for x in stats], "Influence": [y[1]["INFLUENCE"] for y in stats]})
#
#     # nx.spring_layout(G)
#     # A = to_agraph(G)
#     # A.graph_attr.update(directed=True, ranksep=4, overlap="scale", ratio="auto", clusterrank="global", center=True)
#     # print(A)
#     # A.layout('dot')
#     # A.draw('dotagain.png')
#
#     # run(mode="COMPOSITE", wc_display=False, plt_display=True, plt_to_file=True)
#
#     # nx.spring_layout(G)
#     # nx.draw_networkx(G)
#     # edge_labels=dict([((u,v,))
#     #              for u,v,d in G.edges(data=True)])
#     # nx.draw_networkx(G, pos, label_pos=0.3, font_size=7)
#     # edge_labels=dict([((u,v,),d['tag'])
#     #          for u,v,d in G.edges(data=True)])
#
#     # plt.show()
#     # print(f"\t{avg(cent_list)}")
#
#
#     # users, posts = load_nps_data()
#
#     # print(len(users))
#     # print(len(posts))
#     # graph = build_nps_graph(users, posts)
#     # print(graph.graph)
#
