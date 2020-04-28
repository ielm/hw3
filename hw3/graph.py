import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from networkx.algorithms.centrality import degree_centrality, in_degree_centrality
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
from typing import Union, List, Tuple
import xml.etree.ElementTree as ET
from manage import NPS_DATA_DIR

from manage import IMAGE_DIR

import threading
import random
import re

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

COLORS = ["#acc2d9", "#56ae57", "#b2996e", "#a8ff04", "#69d84f", "#894585", "#70b23f", "#d4ffff", "#65ab7c", "#952e8f", "#fcfc81", "#a5a391", "#388004", "#4c9085", "#5e9b8a", "#efb435", "#d99b82", "#0a5f38", "#0c06f7", "#61de2a", "#3778bf", "#2242c7", "#533cc6", "#9bb53c", "#05ffa6", "#1f6357", "#017374", "#0cb577", "#ff0789", "#afa88b", "#08787f", "#dd85d7", "#a6c875", "#a7ffb5", "#c2b709", "#e78ea5", "#966ebd", "#ccad60", "#ac86a8", "#947e94", "#983fb2", "#ff63e9", "#b2fba5", "#63b365",
          "#8ee53f", "#b7e1a1", "#ff6f52", "#bdf8a3", "#d3b683", "#fffcc4", "#430541", "#ffb2d0", "#997570", "#ad900d", "#c48efd", "#507b9c", "#7d7103", "#fffd78", "#da467d", "#410200", "#c9d179", "#fffa86", "#5684ae", "#6b7c85", "#6f6c0a", "#7e4071", "#009337", "#d0e429", "#fff917", "#1d5dec", "#054907", "#b5ce08", "#8fb67b", "#c8ffb0", "#fdde6c", "#ffdf22", "#a9be70", "#6832e3", "#fdb147", "#c7ac7d", "#fff39a", "#850e04", "#efc0fe", "#40fd14", "#b6c406", "#9dff00", "#3c4142", "#f2ab15",
          "#ac4f06", "#c4fe82", "#2cfa1f", "#9a6200", "#ca9bf7", "#875f42", "#3a2efe", "#fd8d49", "#8b3103", "#cba560", "#698339", "#0cdc73", "#b75203", "#7f8f4e", "#26538d", "#63a950", "#c87f89", "#b1fc99", "#ff9a8a", "#f6688e", "#76fda8", "#53fe5c", "#4efd54", "#a0febf", "#7bf2da", "#bcf5a6", "#ca6b02", "#107ab0", "#2138ab", "#719f91", "#fdb915", "#fefcaf", "#fcf679", "#1d0200", "#cb6843", "#31668a", "#247afd", "#ffffb6", "#90fda9", "#86a17d", "#fddc5c", "#78d1b6", "#13bbaf", "#fb5ffc",
          "#20f986", "#ffe36e", "#9d0759", "#3a18b1", "#c2ff89", "#d767ad", "#720058", "#ffda03", "#01c08d", "#ac7434", "#014600", "#9900fa", "#02066f", "#8e7618", "#d1768f", "#96b403", "#fdff63", "#95a3a6", "#7f684e", "#751973", "#089404", "#ff6163", "#598556", "#214761", "#3c73a8", "#ba9e88", "#021bf9", "#734a65", "#23c48b", "#8fae22", "#e6f2a2", "#4b57db", "#d90166", "#015482", "#9d0216", "#728f02", "#ffe5ad", "#4e0550", "#f9bc08", "#ff073a", "#c77986", "#d6fffe", "#fe4b03", "#fd5956",
          "#fce166", "#b2713d", "#1f3b4d", "#699d4c", "#56fca2", "#fb5581", "#3e82fc", "#a0bf16", "#d6fffa", "#4f738e", "#ffb19a", "#5c8b15", "#54ac68", "#89a0b0", "#7ea07a", "#1bfc06", "#cafffb", "#b6ffbb", "#a75e09", "#152eff", "#8d5eb7", "#5f9e8f", "#63f7b4", "#606602", "#fc86aa", "#8c0034", "#758000", "#ab7e4c", "#030764", "#fe86a4", "#d5174e", "#fed0fc", "#680018", "#fedf08", "#fe420f", "#6f7c00", "#ca0147", "#1b2431", "#00fbb0", "#db5856", "#ddd618", "#41fdfe", "#cf524e", "#21c36f",
          "#a90308", "#6e1005", "#fe828c", "#4b6113", "#4da409", "#beae8a", "#0339f8", "#a88f59", "#5d21d0", "#feb209", "#4e518b", "#964e02", "#85a3b2", "#ff69af", "#c3fbf4", "#2afeb7", "#005f6a", "#0c1793", "#ffff81", "#f0833a", "#f1f33f", "#b1d27b", "#fc824a", "#71aa34", "#b7c9e2", "#4b0101", "#a552e6", "#af2f0d", "#8b88f8", "#9af764", "#a6fbb2", "#ffc512", "#750851", "#c14a09", "#fe2f4a", "#0203e2", "#0a437a", "#a50055", "#ae8b0c", "#fd798f", "#bfac05", "#3eaf76", "#c74767", "#b9484e",
          "#647d8e", "#bffe28", "#d725de", "#b29705", "#673a3f", "#a87dc2", "#fafe4b", "#c0022f", "#0e87cc", "#8d8468", "#ad03de", "#8cff9e", "#94ac02", "#c4fff7", "#fdee73", "#33b864", "#fff9d0", "#758da3", "#f504c9", "#77a1b5", "#8756e4", "#889717", "#c27e79", "#017371", "#9f8303", "#f7d560", "#bdf6fe", "#75b84f", "#9cbb04", "#29465b", "#696006", "#adf802", "#c1c6fc", "#35ad6b", "#fffd37", "#a442a0", "#f36196", "#947706", "#fff4f2", "#1e9167", "#b5c306", "#feff7f", "#cffdbc", "#0add08",
          "#87fd05", "#1ef876", "#7bfdc7", "#bcecac", "#bbf90f", "#ab9004", "#1fb57a", "#00555a", "#a484ac", "#c45508", "#3f829d", "#548d44", "#c95efb", "#3ae57f", "#016795", "#87a922", "#f0944d", "#5d1451", "#25ff29", "#d0fe1d", "#ffa62b", "#01b44c", "#ff6cb5", "#6b4247", "#c7c10c", "#b7fffa", "#aeff6e", "#ec2d01", "#76ff7b", "#730039", "#040348", "#df4ec8", "#6ecb3c", "#8f9805", "#5edc1f", "#d94ff5", "#c8fd3d", "#070d0d", "#4984b8", "#51b73b", "#ac7e04", "#4e5481", "#876e4b", "#58bc08",
          "#2fef10"]

SID = SentimentIntensityAnalyzer()


def avg(lst):
    return sum(lst) / len(lst)


def load_nps_data(filename: str = "11-09-20s_706posts.xml", debug: bool = False) -> Tuple[list, List[Tuple]]:
    """
    Returns a list of unique users in the chat and a list of posts in the chat.
    Each post is a 4-tuple containing (<Post Class>, <Post User>, <Post Text>, <Post Terminals>)
    <Post Terminals> is a tuple containing (<POS>, <Word>)

    :param filename: name of nps file to load data from
    :return: List[str], List[tuple]
    """
    _users = []
    _posts = []

    filepath = f"{NPS_DATA_DIR}/{filename}"
    with open(filepath) as f:
        tree = ET.parse(f)
        root = tree.getroot()

        for Post in root.findall('./Posts/Post'):
            attribute = Post.attrib
            if debug:
                print(f"\n{attribute['user']} says: {Post.text}. \nIt is a {attribute['class']}")
            _posts.append((attribute['class'],
                           attribute['user'],
                           Post.text,
                           [(t.attrib['pos'], t.attrib['word']) for t in Post.findall('./terminals/t')]))
            if attribute['user'] not in _users:
                _users.append(attribute['user'])
    return _users, _posts


_sentinel = object()


def build_nps_graph(_users: list, _posts: list):
    g = TopicGraph()
    g.add_node(_sentinel, rank="source")
    _seen_users = []
    for i, p in enumerate(_posts):
        post = Post(p, i)
        user = User(post.user, _users.index(post.user))
        post.user = user.id
        g.add_node(post, label=f"{post.id}")
        if user.id not in _seen_users:
            g.add_node(user, rank="min", label=f"{user.id}")
            g.add_edge(user, _sentinel, label="PART_OF", style="dotted")
            _seen_users.append(user.id)
        g.add_edge(post, user, label=f"POSTED_BY")

        punctokenizer = RegexpTokenizer(r'\w+')
        ptokens = punctokenizer.tokenize(post.text)
        # ptokens = word_tokenize(post.text)

        swords = set(stopwords.words("english"))
        system = ["ACTION", "PM", "PART", "PST", "JOIN", "ROOM"]
        topic_cats = ["NN", "NNP", "NNS", "NNPS", "PRP"]  # "VB", "VBG", "VBD", "VBN", "VBP"
        misc_words = ["im", "na", "yr", "gon", "yer", "ya"]

        ptopics = [t for t in nltk.pos_tag(ptokens) if t[1] in topic_cats and
                   t[0] not in swords and t[0] not in system and t[0] not in misc_words]
        topic = Topic(ptopics, user, g.context.get_next_index(), cutoff=g.context.n) if ptopics else None
        update = g.context.update(topic, debug=False)
        if not update["LINK"]:
            if topic:
                g.add_node(topic, label=f"{topic.id}")
                g.add_edge(topic, user, label="INTRODUCED_BY")
        if update["LINK"]:
            g.add_edge(update["TOPIC"], topic, label="REFERENCED_BY")
            g.add_edge(topic, user, label="CONTRIBUTED_BY")
    return g


def analyze_graph(graph: 'TopicGraph'):
    users = {}
    for node in graph.nodes():
        if isinstance(node, User):
            ins = [f"{e}" for e in graph.in_edges(node)]
            outs = [f"{e}" for e in graph.out_edges(node)]
            if node.id not in users.keys():
                user = {"NODE": node,
                        "INS": len(ins),
                        "OUTS": len(outs)}
                users[node.id] = user
            else:
                users[node.id]["INS"] += len(ins)
                users[node.id]["OUTS"] += len(outs)
            # average_user_sentiment(graph, node)

            # users[node.id] = user
        #     print(f"\n{node}")
        #     print(f"INS: {len(ins)}\nOUTS: {len(outs)}")
        # print(f"\n\n{node}: {graph[node]}")
    # print(users)

    # print([v for k, v in sorted(users.items(), key=lambda item: item)])
    # print(users.items())
    centrality = degree_centrality(graph)
    user_stats = []
    for u in [(k, v) for k, v in sorted(users.items(), key=lambda item: item[1]["INS"], reverse=True)]:
        user_posts = get_user_posts(graph, u[1]["NODE"])
        sentiment = average_user_sentiment(user_posts)

        u[1]["SENTIMENT"] = sentiment
        u[1]["NUM_POSTS"] = len(user_posts)
        u[1]["AVG_POST_LEN"] = sum([len(p.text) for p in user_posts]) / len(user_posts)
        u[1]["NUM_TOPICS"] = len(get_user_topics(graph, u[1]["NODE"]))

        for k in centrality.keys():
            if f"{k}" == f"{u[1]['NODE'].id}":
                u[1]["CENTRALITY_SCORE"] = centrality[k]
        # print(f"{u[0]}:\t{u[1]}")
        user_stats.append(u)

    num_in_edges = float(sum(i[1]["INS"] for i in user_stats))

    for u in user_stats:
        u[1]["EDGE_SCORE"] = float(u[1]["INS"]) / num_in_edges
        u[1]["TOPIC_SCORE"] = float(u[1]["NUM_TOPICS"]) / len(get_all_posts(graph))
        u[1]["AVG_POST_LEN_SCORE"] = float(u[1]["AVG_POST_LEN"]) / sum([l[1]["AVG_POST_LEN"] for l in user_stats])

    for u in user_stats:
        _u = u[1]
        influence = avg([_u[score] for score in _u.keys() if "SCORE" in score])
        # weight = float(_u["NUM_POSTS"])/len(get_all_posts(graph))
        _u["RAW_INFLUENCE"] = influence

    inorm = sum(i[1]["RAW_INFLUENCE"] for i in user_stats)
    # print(inorm)
    for u in user_stats:
        _u = u[1]
        _u["INFLUENCE"] = float(_u["RAW_INFLUENCE"]) / inorm
    for i, u in enumerate(sorted(user_stats, key=lambda item: item[1]["INFLUENCE"], reverse=True)):
        _u = u[1]
        # _u["INFLUENCE"] = float(_u["RAW_INFLUENCE"])/inorm
        _u["RANK"] = i
        # print(f"{u[0]}:\n\tSENTIMENT: {_u['SENTIMENT']}\n\t"
        #       f"INFLUENCE: {_u['INFLUENCE']}\n\t"
        #       f"RANK: {i}")
    return user_stats


def get_longstring(posts: list, remove_stopwords=True, remove_users=True):
    from nltk.corpus import stopwords
    longstring = ""
    for p in posts:
        s = p[2]
        s = re.sub('[,.!?]', '', s)
        s = s.lower()

        stokens = word_tokenize(s)
        if remove_stopwords:
            swords = set(stopwords.words("english"))
            miscwords = ["im", "na", "yr", "gon", "wan", "wana", "wanna", "want", "n't", "damn", "pm"]
            stokens = [w for w in stokens if w not in swords]
            stokens = [w for w in stokens if w not in miscwords]
        if remove_users:
            stokens = [w for w in stokens if not 'user' in w]
        longstring = longstring + " ".join(stokens)
    return longstring


def plot(stats, mode: str = "INDIVIDUAL", display: bool = False, to_file: bool = False, filename: str = "img.png"):
    color = 0
    random.shuffle(COLORS)
    if mode == "INDIVIDUAL":
        plt.scatter("Sentiment", "Influence", data={"Sentiment": [x[1]["SENTIMENT"] for x in stats], "Influence": [y[1]["INFLUENCE"] for y in stats]}, c=COLORS[random.randint(0, len(COLORS))])
        plt.title(f"Graph of {filename.split('.')[0].strip('PLOT_')}", fontsize=18, loc="center", pad=10)
    elif mode == "COMPOSITE":
        for _s in stats:
            plt.scatter("Sentiment", "Influence", data={"Sentiment": [x[1]["SENTIMENT"] for x in _s], "Influence": [y[1]["INFLUENCE"] for y in _s]}, c=COLORS[color])
            plt.title(f"Graph of {filename.split('.')[0].strip('PLOT_')}", fontsize=18, loc="center", pad=10)
            color += 1
            if color >= len(COLORS):
                color = 0
    if to_file:
        plt.savefig(f"{IMAGE_DIR}/{filename}")
    if display:
        plt.show()
    plt.close()


def build_wordcloud(posts: list, display: bool = False, to_file: bool = False, filename: str = "wordcloud.png"):
    longstring = get_longstring(posts)
    wordcloud = WordCloud(height=300, width=600, background_color="white", scale=1.5, max_words=10000, contour_width=3, contour_color='steelblue')
    wordcloud.generate(longstring)

    if to_file:
        wordcloud.to_file(f"{IMAGE_DIR}/{filename}")
    if display:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


def build_csv(table):
    import csv
    with open('table.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for row in table:
            filewriter.writerow(row)


def get_user_posts(graph: 'TopicGraph', user: 'User'):
    return [n for n in graph.nodes() if isinstance(n, Post) and n.user == user.id]


def get_user_topics(graph: 'TopicGraph', user: 'User'):
    return [n for n in graph.nodes() if isinstance(n, Topic) and f"{n.user}" == user.id]


def get_all_posts(graph):
    return [n for n in graph.nodes() if isinstance(n, Post)]


def average_user_sentiment(posts):
    # sentiment = {"POS": [], "NEG": [], "NEU": []}
    # sentiment = []
    # for p in posts:
    #     scores = SID.polarity_scores(p.text)
    # print(scores)
    # sentiment["POS"].append(scores["pos"])
    # sentiment["NEG"].append(scores["neg"])
    # sentiment["NEU"].append(scores["neu"])
    # sentiment.append(scores["compound"])

    # return {"POS": avg(sentiment['POS']),
    #         "NEG": avg(sentiment['NEG']),
    #         "NEU": avg(sentiment['NEU'])}
    # return (-1.0 * avg(sentiment['NEG'])) + (avg(sentiment["POS"])) * avg(sentiment["NEU"])

    # return avg(sentiment)
    return avg([SID.polarity_scores(p.text)["compound"] for p in posts])


# ================ OBJECTS ================ #


class User:
    def __init__(self, _user: str, _index: int):
        self._username = _user
        self._id = f"U-{_index}"
        self._has_post = []  # These should be added as edges
        self._introduces_topic = []  # These should be added as edges

    @property
    def name(self):
        return self._username

    @name.setter
    def name(self, value):
        self._username = value

    @property
    def id(self):
        return self._id

    @property
    def post(self):
        return self._has_post

    @post.setter
    def post(self, value):
        self._has_post = value

    @property
    def topic(self):
        return self._introduces_topic

    @topic.setter
    def topic(self, value):
        self._introduces_topic = value

    def __str__(self):
        return f"{self.id}"

    def __repr__(self):
        return f"{self.id}"


class Post:

    def __init__(self, _post: tuple, _conv_index: int):
        self._conv_index = _conv_index
        self.index = self._conv_index
        self._id = f"P-{self.index}"
        self._dialog_act = _post[0]
        self._posted_by = _post[1]
        self._text = _post[2]
        self._terminals = _post[3]
        self._introduces_topic = []  # These should be added as edges

    @classmethod
    def build(cls, _post, _index, debug=False):
        attribute = _post.attrib
        if debug:
            print(f"\n{attribute['user']} says: {_post.text}. \nIt is a {attribute['class']}")
        _p = (attribute['class'],
              attribute['user'],
              _post.text,
              [(t.attrib['pos'], t.attrib['word']) for t in _post.findall('./terminals/t')])
        return Post(_p, _index)

    @property
    def dialog_act(self):
        return self._dialog_act

    @dialog_act.setter
    def dialog_act(self, value):
        self._dialog_act = value

    @property
    def id(self):
        return self._id

    @property
    def user(self):
        return self._posted_by

    @user.setter
    def user(self, value):
        self._posted_by = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def terminals(self):
        return self._terminals

    @terminals.setter
    def terminals(self, value):
        self._terminals = value

    @property
    def topic(self):
        return self._introduces_topic

    @topic.setter
    def topic(self, value):
        self._introduces_topic = value

    def __str__(self):
        return f"{self.id}"

    def __repr__(self):
        return f"{self.id}"


class Topic:

    def __init__(self, constituents, user, index, cutoff):
        self._id = f"T-{index}"
        self._constituents = constituents
        self._posted_by = user
        self._topic_index = index
        self._cutoff = cutoff
        self._counter = AtomicCounter()

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def user(self):
        return self._posted_by

    @user.setter
    def user(self, value):
        self._posted_by = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value

    @property
    def count(self):
        return self._counter.value

    def increment_counter(self, value=1):
        self._counter.increment(value)

    def increment_cutoff(self, value=1):
        self.cutoff = self.cutoff + value

    def constituents(self):
        return self._constituents

    def intersect(self, topic: 'Topic'):
        for c in self.constituents():
            if c in topic.constituents():
                return True
        return False

    def rep(self):
        s = f"{self.id}:\n"
        for c in self.constituents():
            s = f"{s}\t{c[0]} : [{c[1]}]\n"
        s = f"{s}\tCOUNT : {self.count}"
        return s

    def __str__(self):
        return f"{self.id}"

    def __repr__(self):
        return f"{self.id}"


class AtomicCounter:
    def __init__(self, initial=0):
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        with self._lock:
            self.value += num
            return self.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class TopicContext:
    def __init__(self, n: int = 10, debug: bool = False):
        self._store = []
        self._n = n
        self._topic_index = AtomicCounter()
        if debug:
            print(f"STORE: {self.store}\nN: {self.n}\nTOPIC INDEX: {self.index}")

    def update(self, topic: Union[Topic, None] = None, debug: bool = False):
        # update topic memory with current topic
        output = {"LINK": False,
                  "TOPIC": None}
        if debug: print("\n\n\\/ ========== DEBUG ========== \\/\n")

        if topic:
            if debug:
                print(f"TOPIC: {topic.id}\n")
                [print(t.rep()) for t in self.store]

            found = False
            for t in self.store:
                if topic.intersect(t):
                    output = {"LINK": True,
                              "TOPIC": t}
                    t.increment_cutoff(value=2)
                    found = True
                    if debug: print(f"\nFOUND INTERSECTION: {t.id}"
                                    f"\nTOPIC CUTOFF: {t.cutoff}"
                                    f"\nTOPIC COUNT: {t.count}")
                    break
            if not found:
                if debug: print(f"NEW TOPIC: {topic.id}")
                self.add_topic(topic)
        else:
            if debug: print(f"TOPIC IS NONE.")
        self.advance(debug=debug)
        if debug: print("\n/\\ ========== DEBUG ========== /\\\n")
        return output

    def advance(self, debug: bool = False):
        loss = len(self.store)
        for i, t in enumerate(self.store):
            t.increment_counter()
            if t.count == t.cutoff:
                self.remove_topic(i)
        loss -= len(self.store)
        if debug: print(f"STORE LOSS: {loss}")

    def add_topic(self, topic: Topic, debug: bool = False):
        self._store.append(topic)

    def remove_topic(self, index: int, debug: bool = False):
        del self._store[index]

    @property
    def store(self):
        return self._store

    def get_next_index(self):
        self._topic_index.increment()
        return self.index

    @property
    def index(self):
        return self._topic_index

    @property
    def n(self):
        return self._n


class TopicGraph(nx.MultiDiGraph):

    def __init__(self):
        super(TopicGraph, self).__init__()
        self.context = TopicContext()
