import networkx as nx


def build_nps_graph(_users: list, _posts: list):
    G = nx.Graph()
    [G.add_node(user) for user in users]
    print(f"{G.nodes()}")


if __name__ == '__main__':
    from data import load_nps_data

    users, posts = load_nps_data()

    build_nps_graph(users, posts)
