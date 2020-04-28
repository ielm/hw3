from typing import List, Tuple
import xml.etree.ElementTree as ET
from manage import NPS_DATA_DIR


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




if __name__ == '__main__':
    # for f in FILES:
    #     users, posts = load_nps_data(f)
    #     print(f"\nFILE: {f}\n\t USERS: {len(users)}, POSTS: {len(posts)}")
    users, posts = load_nps_data("11-09-20s_706posts.xml", debug = True)