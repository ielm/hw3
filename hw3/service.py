import click
from graph import *


def run(mode: str = "COMPOSITE", filename: str = FILES[-1], wc_display: bool = False, wc_to_file: bool = False, plt_mode: str = "INDIVIDUAL", plt_display: bool = False, plt_to_file: bool = False, csv_build: bool = False):
    stats_table = []
    a = []
    fnames = []

    def _run(_table, _mode: str = "INDIVIDUAL", _filename: str = FILES[-1], _wc_display: bool = False, _wc_to_file: bool = False, _plt_mode: str = "INDIVIDUAL", _plt_display: bool = False, _plt_to_file: bool = False):
        _row = [_filename]
        _users, _posts = load_nps_data(_filename)
        _graph = build_nps_graph(_users, _posts)
        if _wc_to_file or _wc_display:
            build_wordcloud(_posts, display=_wc_display, to_file=_wc_to_file, filename=f"{_filename.split('.')[0]}.png")
        _s = analyze_graph(_graph)
        # [row.append(u[1]) for u in s]
        _s = sorted(_s, key=lambda item: item[1]["RANK"])
        for _u in _s:
            _row.append(_u[0])
            _row.append(f"{_u[1]['RANK']}")
            _row.append(f"{_u[1]['SENTIMENT']}")
            _row.append(f"{_u[1]['INFLUENCE']}")
        _table.append(_row)
        if _plt_mode == "INDIVIDUAL":
            if _plt_display or _plt_to_file:
                plot(_s, _plt_mode, _plt_display, _plt_to_file, filename=f"PLOT_{_filename.split('.')[0]}.png")
        return _s, _table

    if mode == "COMPOSITE":
        for f in FILES:
            s, table = _run(stats_table, _mode=mode, _filename=f, _wc_display=wc_display, _wc_to_file=wc_to_file, _plt_mode=plt_mode, _plt_display=plt_display, _plt_to_file=plt_to_file)
            a.append(s)
            fnames.append(f)
            stats_table = table
    elif mode == "INDIVIDUAL":
        s, table = _run(stats_table, _mode=mode, _filename=filename, _wc_display=wc_display, _wc_to_file=wc_to_file, _plt_mode=plt_mode, _plt_display=plt_display, _plt_to_file=plt_to_file)
        a.append(s)
        fnames.append(filename)
        stats_table = table

    slength = max([len(l) for l in stats_table])
    tags = ["" for _ in range(slength)]
    tags[0] = "file name"
    index = 1
    while index < slength:
        tags[index] = "user"
        tags[index+1] = "rank"
        tags[index+2] = "influence"
        tags[index+3] = "sentiment"
        index += 4
    stats_table.insert(0, tags)
    for row in stats_table:
        for i in range(slength-len(row)):
            row.append('')
    if csv_build:
        build_csv(stats_table)
    if mode == "COMPOSITE":
        if plt_display or plt_to_file:
            plot(a, mode=mode, display=plt_display, to_file=plt_to_file, filename=f"COMPOSITE.png")
    return a, fnames


@click.group()
@click.pass_context
def hw3(ctx):
    """SoCo Homework III

    Do this first\n
    Do this next\n
    Do this last\n
    """
    click.echo("\nSentiment of Influential Users"
               "\n\t\tAuthor: Ivan Leon"
               "\n\t\tProfessor: Tomek Strzalkowski")


@hw3.command()
@click.pass_context
def demo(ctx):
    results, filenames = run(mode="COMPOSITE", wc_display=False, wc_to_file=False, plt_mode="COMPOSITE", plt_display=False, plt_to_file=False, csv_build=False)

    for i, ds in enumerate(results):
        print(f"{filenames[i]}:\n\n5 MOST INFLUENTIAL USERS\n")
        for ii in range(5):
            user = ds[ii]
            print(f"\t{user[0]}:\n"
                  f"\t\tRANK: \t\t{user[1]['RANK']+1}\n"
                  f"\t\tSENTIMENT: \t{user[1]['SENTIMENT']}\n"
                  f"\t\tINFLUENCE: \t{user[1]['INFLUENCE']}")
        print("\t...\n")
        for user in [ds[-(ui+1)] for ui in range(5)][::-1]:
            print(f"\t{user[0]}:\n"
                  f"\t\tRANK: \t\t{user[1]['RANK']+1}\n"
                  f"\t\tSENTIMENT: \t{user[1]['SENTIMENT']}\n"
                  f"\t\tINFLUENCE: \t{user[1]['INFLUENCE']}")
        print("\n\n")

@hw3.command()
@click.option('-f', '--filename',
              default="11-09-adults_706posts.xml",
              prompt="\nname of the file containing the conversation data")
@click.option('--wc_display',
              default=False,
              prompt="\ndisplay the wordcloud")
@click.option('--wc_to_file',
              default=False,
              prompt="\nsave the wordcloud to file")
@click.option("--plt_display",
              default=False,
              prompt="\ndisplay the plot")
@click.option("--plt_to_file",
              default=False,
              prompt="\nsave the plot to file")
@click.option("--csv_build",
              default=False,
              prompt="\nbuild and save the data as a csv file")
@click.pass_context
def run_single_conv(ctx, filename: str, wc_display: bool = False, wc_to_file: bool = False, plt_mode: str = "INDIVIDUAL", plt_display: bool = False, plt_to_file: bool = False, csv_build: bool = False):
    _ = run(mode="INDIVIDUAL", filename=filename, wc_display=wc_display, wc_to_file=wc_to_file, plt_mode=plt_mode, plt_display=plt_display, plt_to_file=plt_to_file, csv_build=csv_build)


@hw3.command()
@click.option('--wc_display',
              default=False,
              prompt="\ndisplay the wordcloud for each conversation")
@click.option('--wc_to_file',
              default=False,
              prompt="\nsave the wordcloud for each conversation to file")
@click.option("--plt_mode",
              default="COMPOSITE",
              prompt="\nplot individual datasets or composite of all datasets [COMPOSITE/INDIVIDUAL]")
@click.option("--plt_display",
              default=False,
              prompt="\ndisplay the plot")
@click.option("--plt_to_file",
              default=False,
              prompt="\nsave the plot to file")
@click.option("--csv_build",
              default=False,
              prompt="\nsave the data as a csv file")
@click.pass_context
def run_all_convs(ctx, wc_display: bool = False, wc_to_file: bool = False, plt_mode: str = "COMPOSITE", plt_display: bool = False, plt_to_file: bool = False, csv_build: bool = False):
    _ = run(mode="COMPOSITE", wc_display=wc_display, wc_to_file=wc_to_file, plt_mode=plt_mode, plt_display=plt_display, plt_to_file=plt_to_file, csv_build=csv_build)


def start():
    hw3(obj={})


if __name__ == '__main__':
    start()