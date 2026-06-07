"""Helper to assemble a Jupyter notebook from a list of (kind, source) cells.

Usage in a builder script:
    from nbbuild import md, code, write_nb
    cells = [md("# title"), code("import pandas as pd")]
    write_nb("practices/practiceX.ipynb", cells)
"""
import nbformat as nbf


def md(source):
    return ("md", source.strip("\n"))


def code(source):
    return ("code", source.strip("\n"))


def write_nb(path, cells):
    nb = nbf.v4.new_notebook()
    out = []
    for kind, source in cells:
        if kind == "md":
            out.append(nbf.v4.new_markdown_cell(source))
        else:
            out.append(nbf.v4.new_code_cell(source))
    nb.cells = out
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    nbf.write(nb, path)
    print(f"wrote {path} with {len(out)} cells")
