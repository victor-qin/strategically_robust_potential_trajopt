"""Build (and optionally execute) the notebook from its `# %%` cell source.

The cell source is the file of record: it reviews as a normal diff, where an
`.ipynb` reviews as a wall of JSON with embedded base64 images.  Regenerate with

    python notebooks/build.py                # convert only
    python notebooks/build.py --execute      # convert and run, embedding outputs

Cells are delimited by a line reading `# %%` for code or `# %% [markdown]` for
prose; markdown cells have their comment prefix stripped.  This is the same
"percent format" jupytext uses, so the file also opens directly in editors that
understand it -- jupytext itself is not a dependency.
"""
import argparse
import pathlib
import re
import sys

import nbformat

HERE = pathlib.Path(__file__).parent
DELIM = re.compile(r'^# %%(\s+\[markdown\])?\s*$')


def parse_cells(text):
    """Split percent-format source into (kind, source) pairs.

    Parameters
    ----------
    text : str
        Contents of a `# %%` cell-source file.

    Returns
    -------
    cells : list of (str, str)
        `kind` is 'code' or 'markdown'; `source` is the cell body with leading
        and trailing blank lines stripped.  Empty cells are dropped.
    """
    cells, kind, buf = [], None, []

    def flush():
        """Emit the cell accumulated so far, dropping it when empty."""
        if kind is None:
            return
        src = '\n'.join(buf).strip('\n')
        if src:
            cells.append((kind, src))

    for line in text.splitlines():
        m = DELIM.match(line)
        if m:
            flush()
            kind = 'markdown' if m.group(1) else 'code'
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def strip_comments(src):
    """Turn a commented markdown cell back into markdown.

    Parameters
    ----------
    src : str
        A markdown cell body, each line prefixed with `# ` (or a bare `#` for a
        blank line).

    Returns
    -------
    markdown : str
        The same text with the comment prefix removed.
    """
    out = []
    for line in src.splitlines():
        if line.startswith('# '):
            out.append(line[2:])
        elif line.strip() == '#':
            out.append('')
        else:
            out.append(line)
    return '\n'.join(out)


def build(source_path, out_path, kernel='python3', kernel_name='Python 3'):
    """Convert a percent-format source file into an unexecuted notebook.

    Parameters
    ----------
    source_path : pathlib.Path
        The `# %%` cell-source file to read.
    out_path : pathlib.Path
        Where to write the `.ipynb`.
    kernel : str
        Kernel name recorded in the notebook metadata.
    kernel_name : str
        Kernel display name recorded in the notebook metadata.

    Returns
    -------
    nb : nbformat.NotebookNode
        The notebook, already written to `out_path`.
    """
    nb = nbformat.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': kernel_name,
                                 'language': 'python', 'name': kernel}
    nb.metadata['language_info'] = {'name': 'python'}
    for kind, src in parse_cells(source_path.read_text()):
        nb.cells.append(nbformat.v4.new_markdown_cell(strip_comments(src))
                        if kind == 'markdown' else nbformat.v4.new_code_cell(src))
    nbformat.write(nb, str(out_path))
    return nb


def main():
    """Command-line entry point: convert, and optionally execute, the notebook.

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', nargs='?', default=str(HERE / 'collision_example.py'))
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--execute', action='store_true',
                    help='run every cell and embed the outputs')
    ap.add_argument('--timeout', type=int, default=1800)
    a = ap.parse_args()

    source_path = pathlib.Path(a.source)
    out_path = pathlib.Path(a.out) if a.out else source_path.with_suffix('.ipynb')
    nb = build(source_path, out_path)
    print(f'{source_path.name} -> {out_path.name}  ({len(nb.cells)} cells)')

    if a.execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=a.timeout, kernel_name='python3',
                                resources={'metadata': {'path': str(out_path.parent)}})
        client.execute()
        nbformat.write(nb, str(out_path))
        size = out_path.stat().st_size / 1e6
        print(f'executed and written: {out_path.name} ({size:.2f} MB)')


if __name__ == '__main__':
    sys.exit(main())
