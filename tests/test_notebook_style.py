"""Presentation changes must preserve executable experiment code."""
import ast
import nbformat
from quantstream.notebook_style import compact


def test_compact_preserves_code_outputs_and_is_idempotent():
    nb = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# Setup\nLong explanation.'),
        nbformat.v4.new_code_cell('budget = 12\nprint(budget)', execution_count=1,
            outputs=[nbformat.v4.new_output('stream',name='stdout',text='12\n')])])
    before = ast.dump(ast.parse(nb.cells[1].source))
    compact(nb,'04_Kaggle_Model_Search.ipynb')
    assert all(c.cell_type == 'code' for c in nb.cells)
    assert ast.dump(ast.parse(nb.cells[0].source)) == before
    assert nb.cells[0].outputs[0].text == '12\n'
    once = nbformat.writes(nb)
    compact(nb,'04_Kaggle_Model_Search.ipynb')
    assert nbformat.writes(nb) == once
