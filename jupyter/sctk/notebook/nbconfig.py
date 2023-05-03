__USING_NOTEBOOK = False

def init_notebook(using_notebook=True):
    global __USING_NOTEBOOK
    __USING_NOTEBOOK = using_notebook

def using_notebook():
    return __USING_NOTEBOOK