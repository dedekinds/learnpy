import functools
def log(text):
    if isinstance(text,str):#如果是字符串的话
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                print('begin')
                print('%s'%text)
                func(*args,**kw)
                print('end')
            return wrapper
        return decorator
    else:
            @functools.wraps(func)
            def wrapper(*args, **kw):
                print('begin')
                func(*args,**kw)
                print('end')
            return wrapper
@log('sdfdsf')
def bar():
    print('i am dedekinds')
bar()
